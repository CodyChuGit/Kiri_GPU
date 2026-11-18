#!/usr/bin/env bash
# One command to put GPU-capable Kiri on the 3090 box and prove it works:
#
#   sudo infra/3090/install.sh                 # build + install + validate
#   sudo infra/3090/install.sh --enable gpu-app  # ...and start that stack
#
# Options:
#   --skip-build     keep the installed Kiri (no rebuild from this checkout)
#   --no-validate    don't run tests/gpu-validate.sh at the end
#   --enable NAME    enable and start kiri-stack@NAME after validation passes
#   --skip-torch     validation without the ~3 GB PyTorch image
#
# What it does (safe to re-run; it never overwrites a stack's .env or state):
#   1. Checks the NVIDIA driver and installs nvidia-container-toolkit (for nvidia-ctk only).
#   2. Builds Kiri from THIS checkout (the Kiri_GPU fork with CDI GPU support)
#      as static musl binaries, packages the full `kiri` .deb (docker CLI +
#      Compose bundled, versions pinned in crates/kiri/src/cli_pins.rs) and
#      installs it. It replaces an upstream kiri/kiri-lite package.
#   3. Installs the systemd pieces: kiri.service drop-in, kiri-stack@.service,
#      box-cdi-refresh.service (regenerates /etc/cdi/nvidia.yaml every boot).
#   4. Builds box-probe (static healthcheck) and installs it with kiri-compose
#      and gpu-raw.sh in /opt/box/bin.
#   5. Creates /box/config/kiri/stacks, /box/runtimes, /models, /scratch dirs and
#      copies the stacks from this repo (existing ones are left alone).
#   6. Runs tests/gpu-validate.sh and, if only the raw --device route works,
#      writes /box/config/kiri/gpu-raw.override.yaml.
#
# Prereq: Ubuntu 26.04 with the NVIDIA driver loaded (box-setup/setup-server.sh).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
BUILD=1 VALIDATE=1 ENABLE="" SKIP_TORCH=""
while [ $# -gt 0 ]; do
  case $1 in
    --skip-build) BUILD=0 ;;
    --no-validate) VALIDATE=0 ;;
    --enable) ENABLE=${2:?--enable needs a stack name}; shift ;;
    --skip-torch) SKIP_TORCH=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m!! %s\033[0m\n' "$*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die "run with sudo"
ADMIN=${SUDO_USER:-}
[ -n "$ADMIN" ] && [ "$ADMIN" != root ] || die "run with sudo from your admin account (SUDO_USER is the account that gets Kiri access)"
as_admin() { sudo -u "$ADMIN" -H bash -lc "$*"; }
[ "$(uname -m)" = x86_64 ] || die "this box is x86_64; got $(uname -m)"

# ---------------------------------------------------------------------------
say "1. NVIDIA driver and nvidia-ctk"
nvidia-smi -L || die "nvidia-smi does not work. Install the driver (box-setup/setup-server.sh) and reboot first."
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update && apt-get install -y nvidia-container-toolkit
fi
nvidia-ctk --version | head -1
# Kiri does the CDI hooks itself; no nvidia-container-runtime config is needed or wanted.

# ---------------------------------------------------------------------------
if [ $BUILD = 1 ]; then
  say "2. Build Kiri (GPU fork) from $repo"
  apt-get install -y build-essential musl-tools pkg-config curl ca-certificates dpkg-dev
  as_admin 'command -v cargo >/dev/null || curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'
  as_admin 'rustup target add x86_64-unknown-linux-musl'
  out=$repo/target/box-3090
  chown "$ADMIN" "$repo/target" 2>/dev/null || true
  as_admin "cd '$repo' && CC_x86_64_unknown_linux_musl=musl-gcc cargo build --release --locked \
    --target x86_64-unknown-linux-musl -p kiri -p kiri-init"
  bindir=$repo/target/x86_64-unknown-linux-musl/release
  as_admin "cd '$repo' && bash packaging/linux/fetch-cli.sh '$out/cli' && rm -rf '$out/pkg' && bash packaging/linux/build.sh '$bindir' '$out/pkg' x86_64 '$out/cli'"
  deb=$(ls "$out"/pkg/kiri_*_amd64.deb | head -1)
  [ -f "$deb" ] || die "no kiri .deb was built"
  if [ -x /usr/bin/kiri ] && cmp -s /usr/bin/kiri "$bindir/kiri" && cmp -s /usr/bin/kiri-init "$bindir/kiri-init"; then
    echo "Installed Kiri already matches this build."
  else
    dpkg -s kiri-lite >/dev/null 2>&1 && apt-get remove -y kiri-lite
    apt-get install -y --allow-downgrades "$deb"
    INSTALLED=1
  fi
else
  say "2. Keeping the installed Kiri (--skip-build)"
  command -v kiri >/dev/null || die "no kiri installed"
fi
kiri --version 2>/dev/null || true
# Members of `kiri` are root-equivalent: only the admin account.
kiri host-access "$ADMIN"

# ---------------------------------------------------------------------------
say "3. systemd units"
install -D -m644 "$here/kiri/systemd/kiri.service.d/10-3090.conf" /etc/systemd/system/kiri.service.d/10-3090.conf
install -m644 "$here/kiri/systemd/kiri-stack@.service" /etc/systemd/system/kiri-stack@.service
install -m644 "$here/kiri/systemd/box-cdi-refresh.service" /etc/systemd/system/box-cdi-refresh.service
systemctl daemon-reload
systemctl enable box-cdi-refresh.service
systemctl restart box-cdi-refresh.service
nvidia-ctk cdi list
systemctl enable kiri.service
if [ -n "${INSTALLED:-}" ]; then systemctl restart kiri.service; else systemctl start kiri.service; fi
for _ in $(seq 30); do [ -S /run/kiri/docker.sock ] && break; sleep 1; done
[ -S /run/kiri/docker.sock ] || die "Kiri socket did not appear; journalctl -u kiri -n 50"

# ---------------------------------------------------------------------------
say "4. box-probe, kiri-compose, gpu-raw.sh -> /opt/box/bin"
as_admin "cd '$here/tools/box-probe' && cargo build --release --locked --target x86_64-unknown-linux-musl"
install -D -m755 "$here/tools/box-probe/target/x86_64-unknown-linux-musl/release/box-probe" /opt/box/bin/box-probe
install -m755 "$here/bin/kiri-compose" "$here/bin/gpu-raw.sh" /opt/box/bin/
printf '%s\n' 'export PATH=$PATH:/opt/box/bin' 'export DOCKER_HOST=unix:///run/kiri/docker.sock' > /etc/profile.d/box-kiri.sh
/opt/box/bin/box-probe 2>&1 | head -1 || true

# ---------------------------------------------------------------------------
say "5. /box, /models, /scratch and stacks"
mountpoint -q /models || echo "note: /models is not a separate mount yet (setup-server.sh formats the 2TB). Using a plain directory."
install -d -m755 /box /box/config /box/config/kiri /box/config/kiri/stacks /box/runtimes /box/exchange /scratch \
  /models /models/shared /models/hf /models/torch
for s in "$here"/kiri/stacks/*/; do
  name=$(basename "$s")
  dst=/box/config/kiri/stacks/$name
  if [ -d "$dst" ]; then
    # Update the compose file, never the .env.
    install -m644 "$s/compose.yaml" "$dst/compose.yaml"
    install -m644 "$s/.env.example" "$dst/.env.example"
  else
    cp -r "$s" "$dst"
  fi
  if [ ! -f "$dst/.env" ]; then
    install -m600 "$dst/.env.example" "$dst/.env"
    echo "Created $dst/.env from the example: edit it (image digest, port) before enabling $name."
  fi
  chmod 600 "$dst/.env"
  install -d -m750 -o "$ADMIN" "/box/runtimes/$name" "/box/runtimes/$name/state" "/models/$name" "/scratch/$name"
done
chown -R "$ADMIN" /box/exchange /models/shared /models/hf /models/torch

# ---------------------------------------------------------------------------
rc=0
if [ $VALIDATE = 1 ]; then
  say "6. GPU validation"
  SKIP_TORCH=$SKIP_TORCH PROBE=/opt/box/bin/box-probe RAW=/opt/box/bin/gpu-raw.sh \
    "$here/tests/gpu-validate.sh" | tee /box/config/kiri/gpu-validate.log || rc=${PIPESTATUS[0]}
  rc=${rc:-0}
  if [ "$rc" = 3 ]; then
    /opt/box/bin/gpu-raw.sh compose app > /box/config/kiri/gpu-raw.override.yaml
    echo "Wrote /box/config/kiri/gpu-raw.override.yaml. In each GPU stack dir: touch gpu-raw"
  fi
  echo "Log: /box/config/kiri/gpu-validate.log"
fi

if [ -n "$ENABLE" ]; then
  if [ "$rc" = 0 ] || [ "$rc" = 3 ]; then
    grep -q 'REPLACE_ME' "/box/config/kiri/stacks/$ENABLE/.env" && die "fill in /box/config/kiri/stacks/$ENABLE/.env first"
    systemctl enable --now "kiri-stack@$ENABLE"
    (cd "/box/config/kiri/stacks/$ENABLE" && /opt/box/bin/kiri-compose ps)
  else
    die "not enabling $ENABLE: GPU validation failed"
  fi
fi

say "Done"
case $rc in
  0) echo "GPUs work through Kiri's native CDI path." ;;
  3) echo "GPUs work only through the raw --device path (override written)." ;;
  *) echo "GPU validation FAILED. See /box/config/kiri/gpu-validate.log." ; exit 1 ;;
esac
