#!/usr/bin/env bash
# Proves the 3090 box can run GPU containers under Kiri, before any real
# service depends on it. Run as root (or a `kiri` group member with sudo for
# the host checks) on the box itself:
#
#   sudo infra/3090/tests/gpu-validate.sh
#
# It tests two independent routes and says which one works:
#   A  native CDI   the Kiri_GPU fork: --gpus all / device=N and Compose
#                   deploy.resources.reservations.devices
#   B  raw          --device /dev/nvidia* + --device-cgroup-rule + the driver
#                   files bind-mounted (bin/gpu-raw.sh), which upstream Kiri
#                   0.2.x accepts even though it refuses --gpus
# and, with whichever works, that limits, restart policy and healthchecks
# still behave with a GPU attached. Prints PASS/FAIL/SKIP lines and a verdict.
# Exit code: 0 if route A passes, 3 if only route B passes, 1 if neither,
# 4 if a GPU route works but limits/restart/healthchecks misbehave with it.
#
# Env: EXPECT_GPUS (default 2)  CUDA_IMAGE  TORCH_IMAGE  PROBE (box-probe path)
#      SKIP_TORCH=1 skips the ~3 GB PyTorch pull
set -uo pipefail
export DOCKER_HOST=${DOCKER_HOST:-unix:///run/kiri/docker.sock}
here=$(cd "$(dirname "$0")" && pwd)
RAW=${RAW:-$here/../bin/gpu-raw.sh}
EXPECT=${EXPECT_GPUS:-2}
# Pinned by digest (multi-arch index). Tags are for humans.
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:13.2.0-base-ubuntu24.04@sha256:36cccda4bebc3b0b1ebe1907ead8169cf144d45df890be871b36b304cf91145a}
TORCH_IMAGE=${TORCH_IMAGE:-pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime@sha256:924707fa828f821c9a82d987b405675df6369eee5ce368aece6757b025ffcde5}
PROBE=${PROBE:-/opt/box/bin/box-probe}
P=gv-test
pass=0; fail=0; skip=0
A_OK=1; B_OK=1; B_NEEDS_LDCONFIG=0; RULES_OK=1
ok()   { echo "PASS  $*"; pass=$((pass+1)); }
bad()  { echo "FAIL  $*"; fail=$((fail+1)); }
na()   { echo "SKIP  $*"; skip=$((skip+1)); }
note() { echo "      $*"; }
hdr()  { echo; echo "== $*"; }
cleanup() {
  docker rm -f $P-lim $P-hc $P-ro >/dev/null 2>&1
  [ -n "${cdir:-}" ] && { docker compose -p $P -f "$cdir/compose.yaml" down -t 1 >/dev/null 2>&1; rm -rf "$cdir"; }
}
trap cleanup EXIT
cleanup

# A small PyTorch check: every GPU computes a matmul that matches the CPU.
TORCH_PY='
import torch, sys
n = torch.cuda.device_count()
print("cuda", torch.cuda.is_available(), "count", n, "torch", torch.__version__, "cuda_rt", torch.version.cuda)
a = torch.randn(1024, 1024); b = torch.randn(1024, 1024); ref = a @ b
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    got = (a.cuda(i) @ b.cuda(i)).cpu()
    ok = torch.allclose(got, ref, rtol=1e-3, atol=1e-2)
    print(f"gpu{i} {p.name} {p.total_memory>>20}MiB sm_{p.major}{p.minor} matmul={ok}")
    if not ok: sys.exit(1)
if n > 1:
    print("p2p 0->1", torch.cuda.can_device_access_peer(0, 1))
print("TORCH_OK" if n > 0 else "NO_GPU")
'

count_gpus() { grep -c '^GPU [0-9]' <<<"$1"; }

# --------------------------------------------------------------------------
hdr "Host"
if ! command -v nvidia-smi >/dev/null || ! host_smi=$(nvidia-smi -L 2>&1); then
  bad "host nvidia-smi: ${host_smi:-not installed}"; echo; echo "VERDICT: driver not working on the host; fix that first."; exit 1
fi
n=$(count_gpus "$host_smi")
[ "$n" = "$EXPECT" ] && ok "host sees $n GPUs" || bad "host sees $n GPUs, expected $EXPECT"
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
note "driver $drv"; sed 's/^/      /' <<<"$host_smi"
gpu1_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 1 2>/dev/null | tr -d ' ')

# /dev/nvidia-uvm only appears once something loads nvidia_uvm.
[ -c /dev/nvidia-uvm ] || nvidia-modprobe -u -c=0 2>/dev/null
for d in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset $(seq -f /dev/nvidia%g 0 $((EXPECT-1))); do
  [ -c "$d" ] && ok "device node $d" || bad "device node $d missing"
done

spec=""
for f in /var/run/cdi/nvidia.yaml /etc/cdi/nvidia.yaml; do [ -r "$f" ] && { spec=$f; break; }; done
if [ -z "$spec" ]; then
  bad "no CDI spec; run: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
elif grep -q "libcuda.so.$drv" "$spec"; then
  ok "CDI spec $spec matches driver $drv"
else
  bad "CDI spec $spec is stale (no libcuda.so.$drv); rerun nvidia-ctk cdi generate"
fi
command -v nvidia-ctk >/dev/null && note "$(nvidia-ctk cdi list 2>/dev/null | tr '\n' ' ')"

if ! sv=$(docker version --format '{{.Server.Version}} {{.Server.Os}}' 2>&1); then
  bad "Kiri API at $DOCKER_HOST: $sv"; echo; echo "VERDICT: Kiri is not running."; exit 1
fi
ok "Kiri API at $DOCKER_HOST ($sv)"

hdr "Pull test images"
for img in "$CUDA_IMAGE" $([ -z "${SKIP_TORCH:-}" ] && echo "$TORCH_IMAGE"); do
  if perr=$(docker pull -q "$img" 2>&1); then ok "pulled ${img%%@*}"; else bad "cannot pull $img: $(tail -1 <<<"$perr")"; pullfail=1; fi
done
if [ -n "${pullfail:-}" ]; then
  echo; echo "VERDICT: could not pull the test images, so nothing about GPUs was tested. Fix the network/registry and rerun."; exit 1
fi

# --------------------------------------------------------------------------
hdr "A. Native CDI (Kiri_GPU fork)"
out=$(docker run --rm --gpus all "$CUDA_IMAGE" nvidia-smi -L 2>&1); rc=$?
if [ $rc -ne 0 ] && grep -qi "gpus\|device request\|DeviceRequests" <<<"$out"; then
  bad "--gpus refused: this Kiri has no GPU support (upstream build?): $(head -1 <<<"$out")"; A_OK=0
elif [ $rc -ne 0 ]; then
  bad "--gpus all nvidia-smi: $(head -3 <<<"$out")"; A_OK=0
else
  c=$(count_gpus "$out")
  [ "$c" = "$n" ] && ok "--gpus all: nvidia-smi sees $c GPUs" || { bad "--gpus all: nvidia-smi sees $c, host $n"; A_OK=0; }
fi
if [ $A_OK = 1 ]; then
  out=$(docker run --rm --gpus device=1 "$CUDA_IMAGE" nvidia-smi --query-gpu=uuid --format=csv,noheader 2>&1)
  [ "$(tr -d ' ' <<<"$out")" = "$gpu1_uuid" ] && ok "--gpus device=1 is exactly GPU 1" || { bad "--gpus device=1 gave: $out"; A_OK=0; }
  out=$(docker run --rm --device nvidia.com/gpu=0 "$CUDA_IMAGE" nvidia-smi -L 2>&1)
  [ "$(count_gpus "$out")" = 1 ] && ok "--device nvidia.com/gpu=0 gives one GPU" || { bad "--device nvidia.com/gpu=0: $out"; A_OK=0; }
  out=$(docker run --rm --read-only --gpus all "$CUDA_IMAGE" nvidia-smi -L 2>&1)
  [ "$(count_gpus "$out")" = "$n" ] && ok "--read-only rootfs + GPUs" || bad "--read-only + GPUs: $(head -2 <<<"$out")"
  if [ -z "${SKIP_TORCH:-}" ]; then
    out=$(docker run --rm --gpus all --shm-size 1g "$TORCH_IMAGE" python -c "$TORCH_PY" 2>&1)
    sed 's/^/      /' <<<"$out"
    grep -q TORCH_OK <<<"$out" && grep -q "count $n" <<<"$out" && ok "PyTorch CUDA matmul on $n GPUs (CDI)" || { bad "PyTorch over CDI"; A_OK=0; }
  else na "PyTorch over CDI (SKIP_TORCH)"; fi

  # Compose: the same reservation block the stacks use.
  cdir=$(mktemp -d)
  cat > "$cdir/compose.yaml" <<EOF
services:
  gpu:
    image: $CUDA_IMAGE
    command: ["sleep", "infinity"]
    init: true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
EOF
  if docker compose -p $P -f "$cdir/compose.yaml" up -d --wait >/dev/null 2>&1 \
     && out=$(docker compose -p $P -f "$cdir/compose.yaml" exec -T gpu nvidia-smi -L 2>&1) \
     && [ "$(count_gpus "$out")" = "$n" ]; then
    ok "Compose deploy.resources.reservations.devices: $n GPUs"
  else bad "Compose GPU reservation: $(head -2 <<<"${out:-up failed}")"; A_OK=0; fi
  docker compose -p $P -f "$cdir/compose.yaml" down -t 1 >/dev/null 2>&1; rm -rf "$cdir"; cdir=""
else
  na "rest of route A (--gpus did not work)"
fi

# --------------------------------------------------------------------------
hdr "B. Raw --device + bind mounts"
if ! raw=$(GPU_RAW_SPEC=$spec "$RAW" args 2>&1); then
  bad "gpu-raw.sh: $raw"; B_OK=0
else
  mapfile -t RAWARGS <<<"$raw"
  note "$(grep -c -- --device= <<<"$raw") devices, $(grep -c -- --volume= <<<"$raw") files from ${spec:-ldconfig}"
  out=$(docker run --rm "${RAWARGS[@]}" "$CUDA_IMAGE" nvidia-smi -L 2>&1); rc=$?
  if [ $rc -eq 0 ] && [ "$(count_gpus "$out")" = "$n" ]; then
    ok "raw: nvidia-smi sees $n GPUs"
  else
    # Some images only find the soname links after an ldconfig run.
    out2=$(docker run --rm "${RAWARGS[@]}" "$CUDA_IMAGE" sh -c 'ldconfig && nvidia-smi -L' 2>&1)
    if [ "$(count_gpus "$out2")" = "$n" ]; then
      ok "raw: nvidia-smi sees $n GPUs, but only after ldconfig in the container"; B_NEEDS_LDCONFIG=1
    else bad "raw nvidia-smi: $(head -3 <<<"$out")"; B_OK=0; fi
  fi
  if [ $B_OK = 1 ] && [ -z "${SKIP_TORCH:-}" ]; then
    pre=""; [ $B_NEEDS_LDCONFIG = 1 ] && pre="ldconfig && "
    out=$(docker run --rm "${RAWARGS[@]}" --shm-size 1g "$TORCH_IMAGE" sh -c "${pre}python -c '$TORCH_PY'" 2>&1)
    sed 's/^/      /' <<<"$out"
    grep -q TORCH_OK <<<"$out" && grep -q "count $n" <<<"$out" && ok "PyTorch CUDA matmul on $n GPUs (raw)" || { bad "PyTorch over raw devices"; B_OK=0; }
  elif [ $B_OK = 1 ]; then na "PyTorch raw (SKIP_TORCH)"; fi
  # Only GPU 0 is visible when only /dev/nvidia0 is passed.
  if [ $B_OK = 1 ] && raw0=$(GPU_RAW_SPEC=$spec GPU_RAW_GPUS=0 "$RAW" args 2>&1); then
    mapfile -t R0 <<<"$raw0"
    out=$(docker run --rm "${R0[@]}" "$CUDA_IMAGE" sh -c "${pre:-}nvidia-smi -L" 2>&1)
    [ "$(count_gpus "$out")" = 1 ] && ok "raw: GPU_RAW_GPUS=0 gives one GPU" || bad "raw single GPU: $(head -2 <<<"$out")"
  fi
fi

# --------------------------------------------------------------------------
hdr "Service rules with a GPU attached"
if [ $A_OK = 1 ]; then GPUARGS=(--gpus all); route=A
elif [ $B_OK = 1 ]; then GPUARGS=("${RAWARGS[@]}"); route=B
else GPUARGS=(); route=""; fi
if [ -n "$route" ]; then
  note "using route $route"
  if ! err=$(docker run -d --name $P-lim "${GPUARGS[@]}" --restart unless-stopped --memory 2g --cpus 1.5 --pids-limit 256 \
      "$CUDA_IMAGE" sleep infinity 2>&1); then
    bad "GPU container with memory/CPU/pids limits did not start: $(head -1 <<<"$err")"; RULES_OK=0
  else
  mem=$(docker exec $P-lim cat /sys/fs/cgroup/memory.max 2>/dev/null)
  cpu=$(docker exec $P-lim cat /sys/fs/cgroup/cpu.max 2>/dev/null)
  pids=$(docker exec $P-lim cat /sys/fs/cgroup/pids.max 2>/dev/null)
  pol=$(docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' $P-lim)
  [ "$mem" = 2147483648 ] && ok "memory.max=$mem" || { bad "memory.max=$mem"; RULES_OK=0; }
  [ "${cpu%% *}" = 150000 ] && ok "cpu.max=$cpu" || { bad "cpu.max=$cpu"; RULES_OK=0; }
  [ "$pids" = 256 ] && ok "pids.max=$pids" || { bad "pids.max=$pids"; RULES_OK=0; }
  [ "$pol" = unless-stopped ] && ok "restart policy $pol" || { bad "restart policy=$pol"; RULES_OK=0; }
  out=$(docker exec $P-lim sh -c "${pre:-}nvidia-smi -L" 2>&1)
  [ "$(count_gpus "$out")" -ge 1 ] && ok "GPU usable inside limited container" || { bad "GPU inside limited container: $out"; RULES_OK=0; }
  # The container's own processes can't escape into other devices.
  disk=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk"{print $1; exit}')
  if [ -n "$disk" ]; then
    docker exec $P-lim sh -c "mknod /tmp/d b $(stat -c '0x%t 0x%T' "$disk" | xargs printf '%d %d') 2>/dev/null; head -c1 /tmp/d" >/dev/null 2>&1 \
      && bad "container can read host disk $disk" || ok "host disk $disk not readable from the container"
  fi
  fi

  if [ -x "$PROBE" ]; then
    docker run -d --name $P-hc "${GPUARGS[@]}" -v "$PROBE:/opt/box/bin/box-probe:ro" \
      --health-cmd '/opt/box/bin/box-probe tcp://127.0.0.1:18998' --health-interval 2s --health-retries 2 \
      "$CUDA_IMAGE" sh -c 'sleep 8; exec sleep infinity' >/dev/null
    sleep 7; h=$(docker inspect -f '{{.State.Health.Status}}' $P-hc)
    [ "$h" = unhealthy ] && ok "healthcheck goes unhealthy when the port is closed" || { bad "health=$h (want unhealthy)"; RULES_OK=0; }
    if [ -n "${SKIP_TORCH:-}" ]; then na "healthy probe (needs python in TORCH_IMAGE)"; else
      docker rm -f $P-hc >/dev/null
      docker run -d --name $P-hc "${GPUARGS[@]}" --network host -v "$PROBE:/opt/box/bin/box-probe:ro" \
        --health-cmd '/opt/box/bin/box-probe http://127.0.0.1:18998/' --health-interval 2s --health-retries 2 \
        "$TORCH_IMAGE" python -m http.server 18998 --bind 127.0.0.1 >/dev/null
      for _ in $(seq 20); do h=$(docker inspect -f '{{.State.Health.Status}}' $P-hc); [ "$h" = healthy ] && break; sleep 1; done
      [ "$h" = healthy ] && ok "box-probe healthcheck healthy (host network)" || { bad "health=$h (want healthy)"; RULES_OK=0; }
    fi
  else bad "$PROBE missing (install.sh builds it)"; RULES_OK=0; fi
else
  na "no GPU route works"
fi

# --------------------------------------------------------------------------
echo
echo "== $pass passed, $fail failed, $skip skipped"
if [ $RULES_OK = 0 ] && { [ $A_OK = 1 ] || [ $B_OK = 1 ]; }; then
  echo "VERDICT: FAIL. GPUs reach containers (route A=$A_OK, B=$B_OK), but limits, restart policy or healthchecks"
  echo "         did not behave with a GPU attached. Don't run services yet; send this output to Claude."
  exit 4
elif [ $A_OK = 1 ]; then
  echo "VERDICT: PASS (native CDI). Use deploy.resources.reservations.devices in compose files."
  [ $B_OK = 1 ] && echo "         Raw --device route also works (fallback for upstream Kiri)."
  exit 0
elif [ $B_OK = 1 ]; then
  echo "VERDICT: PASS (raw only). Native CDI failed; use the generated override:"
  echo "         bin/gpu-raw.sh compose <service> > /box/config/kiri/gpu-raw.override.yaml"
  [ $B_NEEDS_LDCONFIG = 1 ] && echo "         Note: this image needs 'ldconfig' before the app starts."
  exit 3
else
  echo "VERDICT: FAIL. Kiri cannot give containers the GPUs on this box. Send this output to Claude."
  exit 1
fi
