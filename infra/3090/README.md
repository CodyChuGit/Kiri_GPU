# The 3090 box: GPU services under Kiri

The TRX40 / Threadripper 3970X box with 2x RTX 3090 runs its heavy AI work as
Kiri containers. Kiri serves the Docker Engine API on `/run/kiri/docker.sock`,
so every service is a plain `compose.yaml` driven by `docker compose`. Same
pattern as `CodyChuGit/NAS` `infra/nas/kiri/`.

The host keeps only the NVIDIA driver, Kiri, `nvidia-ctk`, Tailscale and SSH.

```
infra/3090/
  install.sh                       one command: build + install Kiri (this fork), units, probe, stacks, validate
  bin/gpu-raw.sh                   device/mount list for the raw --device route (args | compose <svc> | list)
  bin/kiri-compose                 `docker compose` for a stack dir, adds the raw-GPU override when asked
  kiri/systemd/
    kiri.service.d/10-3090.conf    restart Kiri if it dies, start after the CDI spec is fresh
    kiri-stack@.service            brings /box/config/kiri/stacks/<name> up at boot
    box-cdi-refresh.service        `nvidia-ctk cdi generate` + nvidia-modprobe at every boot
  kiri/stacks/gpu-app/             template GPU service (compose.yaml + .env.example)
  kiri/stacks/maya-deepseek/ maya-glm/ xolotl-qwen/ avatar/ vulkan-test/
                                   the box's kiri-sched GPU runtimes (run by kiri-run@, never by hand)
  kiri/sched.toml.example          /etc/kiri-sched/sched.toml for those runtimes (home = maya-deepseek)
  tests/gpu-validate.sh            PASS/FAIL proof that containers get working GPUs
  tools/box-probe/                 static Rust healthcheck (copy of the NAS repo's nas-probe)
```

## Install

On the box, as your admin user, from a checkout of this repo:

```sh
sudo infra/3090/install.sh              # add --skip-torch to skip the 3 GB PyTorch test image
```

It needs the NVIDIA driver working first (`nvidia-smi` lists both cards).
`box-setup/setup-server.sh` from the project does that. It is safe to re-run
and never overwrites a stack's `.env` or state.

## Two ways to give a container the GPUs

Upstream Kiri 0.2.x refuses `--gpus`. This fork (Kiri_GPU, PR #1) adds CDI:

| Route | How | Works on | Compose |
|---|---|---|---|
| A. native CDI | `--gpus all`, `--gpus device=1`, `--device nvidia.com/gpu=0` | this fork | `deploy.resources.reservations.devices` (the template's default) |
| B. raw | `--device /dev/nvidia*` + `--device-cgroup-rule 'c 195:* rwm'` + driver files bind-mounted read-only | upstream Kiri too | override from `gpu-raw.sh compose app` |

Route B's file list comes from the CDI spec `nvidia-ctk` writes (or `ldconfig`
if there is none). Without CDI hooks nothing creates `libcuda.so.1` and friends
in the container, so `gpu-raw.sh` bind-mounts each library under every name the
host links to it by. The override has the driver version in its file names:
regenerate it after every driver update.

`tests/gpu-validate.sh` tries both and ends with a verdict:

- `VERDICT: PASS (native CDI)` (exit 0): use the template as is.
- `VERDICT: PASS (raw only)` (exit 3): `install.sh` writes
  `/box/config/kiri/gpu-raw.override.yaml`; `touch gpu-raw` in a stack dir to use it.
- `VERDICT: FAIL` (exit 1 or 4): Kiri can't do GPUs reliably on this box. Don't
  switch anything yet: send the log (`/box/config/kiri/gpu-validate.log`).

What it checks: host driver and device nodes, CDI spec freshness, both routes
with `nvidia-smi` (all GPUs, one GPU, the right UUID), a PyTorch matmul on every
GPU checked against the CPU, Compose reservations, `--read-only` with GPUs, and
with a GPU attached: memory/CPU/pids limits in the cgroup, restart policy, no
host disks in the container, and a box-probe healthcheck going unhealthy and healthy.

## Adding a service

```sh
sudo cp -r /box/config/kiri/stacks/gpu-app /box/config/kiri/stacks/<name>
sudoedit /box/config/kiri/stacks/<name>/.env      # STACK_NAME=<name>, APP_IMAGE=repo:tag@sha256:..., APP_PORT
sudo install -d -o $USER /box/runtimes/<name>/state /models/<name> /scratch/<name>
sudo systemctl enable --now kiri-stack@<name>
cd /box/config/kiri/stacks/<name> && kiri-compose ps     # (healthy)
```

Every service gets `restart: unless-stopped`, a healthcheck (box-probe, since
many images have no curl), CPU/memory/pids limits, persistent bind mounts, and
an image pinned by digest. Secrets live only in `.env` (mode 600, git-ignored).

GPU stacks are different: they belong to the GPU scheduler (next section),
not to `kiri-stack@`. Light (non-GPU) services stay on `kiri-stack@`.

| Path in the container | On the host | Backed up |
|---|---|---|
| `/state` | `/box/runtimes/<name>/state` | yes (nightly, with /box) |
| `/models/app` | `/models/<name>` (local 2TB) | no, masters on the NAS |
| `/models/shared` (ro), `/models/hf` | `/models/shared`, `/models/hf` | no |
| `/scratch` | `/scratch/<name>` | never |
| `/exchange` | `/box/exchange` | yes |

## GPU stacks and the scheduler

`kiri-sched` (crates/kiri-sched, installed by `install.sh`) keeps one runtime
per card on the two 3090s (a two-card runtime holds both; two one-card
runtimes run side by side), switches between them on demand (from the front door
on `127.0.0.1:7070` or `kiri-sched use <runtime>`), restarts a crashed runtime,
and brings the home runtime back after a reboot. With
`/etc/kiri/gpu-lease-required` present the Kiri daemon refuses any GPU
container the scheduler did not start. So a GPU stack:

- is listed as a `[[runtime]]` in `/etc/kiri-sched/sched.toml` and runs under
  `kiri-run@<stack>` (never `systemctl enable kiri-stack@<stack>`);
- has `restart: "no"` (the scheduler restarts it, with backoff) and the label
  `dev.kiri.sched.runtime: <runtime>`;
- uses `ipc: host` when it spreads one model over both cards (NCCL shm);
- if it needs one card: has `cards = 1` in its `[[runtime]]` and names its
  card with `device_ids: ["${KIRI_SCHED_GPUS:?set by kiri-sched}"]` instead of
  `count: all` (the scheduler writes `/run/kiri-sched/cards/<stack>.env`
  before each start; the lease refuses any other card). See
  `kiri/stacks/card-test-a`.

Containers start at OOM score 0 (or the stack's `oom_score_adj`), not with the
daemon's `OOMScoreAdjust=`, so a runaway engine is killed before Kiri, sshd or
Tailscale.

```sh
kiri-sched status                     # who holds the GPUs, queue, faults
kiri-sched use maya-deepseek [min]    # switch now (and keep it that long)
kiri-sched hold <runtime> [min] --by NAME  # nothing evicts it until release
kiri-sched release <runtime>          # end the holds on it (one hold per card)
kiri-sched reset                      # clear a fault
```

## Network

Stacks use the host network namespace, so Kiri writes no NAT or firewall rules
and the host firewall stays the only one. Bind apps to `127.0.0.1` and publish
with `tailscale serve`, or to `10.25.0.11` when the NAS (`10.25.0.1`, direct
25GbE, `10.25.0.0/24`) or another LAN machine must reach them. Nothing listens on
a public interface. NAS shares belong on the host (an fstab NFS/SMB mount such as
`/mnt/nas`), bind-mounted read-only into the stacks that need them.

## Day to day

```sh
export DOCKER_HOST=unix:///run/kiri/docker.sock
cd /box/config/kiri/stacks/<name>
kiri-compose ps; kiri-compose logs -f app
sudo systemctl reload kiri-stack@<name>    # apply compose/.env changes
```

After an NVIDIA driver update: reboot (box-cdi-refresh regenerates the spec),
then `sudo infra/3090/tests/gpu-validate.sh` (stop kiri-sched and set the lease
marker aside first; see install.sh step 7). After a Kiri update: re-run
`sudo infra/3090/install.sh`.

## What was tested where

- In a cloud container (no GPU): Kiri built from this fork, both routes against
  fake NVIDIA device nodes and a fake driver library: `--gpus all`/`device=1`/
  `nvidia.com/gpu=0`, `--read-only`, Compose reservations, the raw override
  (devices, cgroup rule, soname mounts, CDI reservation dropped), the template
  stack coming up `(healthy)` via box-probe. The `.deb` builds.
- Needs the real box: real `nvidia-smi` and CUDA through the driver, the
  PyTorch check, cgroup limits (the cloud sandbox has no memory/cpu/pids
  controllers), `install.sh` under systemd, `nvidia-modprobe`, the 25GbE link.
