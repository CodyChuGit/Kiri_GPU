# kiri-sched

The GPU scheduler and supervisor for a Kiri GPU server: one front door for
every API call and UI, one runtime per GPU card at a time (runtimes that
need one card run side by side), and a Kiri daemon that refuses GPUs to
anything the scheduler did not grant.

```
5090 rig / browser ─ Tailscale ─ Caddy ─ kiri-sched :7070 ─ the runtime holding the GPUs
                                              │
                          systemctl start/stop kiri-run@<stack>
                                              │
                        Kiri daemon ── asks /run/kiri-sched/lease.sock before any GPU start
```

## What it does

- **Front door.** `POST /v1/chat/completions`, `/v1/messages` and the other
  `/v1/*` calls go to the runtime that serves the body's `"model"`;
  `/rt/<runtime>/…` reaches a runtime's own UI or API (WebSockets too).
  Responses stream through. The client's token is never passed on.
- **One runtime per card.** Each runtime needs `cards` GPUs (all of them by
  default). Runtimes whose cards add up to no more than `gpus` run side by
  side, each on its own cards; a runtime that needs more cards than are free
  evicts only as many runtimes as it must (idle ones and one-card ones
  first). A request for a ready runtime goes straight through, up to its
  `concurrency`. A request for another one waits (the connection is held)
  while the scheduler switches, or is refused with `503`, `Retry-After`,
  `X-Box-State`, `X-Box-Queue-Position` and the reason in an error body that
  OpenAI and Anthropic clients read.
- **When to switch.** An idle runtime gives way at once. A busy one gives way
  to interactive work only after `dwell_secs`; agent work waits for an idle
  box until it has waited `agent_aging_secs`; batch requests never move the
  box but may take a free card. A hold (`kiri-sched hold`) wins over
  everything: its runtime evicts what is in the way, nothing evicts it, and
  only runtimes that cannot fit beside it are refused. A runtime with
  `home_after_idle_secs` gives its cards back once it has had no request,
  nothing waiting and no hold for that long: to the home runtime if home
  then fits, else they stay free. A runtime that crashes comes back on free
  cards after its backoff.
- **How to switch.** One switch at a time. Drain what it evicts (in-flight
  requests get `drain_secs`), stop it and any GPU runtime running without
  the scheduler knowing, wait until `nvidia-smi` shows the freed and the
  target cards at or under `vram_free_mib` (other cards keep working), write
  `cards_dir/<stack>.env` (`KIRI_SCHED_GPUS=1`), then start and wait for the
  readiness URL. A runtime that fails to start is stopped again. A stop
  that fails, or memory that stays in use, is a **fault**: no switch runs
  until `kiri-sched reset` (ready runtimes keep serving).
- **Lease.** Kiri asks before giving any container NVIDIA GPUs. Only a
  runtime the scheduler is starting or running gets a yes, and a runtime
  that holds some of the cards only for its own (`nvidia.com/gpu=<index>`;
  `all`, `count:N` and UUIDs are refused), so a stray `kiri run --gpus all`
  or a container restarting itself cannot land on cards another runtime
  holds. With `/etc/kiri/gpu-lease-required` present, Kiri
  refuses GPU starts while the scheduler is down.
- **Supervision.** At startup it keeps every GPU runtime that is already
  answering on cards it can keep (all of them, or the ones saved for it) and
  stops the rest. A ready runtime that stops answering for `unhealthy_secs`
  is stopped and comes back on its cards if they are still free. Light services are
  started at boot and restarted when they stop answering. Every restart backs
  off (5 s doubling to 5 min), and `fence_after` failures in
  `fence_window_secs` fence a runtime or service instead of looping.
- **State and audit.** Holds, fences, each runtime's cards and the last good runtime are saved
  atomically (write, fsync, rename) and come back after a restart or reboot.
  Every switch, failure, restart, fence, lease answer and refusal is one
  JSON line on stderr (journald).

## Use

```
kiri-sched status                  # JSON: gpus (who holds each card), runtimes, phase, hold, waiting, services
kiri-sched use comfyui             # switch now; keep it for the dwell time once it is ready
kiri-sched use comfyui 45 --by lab # … for 45 minutes after ready, held as admin-socket:lab
kiri-sched hold desktop            # hold until release
kiri-sched release desktop         # name the held runtime (or be its named owner, or --force)
kiri-sched unfence maya-glm
kiri-sched reset                   # after a fault, once the GPUs were checked
kiri-sched hash-token < token      # for [[client]] token_sha256
```

A `use` or `hold` of another runtime while someone else's hold lasts is
refused with 409 and who holds it, instead of quietly replacing it; pass
`--force` to take over. `use` answers with the runtime's state (`ready`,
or `starting`/`stopped` while it switches); its time only starts once the
runtime is ready, and a `use` whose runtime is not ready within its start
timeout (plus the switch's own timeouts) is given up. Scripts should name
themselves with `--by NAME` (or `KIRI_SCHED_BY=NAME`).

Every request needs a client token (`Authorization: Bearer …` or
`x-api-key`). With one, `GET /box/status`, `GET /box/events` (server-sent
events) and `GET /v1/models` (every model with its runtime's state) work for
any client; the `POST /box/*` actions need an `admin` client or the admin
socket (the CLI uses the socket).

## Install (the 3090 box)

Before installing, every GPU runtime must run under its `kiri-run@<stack>`
unit with the `restart: "no"` stacks described below: the scheduler only
adopts a runtime that is running in its unit and answering, and treats GPU
memory used by anything else as a fault (tested on the box Nov 23: with
rt-maya running outside a unit it faulted and started nothing).

See `contrib/`: `kiri-sched.service` (sandboxed, `Type=notify`, watchdog),
`kiri-run@.service` (a stack whose stop really stops it), the polkit rule
that lets the `kiri-sched` user start and stop only `kiri-run@` units, and
`sched.example.toml`. In GPU stacks set `restart: "no"`, and give the
containers no other way onto the GPUs than the scheduler.

Caddy:

```
box.example.ts.net {
	reverse_proxy 127.0.0.1:7070 {
		flush_interval -1
	}
}
```

## Not in this phase

Batch jobs (submit, night window, `box split`, checkpoint and resume),
per-API-key fair share, SSE keep-alive comments while a streaming request
waits, the 5090 rig agent (`boxlet`), the MCP tool and the dashboard view are
the later phases of the plan.
