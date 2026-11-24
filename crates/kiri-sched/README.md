# kiri-sched

The GPU scheduler and supervisor for a Kiri GPU server: one front door for
every API call and UI, one runtime on the GPUs at a time, and a Kiri daemon
that refuses GPUs to anything the scheduler did not grant.

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
- **One runtime at a time.** A request for the runtime on the GPUs goes
  straight through, up to its `concurrency`. A request for another one waits
  (the connection is held) while the scheduler switches, or is refused with
  `503`, `Retry-After`, `X-Box-State`, `X-Box-Queue-Position` and the reason
  in an error body that OpenAI and Anthropic clients read.
- **When to switch.** An idle runtime gives way at once. A busy one gives way
  to interactive work only after `dwell_secs`; agent work waits for an idle
  box until it has waited `agent_aging_secs`; batch requests never move the
  box. A hold (`kiri-sched hold`) wins over everything and refuses the rest.
  A runtime with `home_after_idle_secs` goes back to the home runtime once it
  has had no request, nothing waiting and no hold for that long.
- **How to switch.** Drain (in-flight requests get `drain_secs`), stop every
  other GPU runtime, wait until `nvidia-smi` shows every GPU at or under
  `vram_free_mib`, then start and wait for the readiness URL. A runtime that
  fails to start is stopped again and the GPUs go back to the last good one.
  A stop that fails, or memory that stays in use, is a **fault**: nothing
  starts until `kiri-sched reset`.
- **Lease.** Kiri asks before giving any container NVIDIA GPUs. Only the
  runtime the scheduler is starting or running gets a yes, so a stray
  `kiri run --gpus all` or a container restarting itself cannot land on cards
  another runtime holds. With `/etc/kiri/gpu-lease-required` present, Kiri
  refuses GPU starts while the scheduler is down.
- **Supervision.** At startup it stops any GPU runtime it does not want and
  adopts the one it does if it is already answering. The runtime on the GPUs
  is restarted if it stops answering for `unhealthy_secs`. Light services are
  started at boot and restarted when they stop answering. Every restart backs
  off (5 s doubling to 5 min), and `fence_after` failures in
  `fence_window_secs` fence a runtime or service instead of looping.
- **State and audit.** Holds, fences and the last good runtime are saved
  atomically (write, fsync, rename) and come back after a restart or reboot.
  Every switch, failure, restart, fence, lease answer and refusal is one
  JSON line on stderr (journald).

## Use

```
kiri-sched status                  # JSON: phase, hold, runtimes, waiting, services
kiri-sched use comfyui             # switch now and keep it for the dwell time
kiri-sched use comfyui 45          # … for 45 minutes
kiri-sched hold desktop            # hold until release
kiri-sched release
kiri-sched unfence maya-glm
kiri-sched reset                   # after a fault, once the GPUs were checked
kiri-sched hash-token < token      # for [[client]] token_sha256
```

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
