//! The scheduler's decisions, with no I/O and no clock of its own: every
//! call takes `now` (Unix milliseconds), so tests drive it with a fake
//! clock and check its invariants over long random runs.
//!
//! The invariants, which nothing here may break:
//! 1. At most one runtime is allowed on the GPUs (`lease_holder`), and the
//!    switch driver only starts a runtime after the scheduler moved to
//!    `Starting` for it, which needs the previous one stopped and the GPUs
//!    seen empty.
//! 2. Nothing switches the box away from a hold while it lasts.
//! 3. Waiting work is bounded (`max_waiting`); more is refused, not queued.
//! 4. A runtime that keeps failing is fenced, never restarted in a loop.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::{Class, Config};

/// Where the GPUs stand.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "lowercase")]
pub enum Phase {
    /// No runtime on the GPUs.
    Idle,
    /// `from` finishes what it has in flight; new work for it waits.
    Draining {
        from: String,
        to: String,
    },
    /// `from` (if any) is being stopped and the GPUs checked empty.
    Stopping {
        from: Option<String>,
        to: String,
    },
    /// `rt` is starting: the only runtime that may get the GPUs.
    Starting {
        rt: String,
    },
    Ready {
        rt: String,
        since: u64,
    },
    /// The GPUs could not be shown empty (a stop failed, memory stayed in
    /// use). Nothing starts until someone looks: `kiri-sched reset`.
    Fault {
        reason: String,
    },
}

impl Phase {
    pub fn name(&self) -> &'static str {
        match self {
            Phase::Idle => "idle",
            Phase::Draining { .. } => "draining",
            Phase::Stopping { .. } => "stopping",
            Phase::Starting { .. } => "starting",
            Phase::Ready { .. } => "ready",
            Phase::Fault { .. } => "fault",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hold {
    pub rt: String,
    /// Unix ms; `None` holds until released.
    pub until: Option<u64>,
    pub by: String,
}

/// What survives a restart of the scheduler (and of the box).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Saved {
    #[serde(default)]
    pub hold: Option<Hold>,
    #[serde(default)]
    pub fenced: BTreeMap<String, u64>,
    #[serde(default)]
    pub last_good: Option<String>,
}

/// The answer to a request for a runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Admit {
    /// Its runtime is ready: go (subject to the runtime's concurrency).
    Go,
    /// Waiting for its runtime; poll with the id.
    Wait(u64),
    /// Not now: HTTP status, reason, and seconds until it is worth retrying.
    Refuse {
        status: u16,
        reason: String,
        retry_after: u64,
    },
}

#[derive(Debug, Clone)]
struct Waiter {
    id: u64,
    rt: String,
    class: Class,
    since: u64,
}

#[derive(Debug)]
pub struct Core {
    cfg: Arc<Config>,
    phase: Phase,
    inflight: BTreeMap<String, usize>,
    waiting: Vec<Waiter>,
    next_id: u64,
    hold: Option<Hold>,
    /// Start failures (and crashes) per runtime, Unix ms.
    fails: BTreeMap<String, Vec<u64>>,
    /// No new start before this time (backoff after a failure).
    backoff: BTreeMap<String, u64>,
    fenced: BTreeMap<String, u64>,
    last_good: Option<String>,
    /// How long the last good switch to each runtime took, ms.
    switch_ms: BTreeMap<String, u64>,
    switching: bool,
}

const BACKOFF_FIRST_MS: u64 = 5_000;
const BACKOFF_MAX_MS: u64 = 300_000;

impl Core {
    pub fn new(cfg: Arc<Config>, saved: Saved) -> Core {
        // Only what still names a runtime in this config comes back.
        let known = |rt: &String| cfg.runtime(rt).is_some();
        Core {
            phase: Phase::Idle,
            inflight: BTreeMap::new(),
            waiting: Vec::new(),
            next_id: 1,
            hold: saved.hold.filter(|h| known(&h.rt)),
            fails: BTreeMap::new(),
            backoff: BTreeMap::new(),
            fenced: saved
                .fenced
                .into_iter()
                .filter(|(rt, _)| known(rt))
                .collect(),
            last_good: saved.last_good.filter(known),
            switch_ms: BTreeMap::new(),
            switching: false,
            cfg,
        }
    }

    pub fn config(&self) -> &Arc<Config> {
        &self.cfg
    }

    pub fn phase(&self) -> &Phase {
        &self.phase
    }

    pub fn saved(&self) -> Saved {
        Saved {
            hold: self.hold.clone(),
            fenced: self.fenced.clone(),
            last_good: self.last_good.clone(),
        }
    }

    /// The runtime allowed on the GPUs right now, if any.
    pub fn lease_holder(&self) -> Option<&str> {
        match &self.phase {
            Phase::Starting { rt } | Phase::Ready { rt, .. } => Some(rt),
            Phase::Draining { from, .. } => Some(from),
            Phase::Idle | Phase::Stopping { .. } | Phase::Fault { .. } => None,
        }
    }

    pub fn is_ready(&self, rt: &str) -> bool {
        matches!(&self.phase, Phase::Ready { rt: r, .. } if r == rt)
    }

    fn fenced_until(&self, rt: &str, now: u64) -> Option<u64> {
        self.fenced.get(rt).copied().filter(|&t| t > now)
    }

    fn active_hold(&self, now: u64) -> Option<&Hold> {
        self.hold
            .as_ref()
            .filter(|h| h.until.is_none_or(|t| t > now))
    }

    fn eta_secs(&self, rt: &str) -> u64 {
        let ms = self.switch_ms.get(rt).copied().unwrap_or_else(|| {
            self.cfg
                .runtime(rt)
                .map_or(60_000, |r| r.start_timeout_secs * 1000 / 4)
        });
        (ms / 1000).clamp(5, 3600)
    }

    /// A request for `rt` arrives.
    pub fn admit(&mut self, now: u64, rt: &str, class: Class) -> Admit {
        if self.cfg.runtime(rt).is_none() {
            return Admit::Refuse {
                status: 404,
                reason: format!("no runtime {rt}"),
                retry_after: 0,
            };
        }
        if let Some(t) = self.fenced_until(rt, now) {
            return Admit::Refuse {
                status: 503,
                reason: format!(
                    "{rt} is fenced after failing to start; it is retried at {t} or by kiri-sched unfence"
                ),
                retry_after: (t - now) / 1000 + 1,
            };
        }
        if let Some(h) = self.active_hold(now)
            && h.rt != rt
        {
            let retry = h.until.map_or(300, |t| (t - now) / 1000 + 1);
            return Admit::Refuse {
                status: 503,
                reason: format!("the GPUs are held for {} by {}", h.rt, h.by),
                retry_after: retry,
            };
        }
        if let Phase::Fault { reason } = &self.phase {
            return Admit::Refuse {
                status: 503,
                reason: format!("GPU fault, needs a look: {reason}"),
                retry_after: 60,
            };
        }
        if self.is_ready(rt) {
            return Admit::Go;
        }
        if self.waiting.len() >= self.cfg.sched.max_waiting {
            return Admit::Refuse {
                status: 429,
                reason: "too many requests waiting for the GPUs".into(),
                retry_after: self.eta_secs(rt),
            };
        }
        let id = self.next_id;
        self.next_id += 1;
        self.waiting.push(Waiter {
            id,
            rt: rt.to_owned(),
            class,
            since: now,
        });
        Admit::Wait(id)
    }

    /// A waiting request checks in: `Some(Go)` once its runtime is ready,
    /// `Some(Refuse)` when it should stop waiting, `None` to keep waiting.
    pub fn poll(&mut self, now: u64, id: u64) -> Option<Admit> {
        let w = self.waiting.iter().find(|w| w.id == id)?.clone();
        let deadline = w.since + self.cfg.wait_secs(w.class) * 1000;
        let answer = if self.is_ready(&w.rt) {
            Some(Admit::Go)
        } else if let Some(t) = self.fenced_until(&w.rt, now) {
            Some(Admit::Refuse {
                status: 503,
                reason: format!("{} is fenced after failing to start", w.rt),
                retry_after: (t - now) / 1000 + 1,
            })
        } else if let Some(h) = self.active_hold(now).filter(|h| h.rt != w.rt) {
            Some(Admit::Refuse {
                status: 503,
                reason: format!("the GPUs are held for {} by {}", h.rt, h.by),
                retry_after: h.until.map_or(300, |t| (t - now) / 1000 + 1),
            })
        } else if let Phase::Fault { reason } = &self.phase {
            Some(Admit::Refuse {
                status: 503,
                reason: format!("GPU fault, needs a look: {reason}"),
                retry_after: 60,
            })
        } else if now >= deadline {
            let pos = self.position(id).unwrap_or(0);
            Some(Admit::Refuse {
                status: 503,
                reason: format!("still waiting for {} (position {pos})", w.rt),
                retry_after: self.eta_secs(&w.rt),
            })
        } else {
            None
        };
        if answer.is_some() {
            self.waiting.retain(|x| x.id != id);
        }
        answer
    }

    /// A waiting request went away (the client hung up).
    pub fn forget(&mut self, id: u64) {
        self.waiting.retain(|w| w.id != id);
    }

    /// Place in line among requests for the same runtime, from 1.
    pub fn position(&self, id: u64) -> Option<usize> {
        let w = self.waiting.iter().find(|w| w.id == id)?;
        Some(
            self.waiting
                .iter()
                .filter(|x| x.rt == w.rt && x.since <= w.since && x.id <= w.id)
                .count(),
        )
    }

    pub fn begin(&mut self, rt: &str) {
        *self.inflight.entry(rt.to_owned()).or_default() += 1;
    }

    pub fn end(&mut self, rt: &str) {
        if let Some(n) = self.inflight.get_mut(rt) {
            *n = n.saturating_sub(1);
        }
    }

    pub fn inflight(&self, rt: &str) -> usize {
        self.inflight.get(rt).copied().unwrap_or(0)
    }

    fn startable(&self, rt: &str, now: u64) -> bool {
        self.fenced_until(rt, now).is_none() && self.backoff.get(rt).is_none_or(|&t| t <= now)
    }

    /// The runtime waiting work wants most, other than `cur`: interactive
    /// (and agent work old enough to count as interactive) first, then
    /// agent work; oldest first. Batch requests never move the box.
    fn wanted(&self, now: u64, cur: Option<&str>) -> Option<(String, bool)> {
        let aging = self.cfg.sched.agent_aging_secs * 1000;
        let urgent = |w: &Waiter| {
            w.class == Class::Interactive
                || (w.class == Class::Agent && now.saturating_sub(w.since) >= aging)
        };
        let pick = |want_urgent: bool| {
            self.waiting
                .iter()
                .filter(|w| {
                    Some(w.rt.as_str()) != cur
                        && w.class != Class::Batch
                        && urgent(w) == want_urgent
                        && self.startable(&w.rt, now)
                })
                .min_by_key(|w| (w.since, w.id))
                .map(|w| (w.rt.clone(), want_urgent))
        };
        pick(true).or_else(|| pick(false))
    }

    /// Whether to start a switch now, and to which runtime. The caller runs
    /// the switch and reports back with the `switch_*` calls.
    pub fn want_switch(&mut self, now: u64) -> Option<String> {
        if self.switching {
            return None;
        }
        if self
            .hold
            .as_ref()
            .is_some_and(|h| h.until.is_some_and(|t| t <= now))
        {
            self.hold = None;
        }
        self.fenced.retain(|_, t| *t > now);
        let hold = self.active_hold(now).map(|h| h.rt.clone());
        match &self.phase {
            Phase::Fault { .. }
            | Phase::Draining { .. }
            | Phase::Stopping { .. }
            | Phase::Starting { .. } => None,
            Phase::Idle => {
                if let Some(h) = hold {
                    return self.startable(&h, now).then_some(h);
                }
                if let Some((rt, _)) = self.wanted(now, None) {
                    return Some(rt);
                }
                let home = self.cfg.sched.home.clone();
                [self.last_good.clone(), Some(home)]
                    .into_iter()
                    .flatten()
                    .find(|rt| self.startable(rt, now))
            }
            Phase::Ready { rt: cur, since } => {
                if let Some(h) = hold {
                    return (h != *cur && self.startable(&h, now)).then_some(h);
                }
                let (cand, urgent) = self.wanted(now, Some(cur))?;
                let busy = self.inflight(cur) > 0 || self.waiting.iter().any(|w| &w.rt == cur);
                let dwelt = now.saturating_sub(*since) >= self.cfg.sched.dwell_secs * 1000;
                (!busy || (urgent && dwelt)).then_some(cand)
            }
        }
    }

    /// The switch to `to` begins: the current runtime drains first.
    pub fn switch_begin(&mut self, to: &str) {
        self.switching = true;
        self.phase = match &self.phase {
            Phase::Ready { rt, .. } if rt != to => Phase::Draining {
                from: rt.clone(),
                to: to.to_owned(),
            },
            Phase::Ready { rt, .. } => Phase::Stopping {
                from: Some(rt.clone()),
                to: to.to_owned(),
            },
            _ => Phase::Stopping {
                from: None,
                to: to.to_owned(),
            },
        };
    }

    /// Draining is over (in-flight work done, or its time ran out).
    pub fn switch_stopping(&mut self) {
        if let Phase::Draining { from, to } = &self.phase {
            self.phase = Phase::Stopping {
                from: Some(from.clone()),
                to: to.clone(),
            };
        }
    }

    /// The old runtime is stopped and the GPUs were seen empty: only now may
    /// `to` get the GPUs.
    pub fn switch_starting(&mut self) {
        if let Phase::Stopping { to, .. } = &self.phase {
            self.phase = Phase::Starting { rt: to.clone() };
        }
    }

    pub fn switch_ready(&mut self, now: u64, rt: &str, took_ms: u64) {
        if matches!(&self.phase, Phase::Starting { rt: r } if r == rt) {
            self.phase = Phase::Ready {
                rt: rt.to_owned(),
                since: now,
            };
            self.last_good = Some(rt.to_owned());
            self.switch_ms.insert(rt.to_owned(), took_ms);
            self.fails.remove(rt);
            self.backoff.remove(rt);
        }
        self.switching = false;
    }

    /// `rt` failed to start (and was stopped again): count it, back off, and
    /// fence it if it keeps failing. The GPUs are empty again.
    pub fn switch_failed(&mut self, now: u64, rt: &str) {
        self.record_failure(now, rt);
        self.phase = Phase::Idle;
        self.switching = false;
    }

    /// A stop failed or the GPUs stayed in use: stop everything.
    pub fn switch_fault(&mut self, reason: String) {
        self.phase = Phase::Fault { reason };
        self.switching = false;
    }

    /// The ready runtime died or stopped answering; the driver has stopped it.
    pub fn crashed(&mut self, now: u64, rt: &str) {
        if self.is_ready(rt) {
            self.record_failure(now, rt);
            self.phase = Phase::Idle;
        }
    }

    /// At startup: the runtime found running and answering is the one the
    /// GPUs stay with (the driver stopped any other).
    pub fn adopt(&mut self, now: u64, rt: &str) {
        if self.cfg.runtime(rt).is_some() && matches!(self.phase, Phase::Idle) {
            self.phase = Phase::Ready {
                rt: rt.to_owned(),
                since: now,
            };
            self.last_good = Some(rt.to_owned());
        }
    }

    fn record_failure(&mut self, now: u64, rt: &str) {
        let window = self.cfg.sched.fence_window_secs * 1000;
        let list = self.fails.entry(rt.to_owned()).or_default();
        list.push(now);
        list.retain(|&t| now.saturating_sub(t) < window);
        let n = list.len() as u32;
        if n >= self.cfg.sched.fence_after {
            self.fenced
                .insert(rt.to_owned(), now + self.cfg.sched.fence_secs * 1000);
            self.fails.remove(rt);
            self.backoff.remove(rt);
        } else {
            let wait = (BACKOFF_FIRST_MS << (n.saturating_sub(1)).min(16)).min(BACKOFF_MAX_MS);
            self.backoff.insert(rt.to_owned(), now + wait);
        }
        if self.last_good.as_deref() == Some(rt) && self.fenced.contains_key(rt) {
            self.last_good = None;
        }
    }

    pub fn unfence(&mut self, rt: &str) -> bool {
        self.fails.remove(rt);
        self.backoff.remove(rt);
        self.fenced.remove(rt).is_some()
    }

    /// Clears a fault once someone checked the GPUs.
    pub fn reset(&mut self) -> bool {
        if matches!(self.phase, Phase::Fault { .. }) {
            self.phase = Phase::Idle;
            true
        } else {
            false
        }
    }

    pub fn set_hold(&mut self, rt: &str, until: Option<u64>, by: &str) -> Result<(), String> {
        if self.cfg.runtime(rt).is_none() {
            return Err(format!("no runtime {rt}"));
        }
        self.hold = Some(Hold {
            rt: rt.to_owned(),
            until,
            by: by.to_owned(),
        });
        Ok(())
    }

    pub fn release(&mut self) -> Option<Hold> {
        self.hold.take()
    }

    #[cfg(test)]
    pub fn hold(&self, now: u64) -> Option<&Hold> {
        self.active_hold(now)
    }

    /// The Kiri daemon asks before a container gets GPUs. The container
    /// names its runtime with the `dev.kiri.sched.runtime` label, or by its
    /// Compose project (the stack name).
    pub fn lease_ok(&self, labels: &Value) -> Result<(), String> {
        let label = |k: &str| {
            labels
                .get(k)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
        };
        let rt = match (
            label("dev.kiri.sched.runtime"),
            label("com.docker.compose.project"),
        ) {
            (Some(name), _) => self.cfg.runtime(name).map(|r| r.name.clone()),
            (None, Some(stack)) => self.cfg.runtime_for_stack(stack).map(|r| r.name.clone()),
            (None, None) => None,
        };
        let Some(rt) = rt else {
            return Err("this container is not a GPU runtime in the scheduler's config".into());
        };
        match self.lease_holder() {
            Some(h) if h == rt => Ok(()),
            Some(h) => Err(format!(
                "the GPUs belong to {h}; ask the scheduler for {rt} instead (kiri-sched use {rt})"
            )),
            None => Err(format!(
                "no runtime holds the GPUs right now ({}); {rt} starts only through the scheduler",
                self.phase.name()
            )),
        }
    }

    pub fn status(&self, now: u64) -> Value {
        let mut waiting: BTreeMap<&str, Value> = BTreeMap::new();
        for w in &self.waiting {
            let e = waiting.entry(&w.rt).or_insert_with(
                || json!({"interactive": 0, "agent": 0, "batch": 0, "oldest_secs": 0}),
            );
            e[w.class.as_str()] = (e[w.class.as_str()].as_u64().unwrap_or(0) + 1).into();
            let age = now.saturating_sub(w.since) / 1000;
            if age > e["oldest_secs"].as_u64().unwrap_or(0) {
                e["oldest_secs"] = age.into();
            }
        }
        let runtimes: Vec<Value> = self
            .cfg
            .runtimes
            .iter()
            .map(|r| {
                let state = if self.fenced_until(&r.name, now).is_some() {
                    "fenced"
                } else if self.is_ready(&r.name) {
                    "ready"
                } else if self.lease_holder() == Some(r.name.as_str()) {
                    self.phase.name()
                } else if self.backoff.get(&r.name).is_some_and(|&t| t > now) {
                    "backoff"
                } else {
                    "stopped"
                };
                json!({
                    "name": r.name,
                    "state": state,
                    "models": r.models,
                    "inflight": self.inflight(&r.name),
                    "waiting": waiting.get(r.name.as_str()).cloned().unwrap_or(json!(null)),
                    "fenced_until": self.fenced_until(&r.name, now),
                    "last_switch_secs": self.switch_ms.get(&r.name).map(|ms| ms / 1000),
                })
            })
            .collect();
        json!({
            "phase": self.phase,
            "hold": self.active_hold(now),
            "home": self.cfg.sched.home,
            "runtimes": runtimes,
            "waiting_total": self.waiting.len(),
        })
    }

    /// The model list clients see: every model, with its runtime's state.
    pub fn models(&self, now: u64) -> Value {
        let st = self.status(now);
        let mut data = Vec::new();
        for r in st["runtimes"].as_array().into_iter().flatten() {
            for m in r["models"].as_array().into_iter().flatten() {
                data.push(json!({"id": m, "object": "model", "owned_by": r["name"], "status": r["state"]}));
            }
        }
        json!({"object": "list", "data": data})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EXAMPLE;

    const S: u64 = 1000;

    fn core() -> Core {
        Core::new(Arc::new(Config::parse(EXAMPLE).unwrap()), Saved::default())
    }

    /// Runs a whole switch as the driver would, succeeding.
    fn switch_ok(c: &mut Core, now: u64) -> String {
        let to = c.want_switch(now).expect("a switch");
        c.switch_begin(&to);
        c.switch_stopping();
        c.switch_starting();
        assert_eq!(c.lease_holder(), Some(to.as_str()));
        c.switch_ready(now, &to, 10 * S);
        to
    }

    #[test]
    fn boots_to_home_and_serves_it() {
        let mut c = core();
        assert_eq!(c.admit(0, "maya", Class::Agent), Admit::Wait(1));
        assert_eq!(switch_ok(&mut c, 0), "maya");
        assert_eq!(c.poll(S, 1), Some(Admit::Go));
        assert_eq!(c.admit(S, "maya", Class::Agent), Admit::Go);
        // Nothing else wanted: no switch.
        assert_eq!(c.want_switch(2 * S), None);
    }

    #[test]
    fn idle_box_switches_at_once_busy_box_waits_for_dwell() {
        let mut c = core();
        switch_ok(&mut c, 0);
        c.begin("maya");
        let Admit::Wait(id) = c.admit(S, "comfyui", Class::Interactive) else {
            panic!()
        };
        // maya is busy and has not dwelt 180 s: wait.
        assert_eq!(c.want_switch(10 * S), None);
        assert_eq!(c.want_switch(181 * S), Some("comfyui".into()));
        c.switch_begin("comfyui");
        assert_eq!(
            c.phase(),
            &Phase::Draining {
                from: "maya".into(),
                to: "comfyui".into()
            }
        );
        // While draining, maya still holds the lease; new maya work waits.
        assert_eq!(c.lease_holder(), Some("maya"));
        assert!(matches!(
            c.admit(182 * S, "maya", Class::Agent),
            Admit::Wait(_)
        ));
        c.end("maya");
        c.switch_stopping();
        assert_eq!(c.lease_holder(), None);
        c.switch_starting();
        c.switch_ready(190 * S, "comfyui", 8 * S);
        assert_eq!(c.poll(190 * S, id), Some(Admit::Go));
    }

    #[test]
    fn agent_work_switches_only_an_idle_box_until_it_ages() {
        let mut c = core();
        switch_ok(&mut c, 0);
        c.begin("maya");
        c.admit(S, "comfyui", Class::Agent);
        assert_eq!(c.want_switch(500 * S), None);
        // Aged past 600 s it counts as interactive, and maya has dwelt.
        assert_eq!(c.want_switch(602 * S), Some("comfyui".into()));
    }

    #[test]
    fn batch_requests_never_switch() {
        let mut c = core();
        switch_ok(&mut c, 0);
        c.admit(S, "comfyui", Class::Batch);
        assert_eq!(c.want_switch(10_000 * S), None);
    }

    #[test]
    fn holds_win_and_refuse_others() {
        let mut c = core();
        switch_ok(&mut c, 0);
        c.begin("maya");
        c.set_hold("comfyui", Some(3600 * S), "cody").unwrap();
        // A hold switches even a busy box that has not dwelt.
        assert_eq!(c.want_switch(S), Some("comfyui".into()));
        c.switch_begin("comfyui");
        c.end("maya");
        c.switch_stopping();
        c.switch_starting();
        c.switch_ready(2 * S, "comfyui", S);
        match c.admit(3 * S, "maya", Class::Interactive) {
            Admit::Refuse {
                status: 503,
                retry_after,
                ..
            } => assert!(retry_after > 3000),
            other => panic!("{other:?}"),
        }
        // Past its end the hold is gone and maya may come back.
        assert!(matches!(
            c.admit(3601 * S, "maya", Class::Interactive),
            Admit::Wait(_)
        ));
        assert_eq!(c.want_switch(3601 * S), Some("maya".into()));
    }

    #[test]
    fn failures_back_off_then_fence() {
        let mut c = core();
        c.admit(0, "comfyui", Class::Interactive);
        let mut now = 0;
        for i in 0..3 {
            assert_eq!(c.want_switch(now), Some("comfyui".into()), "try {i}");
            c.switch_begin("comfyui");
            c.switch_starting();
            c.switch_failed(now, "comfyui");
            // Backed off: the box goes home meanwhile, not round in a loop.
            if i < 2 {
                assert_eq!(c.want_switch(now + 1), Some("maya".into()));
                now += 100 * S;
            }
        }
        assert!(matches!(
            c.admit(now, "comfyui", Class::Interactive),
            Admit::Refuse { status: 503, .. }
        ));
        assert!(c.saved().fenced.contains_key("comfyui"));
        assert!(c.unfence("comfyui"));
        assert!(matches!(
            c.admit(now, "comfyui", Class::Interactive),
            Admit::Wait(_)
        ));
    }

    #[test]
    fn fault_stops_everything_until_reset() {
        let mut c = core();
        c.switch_begin("maya");
        c.switch_fault("GPU 1 still uses 9000 MiB".into());
        assert_eq!(c.want_switch(S), None);
        assert!(matches!(
            c.admit(S, "maya", Class::Interactive),
            Admit::Refuse { status: 503, .. }
        ));
        assert!(c.reset());
        assert_eq!(c.want_switch(S), Some("maya".into()));
    }

    #[test]
    fn waiting_is_bounded_and_times_out() {
        let mut c = core();
        let mut ids = Vec::new();
        for _ in 0..256 {
            let Admit::Wait(id) = c.admit(0, "comfyui", Class::Batch) else {
                panic!()
            };
            ids.push(id);
        }
        assert!(matches!(
            c.admit(0, "comfyui", Class::Batch),
            Admit::Refuse { status: 429, .. }
        ));
        assert_eq!(c.poll(29 * S, ids[0]), None);
        assert!(matches!(
            c.poll(30 * S, ids[0]),
            Some(Admit::Refuse { status: 503, .. })
        ));
        assert_eq!(c.position(ids[255]), Some(255));
    }

    #[test]
    fn lease_answers() {
        let mut c = core();
        let maya = json!({"com.docker.compose.project": "maya"});
        let comfy = json!({"dev.kiri.sched.runtime": "comfyui"});
        assert!(c.lease_ok(&maya).is_err());
        c.switch_begin("maya");
        assert!(
            c.lease_ok(&maya).is_err(),
            "not before the GPUs are seen empty"
        );
        c.switch_starting();
        assert!(c.lease_ok(&maya).is_ok());
        assert!(c.lease_ok(&comfy).unwrap_err().contains("belong to maya"));
        assert!(c.lease_ok(&json!({})).is_err());
        assert!(
            c.lease_ok(&json!({"com.docker.compose.project": "dashboard"}))
                .is_err()
        );
    }

    #[test]
    fn saved_state_comes_back() {
        let mut c = core();
        c.set_hold("comfyui", None, "cody").unwrap();
        let saved: Saved =
            serde_json::from_value(serde_json::to_value(c.saved()).unwrap()).unwrap();
        let mut c = Core::new(Arc::new(Config::parse(EXAMPLE).unwrap()), saved);
        assert_eq!(c.want_switch(0), Some("comfyui".into()));
        // A saved hold on a runtime no longer in the config is dropped.
        let old = Saved {
            hold: Some(Hold {
                rt: "gone".into(),
                until: None,
                by: "x".into(),
            }),
            ..Saved::default()
        };
        let c = Core::new(Arc::new(Config::parse(EXAMPLE).unwrap()), old);
        assert!(c.hold(0).is_none());
    }

    /// A long random run of requests, completions, holds, failures and
    /// crashes, checking the invariants after every step.
    #[test]
    fn invariants_hold_over_random_runs() {
        for seed in 1..=200u64 {
            let mut rng = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut next = |n: u64| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng % n
            };
            let mut c = core();
            let mut now = 0u64;
            let mut waits: Vec<u64> = Vec::new();
            let rts = ["maya", "comfyui"];
            let classes = [Class::Interactive, Class::Agent, Class::Batch];
            for _ in 0..2000 {
                now += next(20) * S;
                match next(10) {
                    0..=2 => {
                        if let Admit::Wait(id) =
                            c.admit(now, rts[next(2) as usize], classes[next(3) as usize])
                        {
                            waits.push(id);
                        }
                    }
                    3 => {
                        if let Some(rt) = c.lease_holder().map(str::to_owned)
                            && c.is_ready(&rt)
                        {
                            c.begin(&rt);
                        }
                    }
                    4 => {
                        for rt in rts {
                            if c.inflight(rt) > 0 {
                                c.end(rt);
                            }
                        }
                    }
                    5 => {
                        if next(4) == 0 {
                            let _ =
                                c.set_hold(rts[next(2) as usize], Some(now + next(600) * S), "t");
                        } else {
                            c.release();
                        }
                    }
                    6 => {
                        if let Some(rt) = c.lease_holder().map(str::to_owned) {
                            c.crashed(now, &rt);
                        }
                    }
                    _ => {
                        let held = c.hold(now).map(|h| h.rt.clone());
                        let before = c.lease_holder().map(str::to_owned);
                        if let Some(to) = c.want_switch(now) {
                            // Invariant 2: never away from a live hold.
                            if let Some(h) = &held {
                                assert_eq!(&to, h, "seed {seed}: switched away from a hold");
                            }
                            assert_ne!(
                                Some(&to),
                                before.as_ref().filter(|_| c.is_ready(&to)),
                                "seed {seed}: switch to itself"
                            );
                            c.switch_begin(&to);
                            c.switch_stopping();
                            assert_eq!(
                                c.lease_holder(),
                                None,
                                "seed {seed}: lease held while stopping"
                            );
                            c.switch_starting();
                            if next(5) == 0 {
                                c.switch_failed(now, &to);
                            } else {
                                c.switch_ready(now, &to, S);
                            }
                        }
                    }
                }
                waits.retain(|&id| c.poll(now, id).is_none());
                // Invariant 3: bounded.
                assert!(c.waiting.len() <= c.cfg.sched.max_waiting);
                // Invariant 1: one holder, and it is a configured runtime.
                if let Some(h) = c.lease_holder() {
                    assert!(c.cfg.runtime(h).is_some());
                }
                // Invariant 4: a fenced runtime is never the one starting.
                if let Phase::Starting { rt } = c.phase() {
                    assert!(c.fenced_until(rt, now).is_none());
                }
            }
        }
    }
}
