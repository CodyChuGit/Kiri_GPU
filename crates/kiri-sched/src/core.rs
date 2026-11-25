//! The scheduler's decisions, with no I/O and no clock of its own: every
//! call takes `now` (Unix milliseconds), so tests drive it with a fake
//! clock and check its invariants over long random runs.
//!
//! The box has `gpus` cards. Each runtime needs `cards` of them (all, by
//! default); runtimes that fit side by side run side by side, each on its
//! own cards.
//!
//! The invariants, which nothing here may break:
//! 1. No card belongs to two runtimes, and a runtime only gets cards (the
//!    lease) once the switch driver stopped what was on them and saw them
//!    empty. One switch runs at a time.
//! 2. Nothing evicts a hold while it lasts.
//! 3. Waiting work is bounded (`max_waiting`); more is refused, not queued.
//! 4. A runtime that keeps failing is fenced, never restarted in a loop.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::{Class, Config};

/// Where one runtime on the GPUs stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SlotState {
    /// Being started: the only runtime that may get these cards.
    Starting,
    Ready,
    /// Finishes what it has in flight before it is stopped; new work for it
    /// waits. It still holds its cards.
    Draining,
    /// Being stopped; its cards are checked empty before anything else
    /// gets them.
    Stopping,
    /// Its stop failed: it may still be on its cards, so they are not free
    /// and it has no lease. Looked at again after `kiri-sched reset`.
    Stuck,
}

impl SlotState {
    pub fn name(self) -> &'static str {
        match self {
            SlotState::Starting => "starting",
            SlotState::Ready => "ready",
            SlotState::Draining => "draining",
            SlotState::Stopping => "stopping",
            SlotState::Stuck => "stuck",
        }
    }
}

/// One runtime on the GPUs, and its cards.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Slot {
    pub cards: Vec<usize>,
    pub state: SlotState,
    /// When it became ready (Unix ms).
    pub since: u64,
}

/// A switch: stop `stop` (draining them first), then start `start`, if
/// any, on `cards`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Plan {
    pub start: Option<String>,
    pub cards: Vec<usize>,
    pub stop: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hold {
    pub rt: String,
    /// Unix ms; `None` holds until released (or, with `after_ready_secs`,
    /// until its runtime is ready and that long has passed).
    pub until: Option<u64>,
    pub by: String,
    /// A `use`: the hold's time starts when its runtime is ready, so a slow
    /// load or a switch that waits does not eat it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_ready_secs: Option<u64>,
    /// A `use` whose runtime is not ready by then is given up.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_by: Option<u64>,
}

/// How long a hold lasts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoldFor {
    /// Until this Unix ms, or until released.
    Until(Option<u64>),
    /// This many seconds once its runtime is ready (`use`).
    AfterReady(u64),
}

/// A refused admin action: HTTP status and reason.
pub type Refusal = (u16, String);

/// What survives a restart of the scheduler (and of the box).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Saved {
    /// The first hold, as older versions read it.
    #[serde(default)]
    pub hold: Option<Hold>,
    #[serde(default)]
    pub holds: Vec<Hold>,
    #[serde(default)]
    pub fenced: BTreeMap<String, u64>,
    #[serde(default)]
    pub last_good: Option<String>,
    /// The cards each runtime on the GPUs had, so one still running after a
    /// restart of the scheduler can be kept on them.
    #[serde(default)]
    pub cards: BTreeMap<String, Vec<usize>>,
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
    /// The runtimes on the GPUs, by name.
    running: BTreeMap<String, Slot>,
    /// The switch under way, if any.
    trans: Option<Plan>,
    /// Cards could not be shown empty (a stop failed, memory stayed in
    /// use). No switch runs until someone looks: `kiri-sched reset`.
    fault: Option<String>,
    inflight: BTreeMap<String, usize>,
    waiting: Vec<Waiter>,
    next_id: u64,
    /// Holds, each on one runtime; together they always fit on the box.
    holds: Vec<Hold>,
    /// Start failures (and crashes) per runtime, Unix ms.
    fails: BTreeMap<String, Vec<u64>>,
    /// No new start before this time (backoff after a failure).
    backoff: BTreeMap<String, u64>,
    fenced: BTreeMap<String, u64>,
    last_good: Option<String>,
    /// Cards from the saved state, for `adopt` at startup.
    boot_cards: BTreeMap<String, Vec<usize>>,
    /// How long the last good switch to each runtime took, ms.
    switch_ms: BTreeMap<String, u64>,
    /// Since when each ready runtime with an idle limit has had nothing to
    /// do (no hold, nothing in flight or waiting); see `home_after_idle_secs`.
    idle_since: BTreeMap<String, u64>,
    /// Runtimes that crashed and come back on free cards (after backoff)
    /// unless something else took them.
    revive: BTreeSet<String>,
    /// Why `want_switch` last asked for a switch, for the audit log.
    why: &'static str,
}

const BACKOFF_FIRST_MS: u64 = 5_000;
const BACKOFF_MAX_MS: u64 = 300_000;

impl Core {
    pub fn new(cfg: Arc<Config>, saved: Saved) -> Core {
        // Only what still names a runtime in this config comes back.
        let known = |rt: &String| cfg.runtime(rt).is_some();
        Core {
            running: BTreeMap::new(),
            trans: None,
            fault: None,
            inflight: BTreeMap::new(),
            waiting: Vec::new(),
            next_id: 1,
            holds: if saved.holds.is_empty() {
                saved.hold.into_iter().collect()
            } else {
                saved.holds
            }
            .into_iter()
            .filter(|h| known(&h.rt))
            .collect(),
            fails: BTreeMap::new(),
            backoff: BTreeMap::new(),
            fenced: saved
                .fenced
                .into_iter()
                .filter(|(rt, _)| known(rt))
                .collect(),
            last_good: saved.last_good.filter(known),
            boot_cards: saved
                .cards
                .into_iter()
                .filter(|(rt, _)| known(rt))
                .collect(),
            switch_ms: BTreeMap::new(),
            idle_since: BTreeMap::new(),
            revive: BTreeSet::new(),
            why: "",
            cfg,
        }
    }

    pub fn config(&self) -> &Arc<Config> {
        &self.cfg
    }

    /// Why the last switch `want_switch` returned was wanted.
    pub fn why(&self) -> &'static str {
        self.why
    }

    pub fn fault(&self) -> Option<&str> {
        self.fault.as_deref()
    }

    pub fn slot(&self, rt: &str) -> Option<&Slot> {
        self.running.get(rt)
    }

    /// The runtimes on the GPUs (in any state), by name.
    #[cfg(test)]
    pub fn running(&self) -> &BTreeMap<String, Slot> {
        &self.running
    }

    /// The runtimes that are ready, by name.
    pub fn ready(&self) -> Vec<String> {
        self.running
            .iter()
            .filter(|(_, s)| s.state == SlotState::Ready)
            .map(|(rt, _)| rt.clone())
            .collect()
    }

    pub fn saved(&self) -> Saved {
        Saved {
            hold: self.holds.first().cloned(),
            holds: self.holds.clone(),
            fenced: self.fenced.clone(),
            last_good: self.last_good.clone(),
            cards: self
                .running
                .iter()
                .map(|(rt, s)| (rt.clone(), s.cards.clone()))
                .collect(),
        }
    }

    /// The cards a runtime found running at startup may keep: all of them
    /// for a runtime that needs all, else the ones saved for it.
    pub fn boot_cards(&self, rt: &str) -> Option<Vec<usize>> {
        let gpus = self.cfg.sched.gpus;
        if self.cfg.cards(rt) == gpus {
            Some((0..gpus).collect())
        } else {
            self.boot_cards.get(rt).cloned()
        }
    }

    /// Whether `rt` may have its cards right now (the lease).
    #[cfg(test)]
    pub fn holds_lease(&self, rt: &str) -> bool {
        self.running
            .get(rt)
            .is_some_and(|s| !matches!(s.state, SlotState::Stopping | SlotState::Stuck))
    }

    pub fn is_ready(&self, rt: &str) -> bool {
        self.running
            .get(rt)
            .is_some_and(|s| s.state == SlotState::Ready)
    }

    fn fenced_until(&self, rt: &str, now: u64) -> Option<u64> {
        self.fenced.get(rt).copied().filter(|&t| t > now)
    }

    fn hold_live(h: &Hold, now: u64) -> bool {
        match (h.after_ready_secs, h.until) {
            (Some(_), None) => h.start_by.is_none_or(|t| t > now),
            (_, until) => until.is_none_or(|t| t > now),
        }
    }

    fn active_holds(&self, now: u64) -> impl Iterator<Item = &Hold> {
        self.holds.iter().filter(move |h| Core::hold_live(h, now))
    }

    /// The runtimes held right now, each once.
    fn held(&self, now: u64) -> BTreeSet<String> {
        self.active_holds(now).map(|h| h.rt.clone()).collect()
    }

    /// Cards the held runtimes need, beside `rt` (counted once if held).
    fn held_cards_with(&self, rt: &str, now: u64) -> usize {
        let mut set = self.held(now);
        set.insert(rt.to_owned());
        set.iter().map(|r| self.cfg.cards(r)).sum()
    }

    /// A `use` hold starts its clock once its runtime is ready.
    fn arm_hold(&mut self, now: u64, rt: &str) {
        for h in self.holds.iter_mut().filter(|h| h.rt == rt && h.until.is_none()) {
            if let Some(secs) = h.after_ready_secs.take() {
                h.until = Some(now + secs * 1000);
                h.start_by = None;
            }
        }
    }

    /// Holds refuse only the runtimes that cannot run beside them: the
    /// first hold in the way, if any.
    fn blocking_hold(&self, rt: &str, now: u64) -> Option<&Hold> {
        if self.held_cards_with(rt, now) <= self.cfg.sched.gpus {
            return None;
        }
        self.active_holds(now).find(|h| h.rt != rt)
    }

    fn eta_secs(&self, rt: &str) -> u64 {
        let ms = self.switch_ms.get(rt).copied().unwrap_or_else(|| {
            self.cfg
                .runtime(rt)
                .map_or(60_000, |r| r.start_timeout_secs * 1000 / 4)
        });
        (ms / 1000).clamp(5, 3600)
    }

    fn held_refusal(&self, h: &Hold, now: u64) -> Admit {
        Admit::Refuse {
            status: 503,
            reason: format!("the GPUs are held for {} by {}", h.rt, h.by),
            retry_after: h.until.map_or(300, |t| (t - now) / 1000 + 1),
        }
    }

    fn fault_refusal(reason: &str) -> Admit {
        Admit::Refuse {
            status: 503,
            reason: format!("GPU fault, needs a look: {reason}"),
            retry_after: 60,
        }
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
        if self.is_ready(rt) {
            return Admit::Go;
        }
        if let Some(h) = self.blocking_hold(rt, now) {
            return self.held_refusal(h, now);
        }
        if let Some(reason) = &self.fault {
            return Core::fault_refusal(reason);
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
        } else if let Some(h) = self.blocking_hold(&w.rt, now) {
            Some(self.held_refusal(h, now))
        } else if let Some(reason) = &self.fault {
            Some(Core::fault_refusal(reason))
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

    fn busy(&self, rt: &str) -> bool {
        self.inflight(rt) > 0 || self.waiting.iter().any(|w| w.rt == rt)
    }

    fn startable(&self, rt: &str, now: u64) -> bool {
        self.fenced_until(rt, now).is_none() && self.backoff.get(rt).is_none_or(|&t| t <= now)
    }

    fn free_cards(&self) -> BTreeSet<usize> {
        let used: BTreeSet<usize> = self
            .running
            .values()
            .flat_map(|s| s.cards.iter().copied())
            .collect();
        (0..self.cfg.sched.gpus)
            .filter(|c| !used.contains(c))
            .collect()
    }

    /// Cards for `target`: free ones first, then those of `forced` (stopped
    /// anyway), then of ready runtimes `may_evict` allows, fewest cards and
    /// idle ones first. `None` if it does not fit.
    fn plan_for(
        &self,
        target: &str,
        forced: &[String],
        may_evict: impl Fn(&str, &Slot) -> bool,
    ) -> Option<Plan> {
        if self.running.contains_key(target) {
            return None;
        }
        let need = self.cfg.cards(target);
        let mut stop: Vec<String> = forced.to_vec();
        let mut free = self.free_cards();
        for rt in forced {
            if let Some(s) = self.running.get(rt) {
                free.extend(s.cards.iter().copied());
            }
        }
        if free.len() < need {
            let mut cands: Vec<(&String, &Slot)> = self
                .running
                .iter()
                .filter(|(rt, s)| {
                    !stop.contains(rt) && s.state == SlotState::Ready && may_evict(rt, s)
                })
                .collect();
            cands.sort_by_key(|(rt, s)| (self.busy(rt), s.cards.len(), s.since));
            for (rt, s) in cands {
                if free.len() >= need {
                    break;
                }
                stop.push(rt.clone());
                free.extend(s.cards.iter().copied());
            }
            if free.len() < need {
                return None;
            }
        }
        Some(Plan {
            start: Some(target.to_owned()),
            cards: free.into_iter().take(need).collect(),
            stop,
        })
    }

    /// Whether to start a switch now, and which. The caller runs the switch
    /// and reports back with the `switch_*` calls.
    pub fn want_switch(&mut self, now: u64) -> Option<Plan> {
        if self.trans.is_some()
            || self.fault.is_some()
            || self.running.values().any(|s| s.state == SlotState::Stuck)
        {
            return None;
        }
        self.fenced.retain(|_, t| *t > now);
        let fenced = self.fenced.clone();
        self.holds.retain(|h| {
            Core::hold_live(h, now) && !(h.after_ready_secs.is_some() && fenced.contains_key(&h.rt))
        });
        // The idle clock of each ready runtime with an idle limit starts
        // when it is first seen with nothing to do.
        let held = self.held(now);
        let home = self.cfg.sched.home.clone();
        let idle: Vec<String> = self
            .running
            .iter()
            .filter(|(rt, s)| {
                s.state == SlotState::Ready
                    && **rt != home
                    && !held.contains(*rt)
                    && self
                        .cfg
                        .runtime(rt)
                        .is_some_and(|r| r.home_after_idle_secs.is_some())
                    && !self.busy(rt)
            })
            .map(|(rt, _)| rt.clone())
            .collect();
        self.idle_since.retain(|rt, _| idle.contains(rt));
        for rt in idle {
            self.idle_since.entry(rt).or_insert(now);
        }
        let (why, plan) = self.decide(now, &held)?;
        self.why = why;
        Some(plan)
    }

    fn decide(&self, now: u64, held: &BTreeSet<String>) -> Option<(&'static str, Plan)> {
        let not_held = |rt: &str| !held.contains(rt);
        // 1. Holds: each held runtime gets on, evicting whatever is in the
        // way but never another held one (the holds fit together). One
        // that cannot start yet (backoff) waits without blocking the rest.
        for h in held {
            if !self.running.contains_key(h)
                && self.startable(h, now)
                && let Some(p) = self.plan_for(h, &[], |rt, _| not_held(rt))
            {
                return Some(("hold", p));
            }
        }
        // 2. Waiting work: interactive (and agent work old enough to count as
        // interactive) first, then agent work, then batch; oldest first.
        let aging = self.cfg.sched.agent_aging_secs * 1000;
        let dwell = self.cfg.sched.dwell_secs * 1000;
        let rank = |w: &Waiter| match w.class {
            Class::Interactive => 0,
            Class::Agent if now.saturating_sub(w.since) >= aging => 0,
            Class::Agent => 1,
            Class::Batch => 2,
        };
        let mut waiters: Vec<&Waiter> = self
            .waiting
            .iter()
            .filter(|w| {
                !self.running.contains_key(&w.rt)
                    && self.startable(&w.rt, now)
                    && self.blocking_hold(&w.rt, now).is_none()
            })
            .collect();
        waiters.sort_by_key(|w| (rank(w), w.since, w.id));
        for w in waiters {
            let r = rank(w);
            // Batch work only takes free cards; agent work only idle ones;
            // urgent work also busy ones that have had their dwell time.
            let plan = self.plan_for(&w.rt, &[], |rt, s| {
                not_held(rt)
                    && r < 2
                    && (!self.busy(rt) || (r == 0 && now.saturating_sub(s.since) >= dwell))
            });
            if let Some(p) = plan {
                let dwelt = p.stop.iter().any(|rt| self.busy(rt));
                return Some((
                    if dwelt {
                        "request (current runtime dwelt)"
                    } else {
                        "request"
                    },
                    p,
                ));
            }
        }
        // 3. A runtime that crashed comes back if its cards are still free.
        for rt in &self.revive {
            if self.startable(rt, now) && self.blocking_hold(rt, now).is_none()
                && let Some(p) = self.plan_for(rt, &[], |_, _| false)
            {
                return Some(("restart after a crash", p));
            }
        }
        // 4. A runtime idle past its limit gives its cards back: to home if
        // home then fits, else just stops.
        let expired: Vec<String> = self
            .idle_since
            .iter()
            .filter(|(rt, t)| {
                self.cfg
                    .runtime(rt)
                    .and_then(|r| r.home_after_idle_secs)
                    .is_some_and(|secs| now.saturating_sub(**t) >= secs * 1000)
            })
            .map(|(rt, _)| rt.clone())
            .collect();
        if let Some(first) = expired.first() {
            let home = &self.cfg.sched.home;
            let to_home = (self.startable(home, now) && self.blocking_hold(home, now).is_none())
                .then(|| {
                    self.plan_for(home, std::slice::from_ref(first), |rt, _| {
                        expired.iter().any(|e| e == rt)
                    })
                })
                .flatten();
            return Some((
                "idle: home_after_idle_secs",
                to_home.unwrap_or(Plan {
                    start: None,
                    cards: Vec::new(),
                    stop: vec![first.clone()],
                }),
            ));
        }
        // 5. Nothing on the GPUs: the last good runtime, or home.
        if self.running.is_empty() {
            let home = self.cfg.sched.home.clone();
            return [self.last_good.clone(), Some(home)]
                .into_iter()
                .flatten()
                .filter(|rt| self.startable(rt, now) && self.blocking_hold(rt, now).is_none())
                .find_map(|rt| self.plan_for(&rt, &[], |_, _| false))
                .map(|p| ("nothing on the GPUs: last good or home", p));
        }
        None
    }

    /// A switch begins: the runtimes it stops drain first.
    pub fn switch_begin(&mut self, plan: &Plan) {
        for rt in &plan.stop {
            if let Some(s) = self.running.get_mut(rt) {
                s.state = SlotState::Draining;
            }
            self.idle_since.remove(rt);
        }
        self.trans = Some(plan.clone());
    }

    /// The switch under way, if any.
    #[cfg(test)]
    pub fn switching(&self) -> Option<&Plan> {
        self.trans.as_ref()
    }

    /// Draining is over (in-flight work done, or its time ran out): the
    /// runtimes being stopped lose their lease.
    pub fn switch_stopping(&mut self) {
        let Some(p) = &self.trans else { return };
        for rt in &p.stop {
            if let Some(s) = self.running.get_mut(rt) {
                s.state = SlotState::Stopping;
            }
        }
    }

    /// The stopped runtimes are gone and the cards were seen empty: only
    /// now may the new runtime get them. A switch that only stops ends here.
    pub fn switch_starting(&mut self) {
        let Some(p) = self.trans.clone() else { return };
        for rt in &p.stop {
            self.running.remove(rt);
        }
        match &p.start {
            Some(rt) => {
                self.revive.remove(rt);
                self.running.insert(
                    rt.clone(),
                    Slot {
                        cards: p.cards.clone(),
                        state: SlotState::Starting,
                        since: 0,
                    },
                );
            }
            None => {
                // Stopped for idleness: not where the box should come back to.
                if self.last_good.as_ref().is_some_and(|g| p.stop.contains(g)) {
                    self.last_good = None;
                }
                self.trans = None;
            }
        }
    }

    pub fn switch_ready(&mut self, now: u64, rt: &str, took_ms: u64) {
        if let Some(s) = self.running.get_mut(rt)
            && s.state == SlotState::Starting
        {
            s.state = SlotState::Ready;
            s.since = now;
            self.last_good = Some(rt.to_owned());
            self.switch_ms.insert(rt.to_owned(), took_ms);
            self.fails.remove(rt);
            self.backoff.remove(rt);
            self.arm_hold(now, rt);
        }
        self.trans = None;
    }

    /// `rt` failed to start (and was stopped again): count it, back off, and
    /// fence it if it keeps failing. Its cards are free again.
    pub fn switch_failed(&mut self, now: u64, rt: &str) {
        self.record_failure(now, rt);
        if self
            .running
            .get(rt)
            .is_some_and(|s| s.state == SlotState::Starting)
        {
            self.running.remove(rt);
        }
        self.trans = None;
    }

    /// A stop failed or cards stayed in use: no switch runs until a reset.
    /// The runtimes the switch touched are no longer counted on the GPUs.
    /// The runtimes the switch touched stay on their cards as `Stuck`: they
    /// may still be running, so their cards are not free.
    pub fn switch_fault(&mut self, reason: String) {
        if let Some(p) = self.trans.take() {
            for rt in p.stop.iter().chain(&p.start) {
                if let Some(s) = self.running.get_mut(rt) {
                    s.state = SlotState::Stuck;
                }
            }
        }
        self.fault = Some(reason);
    }

    /// A runtime that would not stop outside a switch (a crash check, or
    /// startup): it keeps its cards, or all of them if they are unknown.
    pub fn stuck(&mut self, rt: &str) {
        match self.running.get_mut(rt) {
            Some(s) => s.state = SlotState::Stuck,
            None => {
                let cards = self
                    .boot_cards(rt)
                    .unwrap_or_else(|| (0..self.cfg.sched.gpus).collect());
                self.running.insert(
                    rt.to_owned(),
                    Slot {
                        cards,
                        state: SlotState::Stuck,
                        since: 0,
                    },
                );
            }
        }
    }

    /// Stuck runtimes, for the driver to look at once the fault is reset.
    pub fn stuck_list(&self) -> Vec<String> {
        self.running
            .iter()
            .filter(|(_, s)| s.state == SlotState::Stuck)
            .map(|(rt, _)| rt.clone())
            .collect()
    }

    /// A stuck runtime turned out stopped: its cards are free.
    pub fn unstuck_stopped(&mut self, rt: &str) {
        if self.running.get(rt).is_some_and(|s| s.state == SlotState::Stuck) {
            self.running.remove(rt);
        }
    }

    /// A stuck runtime turned out running and answering: it is ready again
    /// on its cards (nothing else could have taken them).
    pub fn unstuck_ready(&mut self, now: u64, rt: &str) {
        if let Some(s) = self.running.get_mut(rt)
            && s.state == SlotState::Stuck
        {
            s.state = SlotState::Ready;
            s.since = now;
        }
    }

    /// A ready runtime died or stopped answering; the driver has stopped it.
    pub fn crashed(&mut self, now: u64, rt: &str) {
        if self.is_ready(rt) {
            self.record_failure(now, rt);
            self.running.remove(rt);
            self.idle_since.remove(rt);
            if !self.fenced.contains_key(rt) {
                self.revive.insert(rt.to_owned());
            }
        }
    }

    /// At startup: a runtime found running and answering stays, on the cards
    /// it had, if those are free. `false`: the driver must stop it.
    pub fn adopt(&mut self, now: u64, rt: &str, cards: &[usize]) -> bool {
        let free = self.free_cards();
        let ok = self.cfg.runtime(rt).is_some()
            && !self.running.contains_key(rt)
            && self.trans.is_none()
            && cards.len() == self.cfg.cards(rt)
            && cards.iter().all(|c| free.contains(c))
            && cards.iter().collect::<BTreeSet<_>>().len() == cards.len();
        if ok {
            self.running.insert(
                rt.to_owned(),
                Slot {
                    cards: cards.to_vec(),
                    state: SlotState::Ready,
                    since: now,
                },
            );
            self.last_good = Some(rt.to_owned());
            self.arm_hold(now, rt);
        }
        ok
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
            self.revive.remove(rt);
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
        self.fault.take().is_some()
    }

    /// Holds the GPUs for `rt`. Refused (409) while another runtime is
    /// held by someone else, unless `force`: a hold is never replaced by
    /// accident. The same caller, or anyone holding the same runtime,
    /// changes it freely.
    pub fn set_hold(
        &mut self,
        now: u64,
        rt: &str,
        hold_for: HoldFor,
        by: &str,
        force: bool,
    ) -> Result<(), Refusal> {
        let Some(r) = self.cfg.runtime(rt) else {
            return Err((404, format!("no runtime {rt}")));
        };
        if let Some(t) = self.fenced_until(rt, now) {
            return Err((
                503,
                format!("{rt} is fenced until {t} after failing to start; kiri-sched unfence {rt} first"),
            ));
        }
        if let Some(reason) = &self.fault {
            return Err((503, format!("GPU fault, nothing switches until kiri-sched reset: {reason}")));
        }
        // A named caller has one hold: a new one replaces its own. A bare
        // admin-socket caller is anyone on the box: it only refreshes its
        // hold on the same runtime.
        let named = by != "admin-socket";
        let replaced = |h: &Hold| h.by == by && (named || h.rt == rt);
        self.holds.retain(|h| Core::hold_live(h, now));
        let mut others: Vec<Hold> = self.holds.iter().filter(|h| !replaced(h)).cloned().collect();
        let fits = |others: &[Hold]| {
            let mut set: BTreeSet<&str> = others.iter().map(|h| h.rt.as_str()).collect();
            set.insert(rt);
            set.iter().map(|r| self.cfg.cards(r)).sum::<usize>() <= self.cfg.sched.gpus
        };
        // A bare caller's earlier bare holds give way to its new one when
        // they are in the way (`use deepseek` then `use qwen` just switches),
        // and stay when they fit beside it.
        while !fits(&others) {
            match others.iter().rposition(|h| h.by == by && h.rt != rt) {
                Some(i) => {
                    others.remove(i);
                }
                None => break,
            }
        }
        if !fits(&others) {
            if !force {
                let list: Vec<String> = others
                    .iter()
                    .filter(|h| h.rt != rt)
                    .map(|h| {
                        format!(
                            "{} by {}{}",
                            h.rt,
                            h.by,
                            h.until.map_or_else(String::new, |t| format!(" until {t}"))
                        )
                    })
                    .collect();
                return Err((409, format!(
                    "the GPUs are held for {}; {rt} does not fit beside that; wait, ask them to release it, or pass --force",
                    list.join(", ")
                )));
            }
            // --force: drop the holds in the way, newest first, until it fits.
            while !fits(&others) {
                match others.iter().rposition(|h| h.rt != rt) {
                    Some(i) => {
                        others.remove(i);
                    }
                    None => break,
                }
            }
        }
        let s = &self.cfg.sched;
        let hold = match hold_for {
            HoldFor::Until(until) => Hold {
                rt: rt.to_owned(),
                until,
                by: by.to_owned(),
                after_ready_secs: None,
                start_by: None,
            },
            HoldFor::AfterReady(secs) => Hold {
                rt: rt.to_owned(),
                until: None,
                by: by.to_owned(),
                after_ready_secs: Some(secs),
                start_by: Some(
                    now + (s.drain_secs + r.stop_timeout_secs + s.vram_wait_secs
                        + r.start_timeout_secs
                        + 60)
                        * 1000,
                ),
            },
        };
        others.push(hold);
        self.holds = others;
        if self.is_ready(rt) {
            self.arm_hold(now, rt);
        }
        Ok(())
    }

    /// Ends holds: every hold on `rt` when it is named, else the named
    /// caller's own. A bare admin-socket caller must name the runtime (or
    /// pass `force`, which ends them all).
    pub fn release(
        &mut self,
        now: u64,
        rt: Option<&str>,
        by: &str,
        force: bool,
    ) -> Result<Vec<Hold>, Refusal> {
        self.holds.retain(|h| Core::hold_live(h, now));
        let take = |h: &Hold| match rt {
            Some(r) => h.rt == r,
            None => force || h.by == by,
        };
        if rt.is_none() && !force && by == "admin-socket" && !self.holds.is_empty() {
            let list: Vec<String> = self.holds.iter().map(|h| format!("{} by {}", h.rt, h.by)).collect();
            return Err((409, format!(
                "held: {}; name the runtime (kiri-sched release <runtime>), or pass --force",
                list.join(", ")
            )));
        }
        let (gone, kept): (Vec<Hold>, Vec<Hold>) = self.holds.drain(..).partition(|h| take(h));
        self.holds = kept;
        Ok(gone)
    }

    /// The state of one runtime, as status shows it.
    pub fn state_of(&self, rt: &str, now: u64) -> &'static str {
        if self.fenced_until(rt, now).is_some() {
            "fenced"
        } else if let Some(s) = self.running.get(rt) {
            s.state.name()
        } else if self.backoff.get(rt).is_some_and(|&t| t > now) {
            "backoff"
        } else {
            "stopped"
        }
    }

    #[cfg(test)]
    pub fn hold(&self, now: u64) -> Option<&Hold> {
        self.active_holds(now).next()
    }

    #[cfg(test)]
    pub fn holds(&self, now: u64) -> Vec<&Hold> {
        self.active_holds(now).collect()
    }

    /// The Kiri daemon asks before a container gets GPUs. The container
    /// names its runtime with the `dev.kiri.sched.runtime` label, or by its
    /// Compose project (the stack name). A runtime that holds every card may
    /// ask for them any way; one that holds some may only name its own by
    /// index (`nvidia.com/gpu=1`, Compose `device_ids: ["1"]`).
    #[cfg(test)]
    pub fn lease_ok(&self, labels: &Value, devices: &[String]) -> Result<(), String> {
        self.lease(labels, devices, false).map(|_| ())
    }

    /// The scheduler's answer to a GPU start. `Ok(Some(devices))` grants it
    /// on those devices instead of the ones asked for: a child container
    /// of a runtime (`dev.kiri.sched.parent`) asking for `all` or a count
    /// gets the parent's cards, when the daemon said it can `narrow`.
    pub fn lease(
        &self,
        labels: &Value,
        devices: &[String],
        narrow: bool,
    ) -> Result<Option<Vec<String>>, String> {
        let label = |k: &str| {
            labels
                .get(k)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
        };
        let parent = label("dev.kiri.sched.parent");
        let rt = match (
            parent,
            label("dev.kiri.sched.runtime"),
            label("com.docker.compose.project"),
        ) {
            (Some(p), _, _) => {
                let Some(r) = self.cfg.runtime_for_stack(p).or_else(|| self.cfg.runtime(p)) else {
                    return Err(format!("dev.kiri.sched.parent {p:?} is not a GPU runtime in the scheduler's config"));
                };
                if !r.children {
                    return Err(format!(
                        "{} does not start GPU containers of its own (set children = true on it in sched.toml)",
                        r.name
                    ));
                }
                Some(r.name.clone())
            }
            (None, Some(name), _) => self.cfg.runtime(name).map(|r| r.name.clone()),
            (None, None, Some(stack)) => self.cfg.runtime_for_stack(stack).map(|r| r.name.clone()),
            (None, None, None) => None,
        };
        let Some(rt) = rt else {
            return Err("this container is not a GPU runtime in the scheduler's config".into());
        };
        let slot = match self.running.get(&rt) {
            Some(s) if !matches!(s.state, SlotState::Stopping | SlotState::Stuck) => s,
            _ => {
                let holders: Vec<String> = self
                    .running
                    .iter()
                    .filter(|(_, s)| !matches!(s.state, SlotState::Stopping | SlotState::Stuck))
                    .map(|(r, s)| format!("{r} {:?}", s.cards))
                    .collect();
                return Err(if holders.is_empty() {
                    format!(
                        "no runtime holds the GPUs right now ({}); {rt} starts only through the scheduler",
                        self.phase_name()
                    )
                } else {
                    format!(
                        "{rt} holds no GPUs (cards held: {}); ask the scheduler for it instead (kiri-sched use {rt})",
                        holders.join(", ")
                    )
                });
            }
        };
        if slot.cards.len() == self.cfg.sched.gpus {
            return Ok(None);
        }
        let own = |d: &str| {
            d.strip_prefix("nvidia.com/gpu=")
                .and_then(|i| i.parse::<usize>().ok())
                .is_some_and(|i| slot.cards.contains(&i))
        };
        if devices.iter().all(|d| own(d)) {
            return Ok(None);
        }
        // A child does not know which card its parent got: `all` or a
        // count it fits in becomes the parent's cards.
        if parent.is_some() && narrow {
            let mut want = 0;
            for d in devices {
                let n = match d.strip_prefix("nvidia.com/gpu=") {
                    Some("all") => slot.cards.len(),
                    Some(c) if c.starts_with("count:") => c[6..].parse::<usize>().unwrap_or(usize::MAX),
                    _ if own(d) => 1,
                    _ => usize::MAX,
                };
                want = want.max(n);
            }
            if want <= slot.cards.len() {
                return Ok(Some(
                    slot.cards[..want].iter().map(|c| format!("nvidia.com/gpu={c}")).collect(),
                ));
            }
        }
        let d = devices.iter().find(|d| !own(d)).map_or("", String::as_str);
        Err(format!(
            "{rt} holds card {:?} only; it may not ask for {d} (use device_ids: [\"${{KIRI_SCHED_GPUS}}\"])",
            slot.cards
        ))
    }

    /// One word for the box: fault, a switch's stage, idle or ready.
    pub fn phase_name(&self) -> &'static str {
        self.phase()["state"].as_str().map_or("idle", |s| match s {
            "fault" => "fault",
            "draining" => "draining",
            "stopping" => "stopping",
            "starting" => "starting",
            "ready" => "ready",
            _ => "idle",
        })
    }

    /// The box as one runtime would see it, for clients of the single-lease
    /// status: the switch under way, else the runtime that became ready
    /// last (`rts` lists every ready one).
    pub fn phase(&self) -> Value {
        if let Some(reason) = &self.fault {
            return json!({"state": "fault", "reason": reason});
        }
        if let Some(p) = &self.trans {
            let from = p.stop.first();
            let draining = p
                .stop
                .iter()
                .any(|rt| self.running.get(rt).is_some_and(|s| s.state == SlotState::Draining));
            let starting = p.start.as_ref().filter(|rt| {
                self.running
                    .get(*rt)
                    .is_some_and(|s| s.state == SlotState::Starting)
            });
            return match (draining, starting) {
                (true, _) => json!({"state": "draining", "from": from, "to": p.start}),
                (false, Some(rt)) => json!({"state": "starting", "rt": rt}),
                (false, None) => json!({"state": "stopping", "from": from, "to": p.start}),
            };
        }
        let ready = self
            .running
            .iter()
            .filter(|(_, s)| s.state == SlotState::Ready)
            .max_by_key(|(_, s)| s.since);
        match ready {
            Some((rt, s)) => {
                json!({"state": "ready", "rt": rt, "since": s.since, "rts": self.ready()})
            }
            None => json!({"state": "idle"}),
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
                let slot = self.running.get(&r.name);
                let state = self.state_of(&r.name, now);
                json!({
                    "name": r.name,
                    "state": state,
                    "cards": slot.map(|s| &s.cards),
                    "needs_cards": r.cards,
                    "models": r.models,
                    "inflight": self.inflight(&r.name),
                    "waiting": waiting.get(r.name.as_str()).cloned().unwrap_or(json!(null)),
                    "fenced_until": self.fenced_until(&r.name, now),
                    "last_switch_secs": self.switch_ms.get(&r.name).map(|ms| ms / 1000),
                })
            })
            .collect();
        let gpus: Vec<Value> = (0..self.cfg.sched.gpus)
            .map(|c| {
                let holder = self.running.iter().find(|(_, s)| s.cards.contains(&c));
                json!({
                    "card": c,
                    "runtime": holder.map(|(rt, _)| rt),
                    "state": holder.map_or("free", |(_, s)| s.state.name()),
                })
            })
            .collect();
        json!({
            "phase": self.phase(),
            "gpus": gpus,
            "switch": self.trans,
            "fault": self.fault,
            "hold": self.active_holds(now).next(),
            "holds": self.active_holds(now).collect::<Vec<_>>(),
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

    /// Runs a whole switch as the driver would, succeeding; returns what
    /// it started (or "" for a switch that only stops).
    fn switch_ok(c: &mut Core, now: u64) -> String {
        let p = c.want_switch(now).expect("a switch");
        run(c, now, &p);
        p.start.unwrap_or_default()
    }

    fn run(c: &mut Core, now: u64, p: &Plan) {
        c.switch_begin(p);
        c.switch_stopping();
        for rt in &p.stop {
            assert!(!c.holds_lease(rt), "{rt} still leased while stopping");
        }
        c.switch_starting();
        if let Some(to) = &p.start {
            assert!(c.holds_lease(to));
            c.switch_ready(now, to, 10 * S);
        }
        // The requests that waited for it go (as the front door's polls do).
        let ids: Vec<u64> = c.waiting.iter().map(|w| w.id).collect();
        for id in ids {
            if c.waiting.iter().any(|w| w.id == id && c.is_ready(&w.rt)) {
                assert_eq!(c.poll(now, id), Some(Admit::Go));
            }
        }
    }

    fn cards(c: &Core, rt: &str) -> Vec<usize> {
        c.slot(rt).map(|s| s.cards.clone()).unwrap_or_default()
    }

    #[test]
    fn boots_to_home_and_serves_it() {
        let mut c = core();
        assert_eq!(c.admit(0, "maya", Class::Agent), Admit::Wait(1));
        assert_eq!(switch_ok(&mut c, 0), "maya");
        assert_eq!(cards(&c, "maya"), [0, 1]);
        assert_eq!(c.poll(S, 1), None, "answered Go already");
        assert_eq!(c.admit(S, "maya", Class::Agent), Admit::Go);
        // Nothing else wanted: no switch.
        assert_eq!(c.want_switch(2 * S), None);
    }

    #[test]
    fn two_one_card_runtimes_share_the_box() {
        let mut c = core();
        switch_ok(&mut c, 0);
        // comfyui evicts maya (idle), takes card 0; card 1 stays free.
        c.admit(S, "comfyui", Class::Interactive);
        let p = c.want_switch(S).unwrap();
        assert_eq!(
            p,
            Plan {
                start: Some("comfyui".into()),
                cards: vec![0],
                stop: vec!["maya".into()]
            }
        );
        run(&mut c, S, &p);
        // mocap goes on the free card without touching comfyui, even busy.
        c.begin("comfyui");
        c.admit(2 * S, "mocap", Class::Agent);
        let p = c.want_switch(2 * S).unwrap();
        assert_eq!(p.stop, Vec::<String>::new());
        assert_eq!(p.cards, [1]);
        run(&mut c, 2 * S, &p);
        assert!(c.is_ready("comfyui") && c.is_ready("mocap"));
        assert_eq!(c.status(3 * S)["gpus"][1]["runtime"], "mocap");
        // maya needs both: it waits for both to be idle (agent work) …
        c.begin("mocap");
        c.admit(3 * S, "maya", Class::Agent);
        assert_eq!(c.want_switch(4 * S), None);
        c.end("comfyui");
        assert_eq!(c.want_switch(5 * S), None, "mocap is still busy");
        c.end("mocap");
        let p = c.want_switch(6 * S).unwrap();
        assert_eq!(p.start.as_deref(), Some("maya"));
        assert_eq!(p.stop.len(), 2);
        run(&mut c, 6 * S, &p);
        assert_eq!(cards(&c, "maya"), [0, 1]);
        assert_eq!(c.running().len(), 1);
    }

    #[test]
    fn a_one_card_request_evicts_only_one_card() {
        // A third one-card runtime, with both cards taken.
        let mut cfg = Config::parse(EXAMPLE).unwrap();
        cfg.runtimes.push(crate::config::Runtime {
            name: "splat".into(),
            ..cfg.runtime("mocap").unwrap().clone()
        });
        cfg.runtimes.last_mut().unwrap().stack = "splat".into();
        cfg.runtimes.last_mut().unwrap().unit = "kiri-run@splat.service".into();
        cfg.runtimes.last_mut().unwrap().upstream = "http://127.0.0.1:8300".into();
        let mut c2 = Core::new(Arc::new(cfg), Saved::default());
        c2.admit(0, "comfyui", Class::Interactive);
        switch_ok(&mut c2, 0);
        c2.admit(0, "mocap", Class::Interactive);
        switch_ok(&mut c2, 0);
        c2.begin("mocap");
        c2.admit(S, "splat", Class::Interactive);
        let p = c2.want_switch(S).unwrap();
        // The idle one goes, the busy one stays.
        assert_eq!(p.stop, ["comfyui"]);
        assert_eq!(p.cards, [0]);
    }

    #[test]
    fn an_idle_render_server_gives_the_gpus_back_to_home() {
        let mut c = core();
        switch_ok(&mut c, 0);
        c.set_hold(0, "comfyui", HoldFor::Until(Some(100 * S)), "cody", false).unwrap();
        assert_eq!(switch_ok(&mut c, S), "comfyui");
        // Held, then busy: it stays.
        assert_eq!(c.want_switch(99 * S), None);
        c.begin("comfyui");
        assert_eq!(c.want_switch(2000 * S), None);
        c.end("comfyui");
        // The idle clock starts when it is last seen idle, not at load.
        assert_eq!(c.want_switch(2001 * S), None);
        assert_eq!(c.want_switch(2600 * S), None);
        // Work in between restarts it.
        c.begin("comfyui");
        assert_eq!(c.want_switch(2601 * S), None);
        c.end("comfyui");
        assert_eq!(c.want_switch(2602 * S), None);
        assert_eq!(c.want_switch(3201 * S), None);
        let p = c.want_switch(3202 * S).unwrap();
        assert_eq!(p.start.as_deref(), Some("maya"));
        assert_eq!(p.stop, ["comfyui"]);
        assert_eq!(c.why(), "idle: home_after_idle_secs");
        // Home itself never times out.
        run(&mut c, 3203 * S, &p);
        assert_eq!(c.want_switch(99_999 * S), None);
    }

    #[test]
    fn an_idle_one_card_runtime_beside_a_busy_one_just_stops() {
        let mut c = core();
        c.admit(0, "comfyui", Class::Interactive);
        switch_ok(&mut c, 0);
        c.admit(0, "mocap", Class::Interactive);
        switch_ok(&mut c, 0);
        c.begin("mocap");
        assert_eq!(c.want_switch(S), None);
        let p = c.want_switch(602 * S).unwrap();
        assert_eq!(
            p,
            Plan {
                start: None,
                cards: vec![],
                stop: vec!["comfyui".into()]
            }
        );
        run(&mut c, 602 * S, &p);
        assert_eq!(c.switching(), None);
        assert_eq!(c.status(0)["gpus"][0]["state"], "free");
        // mocap (no idle limit) ends its work; the box goes home only once
        // it is empty, and not back to comfyui.
        c.end("mocap");
        assert_eq!(c.want_switch(700 * S), None);
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
        let p = c.want_switch(181 * S).unwrap();
        assert_eq!(p.start.as_deref(), Some("comfyui"));
        assert_eq!(c.why(), "request (current runtime dwelt)");
        c.switch_begin(&p);
        assert_eq!(c.phase()["state"], "draining");
        assert_eq!(c.phase()["from"], "maya");
        // While draining, maya still holds the lease; new maya work waits.
        assert!(c.holds_lease("maya"));
        assert!(matches!(
            c.admit(182 * S, "maya", Class::Agent),
            Admit::Wait(_)
        ));
        c.end("maya");
        c.switch_stopping();
        assert!(!c.holds_lease("maya"));
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
        let p = c.want_switch(602 * S).unwrap();
        assert_eq!(p.start.as_deref(), Some("comfyui"));
    }

    #[test]
    fn batch_requests_take_only_free_cards() {
        let mut c = core();
        switch_ok(&mut c, 0);
        c.admit(S, "comfyui", Class::Batch);
        assert_eq!(c.want_switch(10 * S), None);
        // With a card free, batch work may use it.
        let mut c = core();
        c.admit(0, "mocap", Class::Interactive);
        switch_ok(&mut c, 0);
        c.admit(S, "comfyui", Class::Batch);
        let p = c.want_switch(S).unwrap();
        assert_eq!((p.cards, p.stop), (vec![1], vec![]));
    }

    #[test]
    fn holds_win_and_refuse_only_what_cannot_fit_beside_them() {
        let mut c = core();
        switch_ok(&mut c, 0);
        c.begin("maya");
        c.set_hold(0, "comfyui", HoldFor::Until(Some(3600 * S)), "cody", false).unwrap();
        // A hold evicts even a busy runtime that has not dwelt.
        let p = c.want_switch(S).unwrap();
        assert_eq!(p.start.as_deref(), Some("comfyui"));
        c.switch_begin(&p);
        c.end("maya");
        c.switch_stopping();
        c.switch_starting();
        c.switch_ready(2 * S, "comfyui", S);
        // maya needs comfyui's card: refused.
        match c.admit(3 * S, "maya", Class::Interactive) {
            Admit::Refuse {
                status: 503,
                retry_after,
                ..
            } => assert!(retry_after > 3000),
            other => panic!("{other:?}"),
        }
        // mocap fits beside it, and never evicts it.
        assert!(matches!(
            c.admit(3 * S, "mocap", Class::Interactive),
            Admit::Wait(_)
        ));
        let p = c.want_switch(3 * S).unwrap();
        assert_eq!((p.start.as_deref(), p.stop.len()), (Some("mocap"), 0));
        run(&mut c, 3 * S, &p);
        // Past its end the hold is gone and maya may come back.
        assert!(matches!(
            c.admit(3601 * S, "maya", Class::Interactive),
            Admit::Wait(_)
        ));
        let p = c.want_switch(3601 * S).unwrap();
        assert_eq!(p.start.as_deref(), Some("maya"));
        assert_eq!(p.stop.len(), 2);
    }

    #[test]
    fn a_two_card_hold_evicts_both() {
        let mut c = core();
        c.admit(0, "comfyui", Class::Interactive);
        switch_ok(&mut c, 0);
        c.admit(0, "mocap", Class::Interactive);
        switch_ok(&mut c, 0);
        c.begin("comfyui");
        c.begin("mocap");
        c.set_hold(0, "maya", HoldFor::Until(None), "cody", false).unwrap();
        let p = c.want_switch(S).unwrap();
        assert_eq!(p.start.as_deref(), Some("maya"));
        assert_eq!(p.cards, [0, 1]);
        assert_eq!(p.stop.len(), 2);
    }

    #[test]
    fn one_card_holds_by_different_callers_share_the_box() {
        let mut c = core();
        switch_ok(&mut c, 0);
        c.set_hold(0, "comfyui", HoldFor::Until(None), "admin-socket:render", false).unwrap();
        // A second caller holds the other one-card runtime: no 409.
        c.set_hold(0, "mocap", HoldFor::Until(None), "admin-socket:mocap-job", false).unwrap();
        assert_eq!(c.holds(0).len(), 2);
        let p = c.want_switch(S).unwrap();
        assert_eq!(p.start.as_deref(), Some("comfyui"));
        run(&mut c, S, &p);
        let p = c.want_switch(2 * S).unwrap();
        assert_eq!((p.start.as_deref(), p.stop.len()), (Some("mocap"), 0));
        run(&mut c, 2 * S, &p);
        assert!(c.is_ready("comfyui") && c.is_ready("mocap"));
        assert_eq!(c.status(3 * S)["holds"].as_array().unwrap().len(), 2);
        // A two-card hold cannot fit beside them: 409 naming both.
        let e = c.set_hold(3 * S, "maya", HoldFor::Until(None), "admin-socket:x", false).unwrap_err();
        assert_eq!(e.0, 409);
        assert!(e.1.contains("comfyui") && e.1.contains("mocap"), "{}", e.1);
        // Neither is ever evicted while held, even idle and asked for.
        c.admit(4 * S, "maya", Class::Interactive);
        assert_eq!(c.want_switch(5000 * S), None);
        // Releasing one of them still leaves no room for maya.
        assert_eq!(c.release(6000 * S, None, "admin-socket:mocap-job", false).unwrap().len(), 1);
        assert_eq!(c.held(6000 * S), BTreeSet::from(["comfyui".to_string()]));
    }

    #[test]
    fn force_drops_only_the_holds_in_the_way() {
        let mut cfg = Config::parse(EXAMPLE).unwrap();
        cfg.runtimes.push(crate::config::Runtime {
            name: "splat".into(),
            ..cfg.runtime("mocap").unwrap().clone()
        });
        let mut c = Core::new(Arc::new(cfg), Saved::default());
        c.set_hold(0, "comfyui", HoldFor::Until(None), "admin-socket:a", false).unwrap();
        c.set_hold(0, "mocap", HoldFor::Until(None), "admin-socket:b", false).unwrap();
        // splat needs one card: both are held, so 409 without --force.
        assert_eq!(c.set_hold(S, "splat", HoldFor::Until(None), "admin-socket:c", false).unwrap_err().0, 409);
        // --force drops only one hold (the newest), not both.
        c.set_hold(S, "splat", HoldFor::Until(None), "admin-socket:c", true).unwrap();
        assert_eq!(
            c.held(S),
            BTreeSet::from(["comfyui".to_string(), "splat".to_string()])
        );
        // A two-card hold with --force drops everything in the way.
        c.set_hold(2 * S, "maya", HoldFor::Until(None), "admin-socket:d", true).unwrap();
        assert_eq!(c.held(2 * S), BTreeSet::from(["maya".to_string()]));
    }

    #[test]
    fn a_caller_moving_its_own_hold_frees_its_old_card() {
        let mut c = core();
        c.set_hold(0, "comfyui", HoldFor::Until(None), "admin-socket:a", false).unwrap();
        c.set_hold(0, "mocap", HoldFor::Until(None), "admin-socket:b", false).unwrap();
        // b moves to maya: b's own mocap hold is replaced, but a's comfyui
        // hold is still in the way, so it is refused and b keeps mocap.
        assert_eq!(c.set_hold(S, "maya", HoldFor::Until(None), "admin-socket:b", false).unwrap_err().0, 409);
        assert_eq!(c.holds(S).len(), 2);
        c.release(S, None, "admin-socket:a", false).unwrap();
        c.set_hold(2 * S, "maya", HoldFor::Until(None), "admin-socket:b", false).unwrap();
        assert_eq!(c.held(2 * S), BTreeSet::from(["maya".to_string()]));
    }

    #[test]
    fn bare_uses_switch_and_keep_what_fits() {
        let mut c = core();
        // `use comfyui` then `use mocap` with no --by: both fit, both kept.
        c.set_hold(0, "comfyui", HoldFor::AfterReady(60), "admin-socket", false).unwrap();
        c.set_hold(0, "mocap", HoldFor::AfterReady(60), "admin-socket", false).unwrap();
        assert_eq!(c.holds(0).len(), 2);
        // `use maya` (two cards) replaces the bare holds in its way.
        c.set_hold(S, "maya", HoldFor::AfterReady(60), "admin-socket", false).unwrap();
        assert_eq!(c.held(S), BTreeSet::from(["maya".to_string()]));
        // But never a named caller's hold.
        c.release(S, Some("maya"), "admin-socket", false).unwrap();
        c.set_hold(S, "comfyui", HoldFor::Until(None), "admin-socket:render", false).unwrap();
        assert_eq!(c.set_hold(2 * S, "maya", HoldFor::AfterReady(60), "admin-socket", false).unwrap_err().0, 409);
    }

    #[test]
    fn several_holds_survive_a_restart() {
        let mut c = core();
        c.set_hold(0, "comfyui", HoldFor::Until(None), "admin-socket:a", false).unwrap();
        c.set_hold(0, "mocap", HoldFor::Until(None), "admin-socket:b", false).unwrap();
        let saved = c.saved();
        let json = serde_json::to_string(&saved).unwrap();
        // Old readers still see one hold.
        assert!(json.contains("\"hold\":{"), "{json}");
        let back: Saved = serde_json::from_str(&json).unwrap();
        let c2 = Core::new(Arc::new(Config::parse(EXAMPLE).unwrap()), back);
        assert_eq!(c2.holds(0).len(), 2);
    }

    #[test]
    fn failures_back_off_then_fence() {
        let mut c = core();
        c.admit(0, "comfyui", Class::Interactive);
        let mut now = 0;
        for i in 0..3 {
            let p = c.want_switch(now).unwrap_or_else(|| panic!("try {i}"));
            assert_eq!(p.start.as_deref(), Some("comfyui"), "try {i}");
            c.switch_begin(&p);
            c.switch_starting();
            c.switch_failed(now, "comfyui");
            // Backed off: the box goes home meanwhile, not round in a loop.
            if i < 2 {
                let p = c.want_switch(now + 1).unwrap();
                assert_eq!(p.start.as_deref(), Some("maya"));
                now += 100 * S;
                // (maya is never actually started here.)
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
    fn fault_stops_every_switch_until_reset() {
        let mut c = core();
        let p = c.want_switch(0).unwrap();
        c.switch_begin(&p);
        c.switch_fault("GPU 1 still uses 9000 MiB".into());
        assert!(c.running().is_empty());
        assert_eq!(c.want_switch(S), None);
        assert_eq!(c.phase_name(), "fault");
        assert!(matches!(
            c.admit(S, "maya", Class::Interactive),
            Admit::Refuse { status: 503, .. }
        ));
        assert!(c.reset());
        assert_eq!(switch_ok(&mut c, S), "maya");
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
        let all = ["nvidia.com/gpu=all".to_owned()];
        let maya = json!({"com.docker.compose.project": "maya"});
        let comfy = json!({"dev.kiri.sched.runtime": "comfyui"});
        let mocap = json!({"dev.kiri.sched.runtime": "mocap"});
        assert!(c.lease_ok(&maya, &all).is_err());
        let p = c.want_switch(0).unwrap();
        c.switch_begin(&p);
        assert!(
            c.lease_ok(&maya, &all).is_err(),
            "not before the GPUs are seen empty"
        );
        c.switch_starting();
        assert!(c.lease_ok(&maya, &all).is_ok());
        assert!(c.lease_ok(&comfy, &all).unwrap_err().contains("maya [0, 1]"));
        assert!(c.lease_ok(&json!({}), &all).is_err());
        assert!(
            c.lease_ok(&json!({"com.docker.compose.project": "dashboard"}), &all)
                .is_err()
        );
        c.switch_ready(0, "maya", S);
        // A one-card runtime may name only its own card.
        c.admit(S, "comfyui", Class::Interactive);
        switch_ok(&mut c, S);
        c.admit(S, "mocap", Class::Interactive);
        let p = c.want_switch(S).unwrap();
        c.switch_begin(&p);
        c.switch_starting();
        let dev = |d: &str| vec![d.to_owned()];
        assert!(c.lease_ok(&comfy, &dev("nvidia.com/gpu=0")).is_ok());
        assert!(c.lease_ok(&mocap, &dev("nvidia.com/gpu=1")).is_ok());
        for bad in [
            "nvidia.com/gpu=0",
            "nvidia.com/gpu=all",
            "nvidia.com/gpu=count:1",
            "nvidia.com/gpu=GPU-1234",
        ] {
            assert!(c.lease_ok(&mocap, &dev(bad)).is_err(), "{bad}");
        }
        assert!(
            c.lease_ok(&mocap, &["nvidia.com/gpu=1".into(), "nvidia.com/gpu=0".into()])
                .is_err()
        );
    }

    #[test]
    fn children_inherit_the_lease_on_their_parents_cards_only() {
        let mut c = core();
        let child = json!({"dev.kiri.sched.parent": "mocap"});
        let all = ["nvidia.com/gpu=all".to_owned()];
        // Nothing holds mocap's cards: refused.
        assert!(c.lease(&child, &all, true).is_err());
        c.admit(0, "comfyui", Class::Interactive);
        switch_ok(&mut c, 0);
        c.admit(0, "mocap", Class::Interactive);
        switch_ok(&mut c, 0);
        let card = c.running()["mocap"].cards[0];
        let mine = format!("nvidia.com/gpu={card}");
        let other = format!("nvidia.com/gpu={}", 1 - card);
        // all / count:1 become mocap's card; its own card is fine as is.
        assert_eq!(c.lease(&child, &all, true).unwrap(), Some(vec![mine.clone()]));
        assert_eq!(
            c.lease(&child, &["nvidia.com/gpu=count:1".into()], true).unwrap(),
            Some(vec![mine.clone()])
        );
        assert_eq!(c.lease(&child, std::slice::from_ref(&mine), true).unwrap(), None);
        // Never the other card, more cards than it has, or a UUID.
        for bad in [other.as_str(), "nvidia.com/gpu=count:2", "nvidia.com/gpu=GPU-1234"] {
            assert!(c.lease(&child, &[bad.to_owned()], true).is_err(), "{bad}");
        }
        // A daemon that cannot narrow gets a no for `all`.
        assert!(c.lease(&child, &all, false).is_err());
        // Only runtimes with children = true have children.
        let comfy_child = json!({"dev.kiri.sched.parent": "comfyui"});
        assert!(c.lease(&comfy_child, &all, true).unwrap_err().contains("children = true"));
        assert!(c.lease(&json!({"dev.kiri.sched.parent": "nope"}), &all, true).is_err());
        // The parent label wins over a runtime label naming someone else.
        let sneaky = json!({"dev.kiri.sched.parent": "mocap", "dev.kiri.sched.runtime": "comfyui"});
        assert_eq!(c.lease(&sneaky, &all, true).unwrap(), Some(vec![mine]));
    }

    #[test]
    fn saved_state_comes_back() {
        let mut c = core();
        c.set_hold(0, "comfyui", HoldFor::Until(None), "cody", false).unwrap();
        assert_eq!(switch_ok(&mut c, 0), "comfyui");
        let saved: Saved =
            serde_json::from_value(serde_json::to_value(c.saved()).unwrap()).unwrap();
        assert_eq!(saved.cards["comfyui"], [0]);
        let mut c = Core::new(Arc::new(Config::parse(EXAMPLE).unwrap()), saved);
        // Found running: adopted on its own card, not another.
        assert_eq!(c.boot_cards("comfyui"), Some(vec![0]));
        assert_eq!(c.boot_cards("maya"), Some(vec![0, 1]));
        assert_eq!(c.boot_cards("mocap"), None);
        assert!(!c.adopt(0, "comfyui", &[0, 1]));
        assert!(c.adopt(0, "comfyui", &[0]));
        assert!(!c.adopt(0, "mocap", &[0]), "card 0 is taken");
        assert_eq!(c.want_switch(0), None);
        // A saved hold on a runtime no longer in the config is dropped.
        let old = Saved {
            hold: Some(Hold {
                rt: "gone".into(),
                until: None,
                by: "x".into(),
                after_ready_secs: None,
                start_by: None,
            }),
            ..Saved::default()
        };
        let c = Core::new(Arc::new(Config::parse(EXAMPLE).unwrap()), old);
        assert!(c.hold(0).is_none());
    }

    #[test]
    fn a_use_counts_its_time_from_ready_and_is_never_replaced_by_accident() {
        let mut c = core();
        switch_ok(&mut c, 0);
        // `use comfyui` for 180 s while maya is still loading something
        // slow: the time does not run until comfyui is ready.
        c.set_hold(0, "comfyui", HoldFor::AfterReady(180), "admin-socket:lab", false)
            .unwrap();
        let p = c.want_switch(S).unwrap();
        c.switch_begin(&p);
        c.switch_stopping();
        c.switch_starting();
        // Loading takes 800 s, far past the 180 s.
        assert!(c.want_switch(800 * S).is_none());
        assert!(c.hold(800 * S).is_some(), "the hold waits for ready");
        c.switch_ready(800 * S, "comfyui", 799 * S);
        assert_eq!(c.hold(900 * S).unwrap().until, Some(980 * S));
        // Someone else's use of another runtime is refused, not silently
        // taking over; the same runtime, the owner, or --force may.
        let e = c
            .set_hold(901 * S, "maya", HoldFor::AfterReady(60), "admin-socket", false)
            .unwrap_err();
        assert_eq!(e.0, 409);
        assert!(e.1.contains("comfyui by admin-socket:lab"), "{}", e.1);
        // A second hold on the same runtime is fine.
        assert!(c.set_hold(901 * S, "comfyui", HoldFor::AfterReady(600), "admin-socket", false).is_ok());
        assert_eq!(c.holds(901 * S).len(), 2);
        // lab moving its own hold to maya is refused while the bare
        // caller's comfyui hold remains (maya needs comfyui's card).
        assert_eq!(c.set_hold(902 * S, "maya", HoldFor::Until(None), "admin-socket:lab", false).unwrap_err().0, 409);
        // Release: a bare caller must name the runtime; a named one ends
        // only its own holds.
        assert_eq!(c.release(903 * S, None, "admin-socket", false).unwrap_err().0, 409);
        assert!(c.release(903 * S, None, "admin-socket:other", false).unwrap().is_empty());
        assert_eq!(c.release(903 * S, Some("comfyui"), "admin-socket", false).unwrap().len(), 2);
        c.set_hold(902 * S, "maya", HoldFor::Until(None), "admin-socket:lab", false).unwrap();
        assert!(c.release(903 * S, None, "admin-socket:lab", false).unwrap().len() == 1);
        c.set_hold(904 * S, "maya", HoldFor::Until(None), "admin-socket:lab", false).unwrap();
        assert!(c.release(905 * S, Some("maya"), "admin-socket", false).unwrap().len() == 1);
        c.set_hold(904 * S, "maya", HoldFor::Until(None), "admin-socket:lab", false).unwrap();
        assert!(c.release(905 * S, None, "admin-socket", true).unwrap().len() == 1);
        // A use of a runtime that never gets ready is given up in time.
        c.set_hold(1000 * S, "mocap", HoldFor::AfterReady(60), "t", false).unwrap();
        let p = c.want_switch(1000 * S).unwrap();
        c.switch_begin(&p);
        c.switch_starting();
        c.switch_failed(1000 * S, "mocap");
        assert!(c.hold(1500 * S).is_some());
        assert!(c.want_switch(5000 * S).is_some());
        assert!(c.hold(5000 * S).is_none(), "start_by passed");
        // Fenced or faulted: use is refused outright.
        c.fault = Some("x".into());
        assert_eq!(c.set_hold(5000 * S, "maya", HoldFor::AfterReady(60), "t", false).unwrap_err().0, 503);
    }

    /// A long random run of requests, completions, holds, failures and
    /// crashes, checking the invariants after every step.
    #[test]
    fn invariants_hold_over_random_runs() {
        for seed in 1..=300u64 {
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
            let rts = ["maya", "comfyui", "mocap"];
            let classes = [Class::Interactive, Class::Agent, Class::Batch];
            let mut both_one_card = 0;
            let mut many_holds = 0;
            for _ in 0..2000 {
                now += next(20) * S;
                match next(10) {
                    0..=2 => {
                        if let Admit::Wait(id) =
                            c.admit(now, rts[next(3) as usize], classes[next(3) as usize])
                        {
                            waits.push(id);
                        }
                    }
                    3 => {
                        let rt = rts[next(3) as usize];
                        if c.is_ready(rt) {
                            c.begin(rt);
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
                        let by = ["a", "b", "c"][next(3) as usize];
                        if next(3) != 0 {
                            let _ = c.set_hold(
                                now,
                                rts[next(3) as usize],
                                HoldFor::Until(Some(now + next(600) * S)),
                                by,
                                next(4) == 0,
                            );
                        } else {
                            let _ = c.release(now, None, by, false);
                        }
                        // Invariant 5: the live holds always fit together.
                        let need: usize = c.held(now).iter().map(|rt| c.cfg.cards(rt)).sum();
                        assert!(need <= c.cfg.sched.gpus, "seed {seed}: holds overlap");
                        if c.holds(now).len() > 1 {
                            many_holds += 1;
                        }
                    }
                    6 => {
                        let ready = c.ready();
                        if !ready.is_empty() {
                            let rt = ready[next(ready.len() as u64) as usize].clone();
                            c.crashed(now, &rt);
                        }
                    }
                    _ => {
                        let held = c.held(now);
                        if let Some(p) = c.want_switch(now) {
                            // Invariant 2: a live hold is never evicted.
                            for h in &held {
                                assert!(!p.stop.contains(h), "seed {seed}: evicted the hold {h}");
                            }
                            if let Some(to) = &p.start {
                                assert!(!c.running().contains_key(to), "seed {seed}: switch to itself");
                                // Invariant 4: never a fenced runtime.
                                assert!(c.fenced_until(to, now).is_none());
                                assert_eq!(p.cards.len(), c.cfg.cards(to));
                            }
                            for rt in &p.stop {
                                assert!(c.is_ready(rt), "seed {seed}: stops {rt} not ready");
                            }
                            c.switch_begin(&p);
                            c.switch_stopping();
                            for rt in &p.stop {
                                assert!(!c.holds_lease(rt), "seed {seed}: lease held while stopping");
                            }
                            c.switch_starting();
                            if let Some(to) = &p.start {
                                if next(5) == 0 {
                                    c.switch_failed(now, to);
                                } else {
                                    c.switch_ready(now, to, S);
                                }
                            }
                        }
                    }
                }
                waits.retain(|&id| c.poll(now, id).is_none());
                // Invariant 3: bounded.
                assert!(c.waiting.len() <= c.cfg.sched.max_waiting);
                // Invariant 1: no card has two holders, every holder is a
                // configured runtime with the cards it needs.
                let mut seen = BTreeSet::new();
                for (rt, s) in c.running() {
                    assert_eq!(s.cards.len(), c.cfg.cards(rt), "seed {seed}");
                    for &card in &s.cards {
                        assert!(card < c.cfg.sched.gpus);
                        assert!(seen.insert(card), "seed {seed}: card {card} held twice");
                    }
                }
                assert!(c.switching().is_none());
                if c.is_ready("comfyui") && c.is_ready("mocap") {
                    both_one_card += 1;
                }
            }
            if seed == 1 {
                assert!(both_one_card > 0, "the random run never shared the box");
                assert!(many_holds > 0, "the random run never held two runtimes");
            }
        }
    }
}
