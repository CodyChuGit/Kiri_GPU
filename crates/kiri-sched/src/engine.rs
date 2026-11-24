//! Carries out what `core` decides: the switch sequence with a deadline on
//! every step, health checks of the runtime on the GPUs, the light
//! services' restarts, saving state, and the audit log.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::{Value, json};
use tokio::sync::watch;
use tokio::time::{Instant, sleep, timeout};

use crate::config::{Config, Restart};
use crate::core::{Core, Plan, Saved};
use crate::host::Host;

/// One line of JSON on stderr per event (journald keeps it): every switch,
/// failure, restart, fence and refusal, with its reason.
pub fn audit(event: &str, fields: Value) {
    let mut v = json!({"event": event, "at": wall_ms()});
    if let (Some(o), Value::Object(f)) = (v.as_object_mut(), fields) {
        o.extend(f);
    }
    eprintln!("{v}");
}

fn wall_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ServiceStatus {
    pub state: &'static str,
    pub restarts: u32,
    pub last_error: Option<String>,
    pub fenced_until: Option<u64>,
    #[serde(skip)]
    fails: Vec<u64>,
    #[serde(skip)]
    next_try: u64,
    #[serde(skip)]
    unhealthy_since: Option<u64>,
}

#[derive(Debug)]
pub struct Engine<H: Host> {
    core: Mutex<Core>,
    pub host: H,
    services: Mutex<BTreeMap<String, ServiceStatus>>,
    /// Bumped on every change, for waiting requests and /box/events.
    changed: watch::Sender<u64>,
    state_file: Option<PathBuf>,
    /// The clock: Unix ms at `base`, moved on by tokio's (pausable) clock.
    base_ms: u64,
    base: Instant,
}

impl<H: Host> Engine<H> {
    pub fn new(cfg: Arc<Config>, host: H, state_file: Option<PathBuf>) -> Arc<Engine<H>> {
        let saved = state_file.as_deref().map(load_saved).unwrap_or_default();
        let services = cfg
            .services
            .iter()
            .map(|s| {
                (
                    s.name.clone(),
                    ServiceStatus {
                        state: "unknown",
                        ..Default::default()
                    },
                )
            })
            .collect();
        let (changed, _) = watch::channel(0);
        Arc::new(Engine {
            core: Mutex::new(Core::new(cfg, saved)),
            host,
            services: Mutex::new(services),
            changed,
            state_file,
            base_ms: wall_ms(),
            base: Instant::now(),
        })
    }

    pub fn now(&self) -> u64 {
        self.base_ms + self.base.elapsed().as_millis() as u64
    }

    pub fn core(&self) -> MutexGuard<'_, Core> {
        self.core.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn config(&self) -> Arc<Config> {
        self.core().config().clone()
    }

    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.changed.subscribe()
    }

    pub fn notify(&self) {
        self.changed.send_modify(|n| *n += 1);
    }

    /// Saves what must survive a restart, then tells everyone who waits.
    pub fn commit(&self) {
        let saved = self.core().saved();
        if let Some(path) = &self.state_file
            && let Err(e) = save_atomic(path, &saved)
        {
            audit(
                "state-save-failed",
                json!({"path": path, "error": e.to_string()}),
            );
        }
        self.notify();
    }

    pub fn status(&self) -> Value {
        let now = self.now();
        let mut v = self.core().status(now);
        v["services"] =
            serde_json::to_value(&*self.services.lock().unwrap_or_else(|e| e.into_inner()))
                .unwrap_or_default();
        v["now"] = now.into();
        v
    }

    /// At startup: keep every GPU runtime that is already up and answering
    /// on cards it can keep (all of them, or the ones saved for it), and
    /// stop every other, so the box never runs two on one card after a
    /// crash or a reboot.
    pub async fn reconcile(&self) {
        let cfg = self.config();
        for r in &cfg.runtimes {
            if !self.host.is_active(&r.unit).await {
                continue;
            }
            let cards = self.core().boot_cards(&r.name);
            let kept = match cards {
                Some(cards) if self.host.probe(&r.ready).await => {
                    self.core().adopt(self.now(), &r.name, &cards).then_some(cards)
                }
                _ => None,
            };
            if let Some(cards) = kept {
                audit("adopt", json!({"runtime": r.name, "cards": cards}));
                continue;
            }
            audit(
                "reconcile-stop",
                json!({"runtime": r.name, "why": "running at startup but not answering, or on cards it cannot keep"}),
            );
            if let Err(e) = self.stop_within(&r.unit, cfg.sched.stop_timeout_secs).await {
                self.core()
                    .switch_fault(format!("could not stop {} at startup: {e}", r.name));
                audit("fault", json!({"reason": e}));
                self.commit();
                return;
            }
        }
        self.commit();
    }

    async fn stop_within(&self, unit: &str, secs: u64) -> Result<(), String> {
        match timeout(Duration::from_secs(secs), self.host.stop(unit)).await {
            Ok(r) => r,
            Err(_) => Err(format!("stop of {unit} took over {secs} s")),
        }
    }

    /// One pass of the main loop: start a switch if one is due, else check
    /// the runtimes on the GPUs.
    pub async fn step(&self, health: &mut Health) {
        let plan = {
            let mut core = self.core();
            let plan = core.want_switch(self.now());
            if let Some(p) = &plan {
                core.switch_begin(p);
            }
            plan
        };
        match plan {
            Some(p) => {
                for rt in &p.stop {
                    health.failing_since.remove(rt);
                }
                self.notify();
                self.switch(&p).await;
            }
            None => self.check_ready(health).await,
        }
    }

    /// The switch sequence; `core.switch_begin` was called.
    async fn switch(&self, plan: &Plan) {
        let cfg = self.config();
        let started = Instant::now();
        let why = self.core().why();
        audit(
            "switch-begin",
            json!({"from": plan.stop, "to": plan.start, "cards": plan.cards, "why": why}),
        );
        // 1. Drain: in-flight requests get `drain_secs` to finish.
        let deadline = Instant::now() + Duration::from_secs(cfg.sched.drain_secs);
        while plan.stop.iter().any(|f| self.core().inflight(f) > 0) && Instant::now() < deadline {
            sleep(Duration::from_millis(200)).await;
        }
        for f in &plan.stop {
            let left = self.core().inflight(f);
            if left > 0 {
                audit("drain-timeout", json!({"runtime": f, "inflight": left}));
            }
        }
        self.core().switch_stopping();
        self.notify();
        // 2. Stop what the plan stops, and any GPU runtime that runs without
        // the scheduler knowing (the target too, to start it clean).
        let mut freed: Vec<usize> = plan.cards.clone();
        let mut to_stop = Vec::new();
        for r in &cfg.runtimes {
            let (planned, known) = {
                let core = self.core();
                let slot = core.slot(&r.name);
                if let Some(s) = slot.filter(|_| plan.stop.contains(&r.name)) {
                    freed.extend(&s.cards);
                }
                (plan.stop.contains(&r.name), slot.is_some())
            };
            let stray = !planned && !known && self.host.is_active(&r.unit).await;
            if stray {
                audit("stop-stray", json!({"runtime": r.name}));
            }
            if planned || stray {
                to_stop.push(r);
            }
        }
        // All at once: two one-card runtimes giving way to a two-card one
        // cost one stop, not two.
        let secs = cfg.sched.stop_timeout_secs;
        let results = join_all(
            to_stop
                .iter()
                .map(|r| Box::pin(self.stop_within(&r.unit, secs)) as StopFut<'_>)
                .collect(),
        )
        .await;
        for (r, res) in to_stop.iter().zip(results) {
            if let Err(e) = res {
                return self.fault(format!("could not stop {}: {e}", r.name));
            }
        }
        freed.sort_unstable();
        freed.dedup();
        // 3. The cards it frees and the ones it takes must be seen empty
        // before anything else gets them.
        let deadline = Instant::now() + Duration::from_secs(cfg.sched.vram_wait_secs);
        loop {
            let res = self.host.vram_used().await;
            let busy: Result<Vec<(usize, Option<u64>)>, String> = res.map(|used| {
                freed
                    .iter()
                    .map(|&c| (c, used.get(c).copied()))
                    .filter(|(_, m)| m.is_none_or(|m| m > cfg.sched.vram_free_mib))
                    .collect()
            });
            match busy {
                Ok(b) if b.is_empty() => break,
                Ok(b) if Instant::now() >= deadline => {
                    return self.fault(format!("GPU memory still in use after stopping (card, MiB): {b:?} (limit {} MiB)", cfg.sched.vram_free_mib));
                }
                Err(e) if Instant::now() >= deadline => {
                    return self.fault(format!("cannot read GPU memory: {e}"));
                }
                _ => sleep(Duration::from_millis(500)).await,
            }
        }
        // 4. Start the target on its cards and wait for it to answer.
        let Some(to) = plan.start.as_deref() else {
            self.core().switch_starting();
            audit("switch-done", json!({"stopped": plan.stop, "why": why}));
            return self.commit();
        };
        let Some(rt) = cfg.runtime(to).cloned() else {
            return self.fault(format!("runtime {to} vanished from the config"));
        };
        if let Err(e) = self.host.assign_cards(&rt.stack, &plan.cards).await {
            return self.fault(format!("could not hand {to} its cards: {e}"));
        }
        self.core().switch_starting();
        self.commit();
        let ok = match self.host.start(&rt.unit).await {
            Err(e) => Err(e),
            Ok(()) => {
                self.wait_ready(&rt.unit, &rt.ready, rt.start_timeout_secs)
                    .await
            }
        };
        match ok {
            Ok(()) => {
                let took = started.elapsed().as_millis() as u64;
                self.core().switch_ready(self.now(), to, took);
                audit(
                    "switch-ready",
                    json!({"runtime": to, "cards": plan.cards, "secs": took / 1000}),
                );
                self.commit();
            }
            Err(e) => {
                audit("start-failed", json!({"runtime": to, "error": e}));
                // Roll back: the failed runtime is stopped again so its
                // cards are empty for whatever comes next.
                if let Err(e) = self
                    .stop_within(&rt.unit, cfg.sched.stop_timeout_secs)
                    .await
                {
                    return self
                        .fault(format!("{to} failed to start and then would not stop: {e}"));
                }
                self.core().switch_failed(self.now(), to);
                self.commit();
            }
        }
    }

    async fn wait_ready(&self, unit: &str, url: &str, secs: u64) -> Result<(), String> {
        let deadline = Instant::now() + Duration::from_secs(secs);
        // A unit can be "inactive" for a moment right after a --no-block start.
        let grace = Instant::now() + Duration::from_secs(10);
        loop {
            if self.host.probe(url).await {
                return Ok(());
            }
            if Instant::now() > grace && !self.host.is_active(unit).await {
                return Err(format!("{unit} stopped before it was ready"));
            }
            if Instant::now() >= deadline {
                return Err(format!("not ready after {secs} s ({url})"));
            }
            sleep(Duration::from_secs(1)).await;
        }
    }

    fn fault(&self, reason: String) {
        audit("fault", json!({"reason": reason}));
        self.core().switch_fault(reason);
        self.commit();
    }

    async fn check_ready(&self, health: &mut Health) {
        let cfg = self.config();
        let ready = self.core().ready();
        health.failing_since.retain(|rt, _| ready.contains(rt));
        if ready.is_empty() || Instant::now() < health.next_check {
            return;
        }
        health.next_check = Instant::now() + Duration::from_secs(5);
        for rt in ready {
            let Some(r) = cfg.runtime(&rt).cloned() else {
                continue;
            };
            let active = self.host.is_active(&r.unit).await;
            if active && self.host.probe(&r.ready).await {
                health.failing_since.remove(&rt);
                continue;
            }
            let since = *health
                .failing_since
                .entry(rt.clone())
                .or_insert_with(Instant::now);
            if active && since.elapsed() < Duration::from_secs(cfg.sched.unhealthy_secs) {
                continue;
            }
            audit("crashed", json!({"runtime": rt, "unit_active": active}));
            health.failing_since.remove(&rt);
            if let Err(e) = self.stop_within(&r.unit, cfg.sched.stop_timeout_secs).await {
                return self.fault(format!("{rt} stopped answering and would not stop: {e}"));
            }
            self.core().crashed(self.now(), &rt);
            self.commit();
        }
    }

    /// One pass over the light services: start what should run, restart what
    /// stopped or stopped answering, fence what keeps failing.
    pub async fn supervise(&self) {
        let cfg = self.config();
        let now = self.now();
        for s in &cfg.services {
            let active = self.host.is_active(&s.unit).await;
            let healthy = active && (s.ready.is_empty() || self.host.probe(&s.ready).await);
            let action = {
                let mut all = self.services.lock().unwrap_or_else(|e| e.into_inner());
                let st = all.entry(s.name.clone()).or_default();
                if st.fenced_until.is_some_and(|t| t > now) {
                    st.state = "fenced";
                    None
                } else {
                    st.fenced_until = None;
                    if healthy {
                        st.state = "up";
                        st.unhealthy_since = None;
                        None
                    } else if s.restart == Restart::Never {
                        st.state = if active { "unhealthy" } else { "down" };
                        None
                    } else if active
                        && now.saturating_sub(*st.unhealthy_since.get_or_insert(now))
                            < cfg.sched.unhealthy_secs * 1000
                    {
                        st.state = "unhealthy";
                        None
                    } else if now < st.next_try {
                        st.state = "backoff";
                        None
                    } else {
                        Some(active)
                    }
                }
            };
            let Some(was_active) = action else { continue };
            if was_active {
                let _ = self.stop_within(&s.unit, cfg.sched.stop_timeout_secs).await;
            }
            let result = self.host.start(&s.unit).await;
            let mut all = self.services.lock().unwrap_or_else(|e| e.into_inner());
            let st = all.entry(s.name.clone()).or_default();
            st.unhealthy_since = None;
            // A start at boot is not a failure; a restart of something that
            // was meant to be running is.
            if st.state != "unknown" {
                st.restarts += 1;
                st.fails.push(now);
                st.fails
                    .retain(|&t| now.saturating_sub(t) < cfg.sched.fence_window_secs * 1000);
            }
            st.last_error = result.err();
            let n = st.fails.len() as u32;
            if n >= cfg.sched.fence_after {
                st.fenced_until = Some(now + cfg.sched.fence_secs * 1000);
                st.fails.clear();
                st.state = "fenced";
                audit(
                    "service-fenced",
                    json!({"service": s.name, "restarts": st.restarts}),
                );
            } else {
                st.next_try = now + (5_000u64 << n.min(6)).min(300_000);
                st.state = "starting";
                audit(
                    "service-start",
                    json!({"service": s.name, "restarts": st.restarts, "error": st.last_error}),
                );
            }
        }
        self.notify();
    }

    pub fn unfence_service(&self, name: &str) -> bool {
        let mut all = self.services.lock().unwrap_or_else(|e| e.into_inner());
        match all.get_mut(name) {
            Some(st) if st.fenced_until.is_some() => {
                st.fenced_until = None;
                st.next_try = 0;
                true
            }
            _ => false,
        }
    }

    /// The scheduler's main loop; never returns.
    pub async fn run(self: Arc<Self>) {
        self.reconcile().await;
        let mut health = Health::default();
        let mut next_supervise = Instant::now();
        let mut changes = self.subscribe();
        loop {
            self.step(&mut health).await;
            if Instant::now() >= next_supervise {
                self.supervise().await;
                next_supervise = Instant::now() + Duration::from_secs(5);
            }
            // Wake on any change (a request waiting, a hold) or every 500 ms.
            let _ = timeout(Duration::from_millis(500), changes.changed()).await;
        }
    }
}

type StopFut<'a> = std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + 'a>>;

/// Runs the futures together; their results in order.
async fn join_all(mut futs: Vec<StopFut<'_>>) -> Vec<Result<(), String>> {
    let mut out: Vec<Option<Result<(), String>>> = futs.iter().map(|_| None).collect();
    std::future::poll_fn(|cx| {
        let mut pending = false;
        for (f, o) in futs.iter_mut().zip(out.iter_mut()) {
            if o.is_none() {
                match f.as_mut().poll(cx) {
                    std::task::Poll::Ready(r) => *o = Some(r),
                    std::task::Poll::Pending => pending = true,
                }
            }
        }
        if pending {
            std::task::Poll::Pending
        } else {
            std::task::Poll::Ready(())
        }
    })
    .await;
    out.into_iter().map(|o| o.unwrap_or(Ok(()))).collect()
}

#[derive(Debug)]
pub struct Health {
    /// Since when each ready runtime has failed its check.
    failing_since: BTreeMap<String, Instant>,
    next_check: Instant,
}

impl Default for Health {
    fn default() -> Self {
        Health {
            failing_since: BTreeMap::new(),
            next_check: Instant::now(),
        }
    }
}

fn load_saved(path: &Path) -> Saved {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            audit(
                "state-unreadable",
                json!({"path": path, "error": e.to_string()}),
            );
            Saved::default()
        }),
        Err(_) => Saved::default(),
    }
}

/// Write to a temporary file, flush it to disk, rename over the old one and
/// flush the directory: a crash leaves either the old state or the new.
pub fn save_atomic(path: &Path, saved: &Saved) -> std::io::Result<()> {
    use std::io::Write;
    let dir = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)?;
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&serde_json::to_vec_pretty(saved).map_err(std::io::Error::other)?)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    std::fs::File::open(dir)?.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Class, EXAMPLE};
    use crate::core::Admit;
    use crate::host::fake::Fake;

    fn engine(f: impl FnOnce(&mut crate::host::fake::State)) -> Arc<Engine<Fake>> {
        let cfg = Arc::new(Config::parse(EXAMPLE).unwrap());
        let fake = Fake::with(|s| {
            s.url_stack
                .insert("http://127.0.0.1:8080".into(), "maya".into());
            s.url_stack.insert(
                "http://127.0.0.1:8188/system_stats".into(),
                "comfyui".into(),
            );
            s.url_stack
                .insert("http://127.0.0.1:9000/".into(), "dashboard".into());
            s.url_stack
                .insert("http://127.0.0.1:8200".into(), "mocap".into());
            f(s);
        });
        Engine::new(cfg, fake, None)
    }

    async fn settle(e: &Engine<Fake>, health: &mut Health, rounds: usize) {
        for _ in 0..rounds {
            e.step(health).await;
            sleep(Duration::from_millis(500)).await;
        }
    }

    #[tokio::test(start_paused = true)]
    async fn boot_stops_strays_and_brings_home_up() {
        let e = engine(|s| {
            s.active.insert("comfyui".into());
        });
        e.reconcile().await;
        assert!(
            e.host.active().is_empty(),
            "comfyui was not the GPUs' runtime"
        );
        let mut h = Health::default();
        settle(&e, &mut h, 3).await;
        assert!(e.core().is_ready("maya"));
        assert_eq!(e.host.log(), ["stop comfyui", "start maya"]);
    }

    #[tokio::test(start_paused = true)]
    async fn boot_adopts_the_home_runtime_already_up() {
        let e = engine(|s| {
            s.active.insert("maya".into());
        });
        e.reconcile().await;
        assert!(e.core().is_ready("maya"));
        assert!(e.host.log().is_empty(), "nothing restarted");
    }

    #[tokio::test(start_paused = true)]
    async fn switch_drains_stops_checks_vram_then_starts() {
        let e = engine(|_| {});
        let mut h = Health::default();
        settle(&e, &mut h, 2).await;
        let id = {
            let mut c = e.core();
            c.begin("maya");
            let Admit::Wait(id) = c.admit(e.now(), "comfyui", Class::Interactive) else {
                panic!()
            };
            id
        };
        // Dwell (180 s) then drain (60 s, maya never finishes): it still
        // switches, after the drain deadline.
        sleep(Duration::from_secs(181)).await;
        e.step(&mut h).await;
        assert!(e.core().is_ready("comfyui"));
        assert_eq!(e.host.log(), ["start maya", "stop maya", "start comfyui"]);
        assert_eq!(e.core().poll(e.now(), id), Some(Admit::Go));
    }

    /// A request for `rt` as the front door makes it: admitted, the switch
    /// runs, and it is answered.
    async fn ask(e: &Engine<Fake>, h: &mut Health, rt: &str) {
        let Admit::Wait(id) = e.core().admit(e.now(), rt, Class::Interactive) else {
            panic!("{rt} admitted at once")
        };
        e.step(h).await;
        assert_eq!(e.core().poll(e.now(), id), Some(Admit::Go), "{}", e.status());
    }

    #[tokio::test(start_paused = true)]
    async fn one_card_runtimes_run_side_by_side_on_their_own_cards() {
        let e = engine(|_| {});
        let mut h = Health::default();
        settle(&e, &mut h, 2).await;
        ask(&e, &mut h, "comfyui").await;
        ask(&e, &mut h, "mocap").await;
        assert!(e.core().is_ready("comfyui") && e.core().is_ready("mocap"));
        assert_eq!(
            e.host.log(),
            ["start maya", "stop maya", "start comfyui", "start mocap"]
        );
        let cards = e.host.0.lock().unwrap().cards.clone();
        assert_eq!((cards["comfyui"].clone(), cards["mocap"].clone()), (vec![0], vec![1]));
        // Both stay healthy side by side.
        settle(&e, &mut h, 20).await;
        assert!(e.core().is_ready("comfyui") && e.core().is_ready("mocap"));
        // One crashes: only it restarts; the other keeps its card.
        e.host.0.lock().unwrap().active.remove("mocap");
        settle(&e, &mut h, 30).await;
        assert!(e.core().is_ready("comfyui"));
        assert!(e.core().is_ready("mocap"));
        assert!(!e.host.log().contains(&"stop comfyui".to_owned()));
        // maya held: both go, then maya gets both cards.
        let now = e.now();
        e.core()
            .set_hold(now, "maya", crate::core::HoldFor::Until(None), "t", false)
            .unwrap();
        e.step(&mut h).await;
        assert!(e.core().is_ready("maya"));
        assert!(e.host.active().len() == 1);
        // A restart of the scheduler keeps one-card runtimes on their cards.
        let now = e.now();
        e.core().release(now, Some("maya"), "x", false).unwrap();
        ask(&e, &mut h, "mocap").await;
        let st = e.status();
        assert_eq!(e.core().slot("mocap").map(|s| s.cards.clone()), Some(vec![0]), "{st}");
        let saved = e.core().saved();
        let path = std::env::temp_dir().join(format!("kiri-sched-adopt-{}", std::process::id()));
        save_atomic(&path, &saved).unwrap();
        let cfg = e.config();
        let fake = std::mem::take(&mut *e.host.0.lock().unwrap());
        let e2 = Engine::new(cfg, Fake(std::sync::Mutex::new(fake)), Some(path.clone()));
        e2.reconcile().await;
        std::fs::remove_file(&path).unwrap();
        assert!(e2.core().is_ready("mocap"));
        assert_eq!(e2.core().slot("mocap").unwrap().cards, [0]);
    }

    #[tokio::test(start_paused = true)]
    async fn evicted_runtimes_stop_together() {
        let e = engine(|_| {});
        let mut h = Health::default();
        settle(&e, &mut h, 2).await;
        ask(&e, &mut h, "comfyui").await;
        ask(&e, &mut h, "mocap").await;
        e.host.0.lock().unwrap().stop_secs = 30;
        let now = e.now();
        e.core()
            .set_hold(now, "maya", crate::core::HoldFor::Until(None), "t", false)
            .unwrap();
        let t = Instant::now();
        e.step(&mut h).await;
        assert!(e.core().is_ready("maya"));
        let took = t.elapsed();
        assert!(took < Duration::from_secs(45), "stopped one after another: {took:?}");
    }

    #[tokio::test(start_paused = true)]
    async fn leftover_memory_on_another_card_does_not_block_a_one_card_start() {
        let e = engine(|_| {});
        let mut h = Health::default();
        settle(&e, &mut h, 2).await;
        ask(&e, &mut h, "comfyui").await;
        // mocap's card 1 is clean; card 0 is comfyui's and in use.
        ask(&e, &mut h, "mocap").await;
        assert!(e.core().fault().is_none());
        assert!(e.core().is_ready("mocap"));
    }

    #[tokio::test(start_paused = true)]
    async fn failed_start_rolls_back_to_last_good() {
        let e = engine(|s| {
            s.never_ready.insert("comfyui".into());
        });
        let mut h = Health::default();
        settle(&e, &mut h, 2).await;
        e.core().admit(e.now(), "comfyui", Class::Interactive);
        e.step(&mut h).await;
        // comfyui never answered: stopped again, and maya comes back.
        assert!(!e.host.active().contains("comfyui"));
        e.step(&mut h).await;
        assert!(e.core().is_ready("maya"));
        assert_eq!(
            e.host.log(),
            [
                "start maya",
                "stop maya",
                "start comfyui",
                "stop comfyui",
                "start maya"
            ]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_runtime_that_dies_while_loading_fails_fast() {
        let e = engine(|s| {
            s.exits_on_start.insert("comfyui".into());
        });
        let mut h = Health::default();
        settle(&e, &mut h, 2).await;
        e.core().admit(e.now(), "comfyui", Class::Interactive);
        let t = Instant::now();
        e.step(&mut h).await;
        // Its unit ended: given up after the short grace, not start_timeout_secs.
        assert!(t.elapsed() < Duration::from_secs(30), "{:?}", t.elapsed());
        e.step(&mut h).await;
        assert!(e.core().is_ready("maya"));
    }

    #[tokio::test(start_paused = true)]
    async fn leftover_gpu_memory_is_a_fault() {
        let e = engine(|s| s.leak_mib = 9000);
        let mut h = Health::default();
        e.step(&mut h).await;
        assert!(e.core().fault().is_some());
        assert!(e.host.log().is_empty(), "nothing may start");
    }

    #[tokio::test(start_paused = true)]
    async fn stop_that_fails_is_a_fault() {
        let e = engine(|s| {
            s.stop_fails.insert("maya".into());
        });
        let mut h = Health::default();
        settle(&e, &mut h, 2).await;
        e.core().admit(e.now(), "comfyui", Class::Interactive);
        e.step(&mut h).await;
        assert!(e.core().fault().is_some());
        assert!(!e.host.active().contains("comfyui"));
    }

    #[tokio::test(start_paused = true)]
    async fn a_crash_restarts_with_backoff() {
        let e = engine(|_| {});
        let mut h = Health::default();
        settle(&e, &mut h, 2).await;
        e.host.0.lock().unwrap().active.remove("maya");
        settle(&e, &mut h, 30).await;
        assert!(e.core().is_ready("maya"), "{}", e.core().phase());
        assert_eq!(e.host.log(), ["start maya", "stop maya", "start maya"]);
    }

    #[tokio::test(start_paused = true)]
    async fn services_start_restart_and_fence() {
        let e = engine(|s| {
            s.ready_urls.insert("http://127.0.0.1:9000/".into(), false);
        });
        // Boot: started, not counted as a restart.
        e.supervise().await;
        assert_eq!(e.services.lock().unwrap()["dashboard"].restarts, 0);
        // It never answers: restarted after 60 s unhealthy, then fenced.
        for _ in 0..200 {
            sleep(Duration::from_secs(5)).await;
            e.supervise().await;
        }
        let st = e.services.lock().unwrap()["dashboard"].clone();
        assert_eq!(st.state, "fenced");
        assert_eq!(st.restarts, 3);
        assert!(e.unfence_service("dashboard"));
    }

    #[test]
    fn state_saves_atomically() {
        let dir = std::env::temp_dir().join(format!("kiri-sched-state-{}", std::process::id()));
        let path = dir.join("state.json");
        let saved = Saved {
            last_good: Some("maya".into()),
            ..Saved::default()
        };
        save_atomic(&path, &saved).unwrap();
        assert_eq!(load_saved(&path), saved);
        std::fs::write(&path, "{not json").unwrap();
        assert_eq!(load_saved(&path), Saved::default());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
