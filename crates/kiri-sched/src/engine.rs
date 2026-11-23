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
use crate::core::{Core, Phase, Saved};
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

    /// At startup: keep the runtime that is already up if it is the one the
    /// GPUs should go to anyway, and stop every other GPU runtime, so the
    /// box never runs two after a crash or a reboot.
    pub async fn reconcile(&self) {
        let cfg = self.config();
        let mut active = Vec::new();
        for r in &cfg.runtimes {
            if self.host.is_active(&r.unit).await {
                active.push(r.clone());
            }
        }
        let wanted = self.core().want_switch(self.now());
        let keep = match &wanted {
            Some(w) => match active.iter().find(|r| &r.name == w) {
                Some(r) if self.host.probe(&r.ready).await => Some(r.name.clone()),
                _ => None,
            },
            None => None,
        };
        for r in active.iter().filter(|r| Some(&r.name) != keep.as_ref()) {
            audit(
                "reconcile-stop",
                json!({"runtime": r.name, "why": "running at startup but not the runtime the GPUs go to"}),
            );
            if let Err(e) = self.stop_within(&r.unit, cfg.sched.stop_timeout_secs).await {
                self.core()
                    .switch_fault(format!("could not stop {} at startup: {e}", r.name));
                audit("fault", json!({"reason": e}));
                self.commit();
                return;
            }
        }
        if let Some(rt) = keep {
            audit("adopt", json!({"runtime": rt}));
            self.core().adopt(self.now(), &rt);
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
    /// the runtime on the GPUs.
    pub async fn step(&self, health: &mut Health) {
        let to = {
            let mut core = self.core();
            let to = core.want_switch(self.now());
            if let Some(to) = &to {
                core.switch_begin(to);
            }
            to
        };
        match to {
            Some(to) => {
                health.failing_since = None;
                self.notify();
                self.switch(&to).await;
            }
            None => self.check_ready(health).await,
        }
    }

    /// The switch sequence; `core.switch_begin` was called.
    async fn switch(&self, to: &str) {
        let cfg = self.config();
        let started = Instant::now();
        let from = match self.core().phase().clone() {
            Phase::Draining { from, .. } => Some(from),
            Phase::Stopping { from, .. } => from,
            _ => None,
        };
        audit("switch-begin", json!({"from": from, "to": to}));
        // 1. Drain: in-flight requests get `drain_secs` to finish.
        if let Some(f) = &from {
            let deadline = Instant::now() + Duration::from_secs(cfg.sched.drain_secs);
            while self.core().inflight(f) > 0 && Instant::now() < deadline {
                sleep(Duration::from_millis(200)).await;
            }
            let left = self.core().inflight(f);
            if left > 0 {
                audit("drain-timeout", json!({"runtime": f, "inflight": left}));
            }
        }
        self.core().switch_stopping();
        self.notify();
        // 2. Stop every GPU runtime but the target (normally just `from`).
        for r in &cfg.runtimes {
            let running = Some(&r.name) == from.as_ref() || self.host.is_active(&r.unit).await;
            if !running {
                continue;
            }
            if let Err(e) = self.stop_within(&r.unit, cfg.sched.stop_timeout_secs).await {
                return self.fault(format!("could not stop {}: {e}", r.name));
            }
        }
        // 3. The GPUs must be seen empty before anything else gets them.
        let deadline = Instant::now() + Duration::from_secs(cfg.sched.vram_wait_secs);
        loop {
            match self.host.vram_used().await {
                Ok(used) if used.iter().all(|&m| m <= cfg.sched.vram_free_mib) => break,
                Ok(used) if Instant::now() >= deadline => {
                    return self.fault(format!("GPU memory still in use after stopping everything: {used:?} MiB (limit {} MiB)", cfg.sched.vram_free_mib));
                }
                Err(e) if Instant::now() >= deadline => {
                    return self.fault(format!("cannot read GPU memory: {e}"));
                }
                _ => sleep(Duration::from_millis(500)).await,
            }
        }
        // 4. Start the target and wait for it to answer.
        let Some(rt) = cfg.runtime(to).cloned() else {
            return self.fault(format!("runtime {to} vanished from the config"));
        };
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
                audit("switch-ready", json!({"runtime": to, "secs": took / 1000}));
                self.commit();
            }
            Err(e) => {
                audit("start-failed", json!({"runtime": to, "error": e}));
                // Roll back: the failed runtime is stopped again so the GPUs
                // are empty for whatever comes next (the last good one).
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
        let Phase::Ready { rt, .. } = self.core().phase().clone() else {
            health.failing_since = None;
            return;
        };
        if Instant::now() < health.next_check {
            return;
        }
        health.next_check = Instant::now() + Duration::from_secs(5);
        let Some(r) = cfg.runtime(&rt).cloned() else {
            return;
        };
        let active = self.host.is_active(&r.unit).await;
        if active && self.host.probe(&r.ready).await {
            health.failing_since = None;
            return;
        }
        let since = *health.failing_since.get_or_insert_with(Instant::now);
        if active && since.elapsed() < Duration::from_secs(cfg.sched.unhealthy_secs) {
            return;
        }
        audit("crashed", json!({"runtime": rt, "unit_active": active}));
        health.failing_since = None;
        if let Err(e) = self.stop_within(&r.unit, cfg.sched.stop_timeout_secs).await {
            return self.fault(format!("{rt} stopped answering and would not stop: {e}"));
        }
        self.core().crashed(self.now(), &rt);
        self.commit();
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

#[derive(Debug)]
pub struct Health {
    failing_since: Option<Instant>,
    next_check: Instant,
}

impl Default for Health {
    fn default() -> Self {
        Health {
            failing_since: None,
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
    async fn leftover_gpu_memory_is_a_fault() {
        let e = engine(|s| s.leak_mib = 9000);
        let mut h = Health::default();
        e.step(&mut h).await;
        assert!(matches!(e.core().phase(), Phase::Fault { .. }));
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
        assert!(matches!(e.core().phase(), Phase::Fault { .. }));
        assert!(!e.host.active().contains("comfyui"));
    }

    #[tokio::test(start_paused = true)]
    async fn a_crash_restarts_with_backoff() {
        let e = engine(|_| {});
        let mut h = Health::default();
        settle(&e, &mut h, 2).await;
        e.host.0.lock().unwrap().active.remove("maya");
        settle(&e, &mut h, 30).await;
        assert!(e.core().is_ready("maya"), "{:?}", e.core().phase());
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
