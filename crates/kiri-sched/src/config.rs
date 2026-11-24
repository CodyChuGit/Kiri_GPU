//! `sched.toml`: the runtimes, light services and clients the scheduler
//! knows. The whole file is checked before any of it is used; a file that
//! fails a check is refused as a whole, so a bad edit never half-applies.

use std::collections::BTreeSet;
use std::path::PathBuf;

use serde::Deserialize;

/// Who sends a request, which decides how long it may wait and whether it
/// may switch the box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Class {
    /// You, now: may switch the box after the dwell time.
    Interactive,
    /// Agents and scripts: wait longer, switch only an idle box.
    Agent,
    /// Batch submitters: never switch the box from a request.
    Batch,
}

impl Class {
    pub fn parse(s: &str) -> Option<Class> {
        match s {
            "interactive" => Some(Class::Interactive),
            "agent" => Some(Class::Agent),
            "batch" => Some(Class::Batch),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Class::Interactive => "interactive",
            Class::Agent => "agent",
            Class::Batch => "batch",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub sched: Sched,
    #[serde(default, rename = "client")]
    pub clients: Vec<Client>,
    #[serde(default, rename = "runtime")]
    pub runtimes: Vec<Runtime>,
    #[serde(default, rename = "service")]
    pub services: Vec<Service>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sched {
    /// The front door, normally only reachable through Caddy.
    #[serde(default = "d_listen")]
    pub listen: String,
    #[serde(default = "d_admin")]
    pub admin_socket: PathBuf,
    #[serde(default = "d_lease")]
    pub lease_socket: PathBuf,
    #[serde(default = "d_state")]
    pub state_file: PathBuf,
    /// The runtime the GPUs go back to when nothing else wants them.
    pub home: String,
    /// GPUs on the box, numbered from 0 in `nvidia-smi` order (the CDI
    /// device names `nvidia.com/gpu=0`, `=1`, …).
    #[serde(default = "d_gpus")]
    pub gpus: usize,
    /// Where the scheduler tells a one-card stack which card it got:
    /// `<dir>/<stack>.env` with `KIRI_SCHED_GPUS=<index>`, read by
    /// `kiri-run@<stack>.service`.
    #[serde(default = "d_cards_dir")]
    pub cards_dir: PathBuf,
    /// Minimum time a runtime keeps the GPUs once ready, before interactive
    /// work for another runtime may drain it.
    #[serde(default = "d_dwell")]
    pub dwell_secs: u64,
    /// How long in-flight requests get to finish before a switch stops them.
    #[serde(default = "d_drain")]
    pub drain_secs: u64,
    /// How long a stack gets to stop.
    #[serde(default = "d_stop")]
    pub stop_timeout_secs: u64,
    /// A GPU counts as empty at or under this much used memory.
    #[serde(default = "d_vram_free")]
    pub vram_free_mib: u64,
    /// How long the GPUs get to become empty after a stop.
    #[serde(default = "d_vram_wait")]
    pub vram_wait_secs: u64,
    /// Agent work waiting this long counts as interactive.
    #[serde(default = "d_aging")]
    pub agent_aging_secs: u64,
    /// How long a request is held open waiting for its runtime, per class.
    #[serde(default = "d_wait_interactive")]
    pub interactive_wait_secs: u64,
    #[serde(default = "d_wait_agent")]
    pub agent_wait_secs: u64,
    #[serde(default = "d_wait_batch")]
    pub batch_wait_secs: u64,
    /// Requests waiting for a runtime, over all runtimes; more get 429.
    #[serde(default = "d_max_waiting")]
    pub max_waiting: usize,
    /// Largest request body passed through, in MiB.
    #[serde(default = "d_max_body")]
    pub max_body_mib: u64,
    /// Failed starts within `fence_window_secs` that fence a runtime or service.
    #[serde(default = "d_fence_after")]
    pub fence_after: u32,
    #[serde(default = "d_fence_window")]
    pub fence_window_secs: u64,
    /// How long a fence lasts unless cleared by hand.
    #[serde(default = "d_fence_for")]
    pub fence_secs: u64,
    /// A ready runtime that fails its check this long is restarted.
    #[serde(default = "d_unhealthy")]
    pub unhealthy_secs: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Client {
    pub name: String,
    /// SHA-256 of the client's bearer token, in hex (`kiri-sched hash-token`).
    pub token_sha256: String,
    pub class: Class,
    /// May switch the box, hold it, and clear fences through the front door.
    #[serde(default)]
    pub admin: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Runtime {
    pub name: String,
    /// The Kiri stack (its Compose project name). Defaults to the name.
    #[serde(default)]
    pub stack: String,
    /// The systemd unit that runs the stack and whose stop really stops
    /// it. Defaults to `kiri-run@<stack>.service`.
    #[serde(default)]
    pub unit: String,
    /// Where its API listens, `http://127.0.0.1:8080`.
    pub upstream: String,
    /// Readiness URL; a 2xx or 3xx answer means ready. Defaults to the upstream.
    #[serde(default)]
    pub ready: String,
    /// Model names its API serves, as clients send them in `"model"`.
    #[serde(default)]
    pub models: Vec<String>,
    /// Requests it takes at once; more wait.
    #[serde(default = "d_concurrency")]
    pub concurrency: usize,
    #[serde(default = "d_start")]
    pub start_timeout_secs: u64,
    /// Give the GPUs back to the home runtime after this long with no
    /// request in flight or waiting and no hold (for runtimes that block
    /// everything while loaded, such as a render server). Unset: it stays
    /// until something else is asked for.
    #[serde(default)]
    pub home_after_idle_secs: Option<u64>,
    /// How long its stack gets to stop; unset means `[sched]
    /// stop_timeout_secs`. Keep it under kiri-run@'s TimeoutStopSec (300 s).
    #[serde(default)]
    pub stop_timeout_secs: u64,
    /// How many GPUs it needs; unset means all of them. A runtime that needs
    /// fewer shares the box with others that fit beside it.
    #[serde(default)]
    pub cards: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Service {
    pub name: String,
    #[serde(default)]
    pub stack: String,
    #[serde(default)]
    pub unit: String,
    /// Readiness URL; needed unless `restart = "never"`, since a unit that
    /// is active says nothing about the container behind it.
    #[serde(default)]
    pub ready: String,
    #[serde(default)]
    pub restart: Restart,
}

/// Whether the scheduler keeps a light service running.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Restart {
    /// Started at boot, restarted when it stops or stops answering.
    #[default]
    Always,
    /// Watched and reported only.
    Never,
}

fn d_listen() -> String {
    "127.0.0.1:7070".into()
}
fn d_admin() -> PathBuf {
    "/run/kiri-sched/admin.sock".into()
}
fn d_lease() -> PathBuf {
    "/run/kiri-sched/lease.sock".into()
}
fn d_state() -> PathBuf {
    "/var/lib/kiri-sched/state.json".into()
}
fn d_gpus() -> usize {
    2
}
fn d_cards_dir() -> PathBuf {
    "/run/kiri-sched/cards".into()
}
fn d_dwell() -> u64 {
    180
}
fn d_drain() -> u64 {
    60
}
fn d_stop() -> u64 {
    90
}
fn d_vram_free() -> u64 {
    500
}
fn d_vram_wait() -> u64 {
    60
}
fn d_aging() -> u64 {
    600
}
fn d_wait_interactive() -> u64 {
    120
}
fn d_wait_agent() -> u64 {
    600
}
fn d_wait_batch() -> u64 {
    30
}
fn d_max_waiting() -> usize {
    256
}
fn d_max_body() -> u64 {
    32
}
fn d_fence_after() -> u32 {
    3
}
fn d_fence_window() -> u64 {
    600
}
fn d_fence_for() -> u64 {
    1800
}
fn d_unhealthy() -> u64 {
    60
}
fn d_concurrency() -> usize {
    1
}
fn d_start() -> u64 {
    600
}

impl Config {
    /// Reads and checks a config; the error names what is wrong.
    pub fn parse(text: &str) -> Result<Config, String> {
        let mut c: Config = toml::from_str(text).map_err(|e| e.to_string())?;
        for r in &mut c.runtimes {
            if r.stack.is_empty() {
                r.stack = r.name.clone();
            }
            if r.ready.is_empty() {
                r.ready = r.upstream.clone();
            }
            if r.unit.is_empty() {
                r.unit = format!("kiri-run@{}.service", r.stack);
            }
            if r.cards == 0 {
                r.cards = c.sched.gpus;
            }
            if r.stop_timeout_secs == 0 {
                r.stop_timeout_secs = c.sched.stop_timeout_secs;
            }
        }
        for s in &mut c.services {
            if s.stack.is_empty() {
                s.stack = s.name.clone();
            }
            if s.unit.is_empty() {
                s.unit = format!("kiri-run@{}.service", s.stack);
            }
        }
        c.check()?;
        Ok(c)
    }

    fn check(&self) -> Result<(), String> {
        let mut names = BTreeSet::new();
        let mut stacks = BTreeSet::new();
        for (name, stack) in self
            .runtimes
            .iter()
            .map(|r| (&r.name, &r.stack))
            .chain(self.services.iter().map(|s| (&s.name, &s.stack)))
        {
            check_name(name)?;
            check_name(stack)?;
            if !names.insert(name.as_str()) {
                return Err(format!("{name}: named twice"));
            }
            if !stacks.insert(stack.as_str()) {
                return Err(format!("stack {stack}: used twice"));
            }
        }
        if self.runtimes.is_empty() {
            return Err("no [[runtime]]".into());
        }
        if self.runtime(&self.sched.home).is_none() {
            return Err(format!("home {:?} is not a [[runtime]]", self.sched.home));
        }
        let mut models = BTreeSet::new();
        for r in &self.runtimes {
            parse_http_url(&r.upstream).map_err(|e| format!("{}: upstream: {e}", r.name))?;
            parse_http_url(&r.ready).map_err(|e| format!("{}: ready: {e}", r.name))?;
            if r.concurrency == 0 || r.concurrency > 1024 {
                return Err(format!("{}: concurrency must be 1-1024", r.name));
            }
            if r.cards == 0 || r.cards > self.sched.gpus {
                return Err(format!(
                    "{}: cards must be 1-{} (the box's gpus)",
                    r.name, self.sched.gpus
                ));
            }
            if r.start_timeout_secs == 0 {
                return Err(format!("{}: start_timeout_secs must be over 0", r.name));
            }
            for m in &r.models {
                if m.is_empty() || !models.insert(m.as_str()) {
                    return Err(format!("model {m:?}: empty or served by two runtimes"));
                }
            }
        }
        let mut units = BTreeSet::new();
        for u in self
            .runtimes
            .iter()
            .map(|r| &r.unit)
            .chain(self.services.iter().map(|s| &s.unit))
        {
            let ok = u.ends_with(".service")
                && u.len() <= 128
                && u.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"@-_.:".contains(&b));
            if !ok || !units.insert(u.as_str()) {
                return Err(format!("unit {u:?}: must be a .service name, used once"));
            }
        }
        for s in &self.services {
            if s.ready.is_empty() && s.restart != Restart::Never {
                return Err(format!(
                    "{}: a service the scheduler restarts needs a ready URL",
                    s.name
                ));
            }
            if !s.ready.is_empty() {
                parse_http_url(&s.ready).map_err(|e| format!("{}: ready: {e}", s.name))?;
            }
        }
        let mut hashes = BTreeSet::new();
        for c in &self.clients {
            if c.name.is_empty() {
                return Err("a [[client]] has no name".into());
            }
            if c.token_sha256.len() != 64 || !c.token_sha256.bytes().all(|b| b.is_ascii_hexdigit())
            {
                return Err(format!(
                    "client {}: token_sha256 must be 64 hex digits",
                    c.name
                ));
            }
            if !hashes.insert(c.token_sha256.to_ascii_lowercase()) {
                return Err(format!("client {}: token used twice", c.name));
            }
        }
        let s = &self.sched;
        if s.gpus == 0 || s.gpus > 16 {
            return Err("gpus must be 1-16".into());
        }
        if s.fence_after == 0 {
            return Err("fence_after must be over 0".into());
        }
        if s.max_body_mib == 0 || s.max_body_mib > 1024 {
            return Err("max_body_mib must be 1-1024".into());
        }
        s.listen
            .parse::<std::net::SocketAddr>()
            .map_err(|e| format!("listen: {e}"))?;
        Ok(())
    }

    pub fn runtime(&self, name: &str) -> Option<&Runtime> {
        self.runtimes.iter().find(|r| r.name == name)
    }

    pub fn runtime_for_model(&self, model: &str) -> Option<&Runtime> {
        self.runtimes
            .iter()
            .find(|r| r.models.iter().any(|m| m == model))
    }

    pub fn runtime_for_stack(&self, stack: &str) -> Option<&Runtime> {
        self.runtimes.iter().find(|r| r.stack == stack)
    }

    /// GPUs a runtime needs (all of them for a name not in the config).
    pub fn cards(&self, rt: &str) -> usize {
        self.runtime(rt).map_or(self.sched.gpus, |r| r.cards)
    }

    pub fn wait_secs(&self, class: Class) -> u64 {
        match class {
            Class::Interactive => self.sched.interactive_wait_secs,
            Class::Agent => self.sched.agent_wait_secs,
            Class::Batch => self.sched.batch_wait_secs,
        }
    }
}

/// Stack and runtime names end up in unit names and paths.
fn check_name(n: &str) -> Result<(), String> {
    let ok = !n.is_empty()
        && n.len() <= 64
        && n.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
        && !n.starts_with('-');
    if ok {
        Ok(())
    } else {
        Err(format!("{n:?}: names are 1-64 of a-z, 0-9, - and _"))
    }
}

/// An upstream: plain HTTP to a host and port (the runtimes listen locally;
/// TLS is Caddy's job). Returns host, port and path prefix.
pub fn parse_http_url(url: &str) -> Result<(String, u16, String), String> {
    let rest = url
        .strip_prefix("http://")
        .ok_or("must start with http://")?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (h, p.parse::<u16>().map_err(|_| "bad port")?),
        None => (authority, 80),
    };
    if host.is_empty() || port == 0 {
        return Err("needs a host and port".into());
    }
    Ok((host.to_owned(), port, path.to_owned()))
}

#[cfg(test)]
pub const EXAMPLE: &str = r#"
[sched]
home = "maya"
listen = "127.0.0.1:0"

[[client]]
name = "cody"
token_sha256 = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
class = "interactive"
admin = true

[[client]]
name = "agents"
token_sha256 = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
class = "agent"

[[runtime]]
name = "maya"
upstream = "http://127.0.0.1:8080"
models = ["glm-5.3-flash"]
concurrency = 4

[[runtime]]
name = "comfyui"
upstream = "http://127.0.0.1:8188"
ready = "http://127.0.0.1:8188/system_stats"
home_after_idle_secs = 600
cards = 1

[[runtime]]
name = "mocap"
upstream = "http://127.0.0.1:8200"
cards = 1

[[service]]
name = "dashboard"
ready = "http://127.0.0.1:9000/"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_parses_with_defaults() {
        let c = Config::parse(EXAMPLE).unwrap();
        assert_eq!(c.runtime("maya").unwrap().stack, "maya");
        assert_eq!(c.runtime("maya").unwrap().unit, "kiri-run@maya.service");
        assert_eq!(c.runtime("maya").unwrap().ready, "http://127.0.0.1:8080");
        assert_eq!(c.runtime_for_model("glm-5.3-flash").unwrap().name, "maya");
        assert_eq!(c.sched.dwell_secs, 180);
        assert_eq!(c.services[0].restart, Restart::Always);
        assert_eq!(c.sched.gpus, 2);
        assert_eq!(c.cards("maya"), 2);
        assert_eq!(c.cards("comfyui"), 1);
        assert_eq!(c.runtime("maya").unwrap().stop_timeout_secs, 90);
    }

    #[test]
    fn refuses_bad_files_whole() {
        let bad = [
            EXAMPLE.replace("home = \"maya\"", "home = \"nope\""),
            EXAMPLE.replace("name = \"comfyui\"", "name = \"maya\""),
            EXAMPLE.replace("http://127.0.0.1:8188\"\nready", "https://x:1\"\nready"),
            EXAMPLE.replace("concurrency = 4", "concurrency = 0"),
            EXAMPLE.replace("cards = 1\n\n[[runtime]]", "cards = 3\n\n[[runtime]]"),
            EXAMPLE.replace("listen = \"127.0.0.1:0\"", "listen = \"127.0.0.1:0\"\ngpus = 0"),
            EXAMPLE.replace("class = \"agent\"", "class = \"root\""),
            EXAMPLE.replace("2cf24dba5fb0a30e", "2cf24dba5fb0a30"),
            EXAMPLE.replace("name = \"dashboard\"", "name = \"Dash board\""),
            EXAMPLE.replace("[[service]]", "[[service]]\nshell = \"rm -rf /\""),
            EXAMPLE.replace("ready = \"http://127.0.0.1:9000/\"", ""),
            EXAMPLE.replace("name = \"comfyui\"", "name = \"comfyui\"\nunit = \"x; rm\""),
            EXAMPLE.replace(
                "models = [\"glm-5.3-flash\"]",
                "models = [\"glm-5.3-flash\", \"glm-5.3-flash\"]",
            ),
        ];
        for b in &bad {
            assert!(Config::parse(b).is_err(), "accepted:\n{b}");
        }
    }

    #[test]
    fn shipped_example_parses() {
        let text = include_str!("../contrib/sched.example.toml");
        let text = text
            .replacen("<64 hex digits>", &"a".repeat(64), 1)
            .replacen("<64 hex digits>", &"b".repeat(64), 1);
        let c = Config::parse(&text).unwrap();
        assert_eq!(
            c.runtime_for_model("glm-5.3-flash").unwrap().name,
            "maya-glm"
        );
    }

    #[test]
    fn urls() {
        assert_eq!(
            parse_http_url("http://127.0.0.1:8080").unwrap(),
            ("127.0.0.1".into(), 8080, "/".into())
        );
        assert_eq!(
            parse_http_url("http://localhost:1/health").unwrap().2,
            "/health"
        );
        assert!(parse_http_url("http://:80/").is_err());
        assert!(parse_http_url("ftp://a:1").is_err());
    }
}
