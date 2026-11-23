//! What the scheduler does to the machine, behind a trait so the switch
//! and supervisor logic run against a fake machine in tests.

use std::future::Future;
use std::time::Duration;

use bytes::Bytes;
use http_body_util::Empty;
use tokio::process::Command;

use crate::config::parse_http_url;

pub trait Host: Send + Sync + 'static {
    /// Starts a unit without waiting for it to be ready.
    fn start(&self, unit: &str) -> impl Future<Output = Result<(), String>> + Send;
    /// Stops it and returns once the unit is inactive.
    fn stop(&self, unit: &str) -> impl Future<Output = Result<(), String>> + Send;
    fn is_active(&self, unit: &str) -> impl Future<Output = bool> + Send;
    /// Used memory of every GPU, MiB.
    fn vram_used(&self) -> impl Future<Output = Result<Vec<u64>, String>> + Send;
    /// Whether a readiness URL answers 2xx or 3xx.
    fn probe(&self, url: &str) -> impl Future<Output = bool> + Send;
}

/// The real machine: systemd units (`kiri-run@<stack>.service` by
/// default) and `nvidia-smi`.
#[derive(Debug, Default, Clone)]
pub struct System;

async fn systemctl(args: &[&str]) -> Result<(), String> {
    let out = Command::new("systemctl")
        .args(args)
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|e| format!("systemctl: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "systemctl {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    }
}

impl Host for System {
    async fn start(&self, unit: &str) -> Result<(), String> {
        systemctl(&["start", "--no-block", unit]).await
    }

    async fn stop(&self, unit: &str) -> Result<(), String> {
        systemctl(&["stop", unit]).await?;
        if self.is_active(unit).await {
            Err(format!("{unit} is still active after stop"))
        } else {
            Ok(())
        }
    }

    async fn is_active(&self, unit: &str) -> bool {
        // Activating and reloading count as active: the GPUs may be in use.
        match Command::new("systemctl")
            .args(["is-active", unit])
            .kill_on_drop(true)
            .output()
            .await
        {
            Ok(out) => !matches!(
                String::from_utf8_lossy(&out.stdout).trim(),
                "inactive" | "failed" | "unknown" | ""
            ),
            // Not knowing is treated as running: never assume the GPUs free.
            Err(_) => true,
        }
    }

    async fn vram_used(&self) -> Result<Vec<u64>, String> {
        let out = Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            .kill_on_drop(true)
            .output()
            .await
            .map_err(|e| format!("nvidia-smi: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "nvidia-smi: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        parse_vram(&String::from_utf8_lossy(&out.stdout))
    }

    async fn probe(&self, url: &str) -> bool {
        tokio::time::timeout(Duration::from_secs(3), get_status(url))
            .await
            .ok()
            .flatten()
            .is_some_and(|s| (200..400).contains(&s))
    }
}

pub fn parse_vram(text: &str) -> Result<Vec<u64>, String> {
    let v: Result<Vec<u64>, _> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::parse)
        .collect();
    match v {
        Ok(v) if !v.is_empty() => Ok(v),
        _ => Err(format!(
            "nvidia-smi gave no memory figures: {:?}",
            text.trim()
        )),
    }
}

/// One GET over a fresh HTTP/1.1 connection; the status, if any.
async fn get_status(url: &str) -> Option<u16> {
    let (host, port, path) = parse_http_url(url).ok()?;
    let stream = tokio::net::TcpStream::connect((host.as_str(), port))
        .await
        .ok()?;
    let (mut send, conn) =
        hyper::client::conn::http1::handshake(hyper_util::rt::TokioIo::new(stream))
            .await
            .ok()?;
    tokio::spawn(conn);
    let req = hyper::Request::get(path)
        .header(hyper::header::HOST, format!("{host}:{port}"))
        .body(Empty::<Bytes>::new())
        .ok()?;
    Some(send.send_request(req).await.ok()?.status().as_u16())
}

#[cfg(test)]
pub mod fake {
    //! A machine in memory: stacks that start and stop, GPUs whose memory
    //! follows what runs, and failures on demand.
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Mutex;

    use super::Host;

    #[derive(Debug, Default)]
    pub struct State {
        pub active: BTreeSet<String>,
        /// Stacks that start but never become ready.
        pub never_ready: BTreeSet<String>,
        /// Stacks whose stop fails.
        pub stop_fails: BTreeSet<String>,
        /// Memory left on the GPUs after a stop, MiB.
        pub leak_mib: u64,
        /// Every call, in order, for the tests to read.
        pub log: Vec<String>,
        /// Readiness by URL, for services checked without a stack.
        pub ready_urls: BTreeMap<String, bool>,
        /// Stack per readiness URL.
        pub url_stack: BTreeMap<String, String>,
    }

    #[derive(Debug, Default)]
    pub struct Fake(pub Mutex<State>);

    impl Fake {
        pub fn with(f: impl FnOnce(&mut State)) -> Fake {
            let fake = Fake::default();
            f(&mut fake.0.lock().unwrap());
            fake
        }
        pub fn log(&self) -> Vec<String> {
            self.0.lock().unwrap().log.clone()
        }
        pub fn active(&self) -> BTreeSet<String> {
            self.0.lock().unwrap().active.clone()
        }
    }

    /// `kiri-run@maya.service` is `maya` here.
    fn short(unit: &str) -> &str {
        unit.trim_start_matches("kiri-run@")
            .trim_end_matches(".service")
    }

    impl Host for Fake {
        async fn start(&self, unit: &str) -> Result<(), String> {
            let stack = short(unit);
            let mut s = self.0.lock().unwrap();
            s.log.push(format!("start {stack}"));
            s.active.insert(stack.to_owned());
            Ok(())
        }
        async fn stop(&self, unit: &str) -> Result<(), String> {
            let stack = short(unit);
            let mut s = self.0.lock().unwrap();
            s.log.push(format!("stop {stack}"));
            if s.stop_fails.contains(stack) {
                return Err(format!("{stack} would not stop"));
            }
            s.active.remove(stack);
            Ok(())
        }
        async fn is_active(&self, unit: &str) -> bool {
            self.0.lock().unwrap().active.contains(short(unit))
        }
        async fn vram_used(&self) -> Result<Vec<u64>, String> {
            let s = self.0.lock().unwrap();
            let used = if s.active.iter().any(|a| a != "dashboard") {
                20_000
            } else {
                s.leak_mib
            };
            Ok(vec![used, used])
        }
        async fn probe(&self, url: &str) -> bool {
            let s = self.0.lock().unwrap();
            if let Some(&r) = s.ready_urls.get(url) {
                return r;
            }
            match s.url_stack.get(url) {
                Some(stack) => s.active.contains(stack) && !s.never_ready.contains(stack),
                None => false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn vram_lines() {
        assert_eq!(super::parse_vram("12\n24000\n").unwrap(), [12, 24000]);
        assert!(super::parse_vram("").is_err());
        assert!(super::parse_vram("N/A\n").is_err());
    }
}
