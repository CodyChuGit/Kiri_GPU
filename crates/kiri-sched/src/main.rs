//! kiri-sched: the GPU scheduler and supervisor for a Kiri GPU server.
//!
//! One runtime holds the GPUs at a time. Every request for a runtime comes
//! through the front door, which passes it straight on when its runtime is
//! up and otherwise holds it while the scheduler switches the box (drain,
//! stop, GPUs seen empty, start, ready) or refuses it with a reason and a
//! Retry-After. The Kiri daemon asks the lease socket before it gives any
//! container GPUs, so nothing starts on the cards behind the scheduler's
//! back. Light services are kept running, and every decision is logged.
//!
//! ```text
//! kiri-sched daemon [-c /etc/kiri-sched/sched.toml]
//! kiri-sched check-config [-c …]
//! kiri-sched hash-token            (reads a token on stdin)
//! kiri-sched status | release | reset
//! kiri-sched use <runtime> [minutes]
//! kiri-sched hold <runtime> [minutes]   (no minutes: until release)
//! kiri-sched unfence <runtime-or-service>
//! ```

#![forbid(unsafe_code)]

mod config;
mod core;
mod engine;
mod host;
mod lease;
mod notify;
mod server;

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use sha2::{Digest, Sha256};

const DEFAULT_CONFIG: &str = "/etc/kiri-sched/sched.toml";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (config, rest) = split_config(&args);
    let result = match rest.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["daemon"] => daemon(&config),
        ["check-config"] => load(&config).map(|c| println!("ok: {} runtimes, {} services, {} clients", c.runtimes.len(), c.services.len(), c.clients.len())),
        ["hash-token"] => hash_token(),
        ["status"] => admin(&config, "GET", "/box/status"),
        ["release"] => admin(&config, "POST", "/box/release"),
        ["reset"] => admin(&config, "POST", "/box/reset"),
        ["use", rt] => admin(&config, "POST", &format!("/box/use/{rt}")),
        ["use", rt, m] => admin(&config, "POST", &format!("/box/use/{rt}?minutes={m}")),
        ["hold", rt] => admin(&config, "POST", &format!("/box/hold/{rt}")),
        ["hold", rt, m] => admin(&config, "POST", &format!("/box/hold/{rt}?minutes={m}")),
        ["unfence", name] => admin(&config, "POST", &format!("/box/unfence/{name}")),
        _ => Err("usage: kiri-sched [-c config] daemon | check-config | hash-token | status | use <rt> [min] | hold <rt> [min] | release | unfence <name> | reset".into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kiri-sched: {e}");
            ExitCode::FAILURE
        }
    }
}

fn split_config(args: &[String]) -> (PathBuf, Vec<String>) {
    let mut config = PathBuf::from(DEFAULT_CONFIG);
    let mut rest = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "-c" || a == "--config" {
            if let Some(p) = it.next() {
                config = p.into();
            }
        } else {
            rest.push(a.clone());
        }
    }
    (config, rest)
}

fn load(path: &Path) -> Result<config::Config, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    config::Config::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn hash_token() -> Result<(), String> {
    let mut token = String::new();
    std::io::stdin()
        .read_to_string(&mut token)
        .map_err(|e| e.to_string())?;
    let token = token.trim();
    if token.len() < 24 {
        return Err("use a token of at least 24 characters (e.g. openssl rand -hex 24)".into());
    }
    let hex: String = Sha256::digest(token.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!("{hex}");
    Ok(())
}

fn daemon(config_path: &Path) -> Result<(), String> {
    let cfg = Arc::new(load(config_path)?);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    rt.block_on(async move {
        let engine = engine::Engine::new(
            cfg.clone(),
            host::System {
                cards_dir: cfg.sched.cards_dir.clone(),
            },
            Some(cfg.sched.state_file.clone()),
        );
        let front = server::Front::new(engine.clone());
        let tcp = tokio::net::TcpListener::bind(&cfg.sched.listen)
            .await
            .map_err(|e| format!("listen {}: {e}", cfg.sched.listen))?;
        let admin = bind_unix(&cfg.sched.admin_socket, 0o660)?;
        let lease = bind_unix(&cfg.sched.lease_socket, 0o600)?;
        engine::audit(
            "started",
            serde_json::json!({"listen": cfg.sched.listen, "home": cfg.sched.home}),
        );
        tokio::spawn(server::serve_tcp(front.clone(), tcp));
        tokio::spawn(server::serve_unix(front, admin));
        tokio::spawn(lease::serve(engine.clone(), lease));
        tokio::spawn(notify::watchdog());
        notify::send("READY=1");
        tokio::select! {
            () = engine.clone().run() => {}
            _ = tokio::signal::ctrl_c() => {}
            () = terminate() => {}
        }
        notify::send("STOPPING=1");
        engine::audit("stopping", serde_json::json!({}));
        engine.commit();
        Ok(())
    })
}

async fn terminate() {
    match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
        Ok(mut s) => {
            s.recv().await;
        }
        Err(_) => std::future::pending().await,
    }
}

fn bind_unix(path: &Path, mode: u32) -> Result<tokio::net::UnixListener, String> {
    use std::os::unix::fs::PermissionsExt;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    // A socket left by a crash; anything else at that path is not ours.
    if let Ok(meta) = std::fs::symlink_metadata(path) {
        use std::os::unix::fs::FileTypeExt;
        if !meta.file_type().is_socket() {
            return Err(format!("{} exists and is not a socket", path.display()));
        }
        std::fs::remove_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    let l = tokio::net::UnixListener::bind(path).map_err(|e| format!("{}: {e}", path.display()))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(l)
}

/// A request to the admin socket: plain HTTP/1.1, the body printed.
fn admin(config_path: &Path, method: &str, path: &str) -> Result<(), String> {
    let socket = load(config_path)
        .map(|c| c.sched.admin_socket)
        .unwrap_or_else(|_| PathBuf::from("/run/kiri-sched/admin.sock"));
    let mut s = std::os::unix::net::UnixStream::connect(&socket)
        .map_err(|e| format!("{}: {e} (is kiri-sched running?)", socket.display()))?;
    s.set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .map_err(|e| e.to_string())?;
    write!(s, "{method} {path} HTTP/1.1\r\nHost: kiri-sched\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").map_err(|e| e.to_string())?;
    let mut out = String::new();
    s.read_to_string(&mut out).map_err(|e| e.to_string())?;
    let (head, body) = out.split_once("\r\n\r\n").unwrap_or((&out, ""));
    let ok = head.split(' ').nth(1).is_some_and(|c| c.starts_with('2'));
    let body = serde_json::from_str::<serde_json::Value>(body.trim()).map_or_else(
        |_| body.to_owned(),
        |v| serde_json::to_string_pretty(&v).unwrap_or_default(),
    );
    println!("{body}");
    if ok {
        Ok(())
    } else {
        Err(head.lines().next().unwrap_or("no answer").to_owned())
    }
}
