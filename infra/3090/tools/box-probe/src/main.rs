//! box-probe: a static, dependency-free healthcheck binary.
//!
//! Many slim/distroless images ship without curl or wget, so Docker-style
//! HEALTHCHECKs have nothing to run. Build this once with the musl target and
//! bind-mount it read-only into every Kiri container on the 3090 box at /opt/box/bin/box-probe.
//!
//! Usage:
//!   box-probe http://127.0.0.1:5080/healthz   # exit 0 on HTTP 2xx/3xx
//!   box-probe tcp://127.0.0.1:445             # exit 0 if the port accepts
//!
//! Options (env): BOX_PROBE_TIMEOUT_MS (default 3000).
//! Only plain HTTP is supported on purpose: probes target loopback.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::process::ExitCode;
use std::time::Duration;

fn timeout() -> Duration {
    let ms = std::env::var("BOX_PROBE_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(3000);
    Duration::from_millis(ms.max(1))
}

fn resolve(hostport: &str) -> Result<SocketAddr, String> {
    hostport
        .to_socket_addrs()
        .map_err(|e| format!("resolve {hostport}: {e}"))?
        .next()
        .ok_or_else(|| format!("resolve {hostport}: no address"))
}

fn connect(hostport: &str) -> Result<TcpStream, String> {
    let addr = resolve(hostport)?;
    let t = timeout();
    let s = TcpStream::connect_timeout(&addr, t).map_err(|e| format!("connect {addr}: {e}"))?;
    s.set_read_timeout(Some(t)).ok();
    s.set_write_timeout(Some(t)).ok();
    Ok(s)
}

/// Split "host:port/path" into ("host:port", "/path"), defaulting the port.
fn split_target(rest: &str, default_port: u16) -> (String, String) {
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let has_port = if authority.starts_with('[') {
        authority.contains("]:")
    } else {
        authority.contains(':')
    };
    let hostport = if has_port {
        authority.to_string()
    } else {
        format!("{authority}:{default_port}")
    };
    (hostport, path.to_string())
}

fn probe_http(rest: &str) -> Result<(), String> {
    let (hostport, path) = split_target(rest, 80);
    let host = hostport.rsplit_once(':').map(|(h, _)| h).unwrap_or(&hostport);
    let mut s = connect(&hostport)?;
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: box-probe/0.1\r\nConnection: close\r\n\r\n"
    );
    s.write_all(req.as_bytes()).map_err(|e| format!("write: {e}"))?;
    let mut buf = [0u8; 64];
    let mut n = 0;
    while n < 12 {
        match s.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) => return Err(format!("read: {e}")),
        }
    }
    let head = std::str::from_utf8(&buf[..n]).unwrap_or("");
    // "HTTP/1.1 200 ..."
    let code: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .ok_or_else(|| format!("bad status line: {head:?}"))?;
    if (200..400).contains(&code) {
        Ok(())
    } else {
        Err(format!("HTTP {code}"))
    }
}

fn probe_tcp(rest: &str) -> Result<(), String> {
    let (hostport, _) = split_target(rest, 0);
    connect(&hostport).map(|_| ())
}

fn run(target: &str) -> Result<(), String> {
    if let Some(rest) = target.strip_prefix("http://") {
        probe_http(rest)
    } else if let Some(rest) = target.strip_prefix("tcp://") {
        probe_tcp(rest)
    } else {
        Err(format!("unsupported target {target:?} (use http:// or tcp://)"))
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: box-probe <http://host:port/path | tcp://host:port> ...");
        return ExitCode::from(2);
    }
    // Every target must pass.
    for t in &args {
        if let Err(e) = run(t) {
            eprintln!("box-probe: {t}: {e}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_targets() {
        assert_eq!(
            split_target("127.0.0.1:5080/healthz", 80),
            ("127.0.0.1:5080".into(), "/healthz".into())
        );
        assert_eq!(split_target("localhost", 80), ("localhost:80".into(), "/".into()));
        assert_eq!(split_target("[::1]:8686/health", 80), ("[::1]:8686".into(), "/health".into()));
    }

    #[test]
    fn http_and_tcp_against_local_server() {
        use std::net::TcpListener;
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let h = std::thread::spawn(move || {
            for code in ["200 OK", "503 Service Unavailable"] {
                let (mut c, _) = l.accept().unwrap();
                let mut b = [0u8; 512];
                let _ = c.read(&mut b);
                let _ = c.write_all(format!("HTTP/1.1 {code}\r\nContent-Length: 0\r\n\r\n").as_bytes());
            }
            let _ = l.accept(); // tcp probe
        });
        assert!(run(&format!("http://127.0.0.1:{port}/healthz")).is_ok());
        assert!(run(&format!("http://127.0.0.1:{port}/healthz")).is_err());
        assert!(run(&format!("tcp://127.0.0.1:{port}")).is_ok());
        h.join().unwrap();
        assert!(run("ftp://x").is_err());
    }
}
