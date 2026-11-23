//! systemd's notify protocol (`Type=notify`, `WatchdogSec=`), without a
//! library: a datagram to `$NOTIFY_SOCKET`.

use std::os::unix::net::UnixDatagram;
use std::time::Duration;

pub fn send(state: &str) {
    let Ok(path) = std::env::var("NOTIFY_SOCKET") else {
        return;
    };
    let Ok(sock) = UnixDatagram::unbound() else {
        return;
    };
    if let Some(name) = path.strip_prefix('@') {
        #[cfg(target_os = "linux")]
        {
            use std::os::linux::net::SocketAddrExt;
            if let Ok(addr) = std::os::unix::net::SocketAddr::from_abstract_name(name.as_bytes()) {
                let _ = sock.send_to_addr(state.as_bytes(), &addr);
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = name;
    } else {
        let _ = sock.send_to(state.as_bytes(), &path);
    }
}

/// Pings the watchdog at half its interval, from the async runtime: a
/// runtime that stops making progress stops the pings, and systemd
/// restarts the scheduler.
pub async fn watchdog() {
    let Some(usec) = std::env::var("WATCHDOG_USEC")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    else {
        return;
    };
    let every = Duration::from_micros(usec / 2).max(Duration::from_millis(500));
    loop {
        send("WATCHDOG=1");
        tokio::time::sleep(every).await;
    }
}
