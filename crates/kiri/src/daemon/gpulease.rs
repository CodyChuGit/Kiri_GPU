//! A GPU lease check before a container gets GPUs.
//!
//! On a GPU server one scheduler (`kiri-sched`) decides which runtime holds
//! the cards. Kiri asks it before starting any container that wants an
//! NVIDIA GPU, so a plain `kiri run --gpus all` or a stray Compose file
//! cannot put a second runtime on cards another one holds.
//!
//! The question goes over a Unix socket as one line of JSON:
//! `{"op":"gpu-start","id":…,"labels":{…},"devices":[…],"narrow":true}`, and
//! the answer is one line back, `{"ok":true}` or `{"ok":false,"reason":"…"}`.
//! `narrow` says this daemon can start the container on other GPUs than it
//! asked for: a yes may then carry `"devices":["nvidia.com/gpu=1"]`, and
//! the container gets exactly those (a runtime's own child containers,
//! such as game sessions, asking for `all` while the runtime holds one card).
//!
//! - No socket and no [`REQUIRED`] marker: no scheduler on this host, so
//!   the start goes ahead as on any Kiri.
//! - No socket but the marker exists: refused. The host was set up to have
//!   a scheduler, and one that is down must not mean "anything goes".
//! - A socket that does not answer in time, or answers something else:
//!   refused, for the same reason.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Duration;

use serde_json::{Value, json};

use super::cdi::Want;

/// Where the scheduler listens.
pub const SOCKET: &str = "/run/kiri-sched/lease.sock";

/// Present on a host where GPU starts must have the scheduler's yes.
pub const REQUIRED: &str = "/etc/kiri/gpu-lease-required";

/// How long the scheduler has to answer.
const TIMEOUT: Duration = Duration::from_secs(3);

/// The NVIDIA GPUs among what a container asks for, as names
/// (`nvidia.com/gpu=all`, `nvidia.com/gpu=0`, `nvidia.com/gpu=count:2`).
pub fn gpus(wants: &[Want]) -> Vec<String> {
    wants
        .iter()
        .filter_map(|w| match w {
            Want::Gpus(n) if *n < 0 => Some("nvidia.com/gpu=all".to_owned()),
            Want::Gpus(n) => Some(format!("nvidia.com/gpu=count:{n}")),
            Want::Named(name) if name.starts_with("nvidia.com/gpu=") => Some(name.clone()),
            Want::Named(_) => None,
        })
        .collect()
}

/// Asks before a start; `Ok(Some(devices))` is a yes on those GPUs instead
/// of the ones asked for, `Err` carries the refusal for the client.
pub fn check(id: &str, labels: &Value, devices: &[String]) -> Result<Option<Vec<String>>, String> {
    if devices.is_empty() {
        return Ok(None);
    }
    check_at(Path::new(SOCKET), Path::new(REQUIRED), id, labels, devices)
}

fn check_at(socket: &Path, required: &Path, id: &str, labels: &Value, devices: &[String]) -> Result<Option<Vec<String>>, String> {
    if !socket.exists() {
        return if required.exists() {
            Err(format!("GPU start refused: the GPU scheduler is not running ({} is missing and {} requires it)", socket.display(), required.display()))
        } else {
            Ok(None)
        };
    }
    let request = json!({"op": "gpu-start", "id": id, "labels": labels, "devices": devices, "narrow": true});
    let answer = ask(socket, &request).map_err(|e| format!("GPU start refused: no answer from the GPU scheduler: {e}"))?;
    verdict(&answer)
}

#[cfg(unix)]
fn ask(socket: &Path, request: &Value) -> std::io::Result<String> {
    let mut s = std::os::unix::net::UnixStream::connect(socket)?;
    s.set_read_timeout(Some(TIMEOUT))?;
    s.set_write_timeout(Some(TIMEOUT))?;
    let mut line = request.to_string();
    line.push('\n');
    s.write_all(line.as_bytes())?;
    let mut answer = String::new();
    // One line, and never more than a few KiB: a scheduler that sends more
    // is not one this code understands.
    BufReader::new(std::io::Read::take(s, 8192)).read_line(&mut answer)?;
    Ok(answer)
}

#[cfg(not(unix))]
fn ask(_socket: &Path, _request: &Value) -> std::io::Result<String> {
    Err(std::io::Error::other("GPU leases need a Unix socket"))
}

/// The scheduler's answer as a start's outcome; anything unclear is a no.
fn verdict(answer: &str) -> Result<Option<Vec<String>>, String> {
    let unclear = || "GPU start refused: the GPU scheduler's answer was not understood".to_owned();
    let v: Value = serde_json::from_str(answer.trim()).map_err(|_| unclear())?;
    match v["ok"].as_bool() {
        Some(true) if v.get("devices").is_none_or(Value::is_null) => Ok(None),
        Some(true) => {
            // Only single NVIDIA GPUs by index: what a lease limits to.
            let given: Vec<String> = v["devices"].as_array().ok_or_else(unclear)?.iter().map(|d| d.as_str().map(str::to_owned)).collect::<Option<_>>().ok_or_else(unclear)?;
            let ok = !given.is_empty() && given.iter().all(|d| d.strip_prefix("nvidia.com/gpu=").is_some_and(|i| !i.is_empty() && i.bytes().all(|b| b.is_ascii_digit())));
            if ok { Ok(Some(given)) } else { Err(unclear()) }
        }
        Some(false) => Err(format!("GPU start refused by the GPU scheduler: {}", v["reason"].as_str().unwrap_or("no reason given"))),
        None => Err("GPU start refused: the GPU scheduler's answer was not understood".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_nvidia_gpus_count() {
        let w = [Want::Gpus(-1), Want::Gpus(2), Want::Named("nvidia.com/gpu=1".into()), Want::Named("vendor.com/fpga=0".into())];
        assert_eq!(gpus(&w), ["nvidia.com/gpu=all", "nvidia.com/gpu=count:2", "nvidia.com/gpu=1"]);
        assert!(gpus(&[Want::Named("vendor.com/fpga=0".into())]).is_empty());
    }

    #[test]
    fn verdicts() {
        assert_eq!(verdict("{\"ok\":true}\n"), Ok(None));
        assert_eq!(verdict("{\"ok\":true,\"devices\":[\"nvidia.com/gpu=1\"]}"), Ok(Some(vec!["nvidia.com/gpu=1".to_owned()])));
        for bad in ["[]", "[\"nvidia.com/gpu=all\"]", "[\"vendor.com/x=1\"]", "[1]", "\"nvidia.com/gpu=1\""] {
            assert!(verdict(&format!("{{\"ok\":true,\"devices\":{bad}}}")).is_err(), "{bad}");
        }
        assert!(verdict("{\"ok\":false,\"reason\":\"maya holds the GPUs\"}").unwrap_err().contains("maya holds the GPUs"));
        assert!(verdict("").is_err());
        assert!(verdict("yes").is_err());
        assert!(verdict("{\"ok\":\"true\"}").is_err());
    }

    #[test]
    fn no_scheduler_installed_is_allowed_unless_required() {
        let dir = std::env::temp_dir().join(format!("kiri-gpulease-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let sock = dir.join("lease.sock");
        let req = dir.join("required");
        let dev = ["nvidia.com/gpu=all".to_owned()];
        assert!(check_at(&sock, &req, "abc", &json!({}), &dev).is_ok());
        std::fs::write(&req, "").unwrap();
        assert!(check_at(&sock, &req, "abc", &json!({}), &dev).unwrap_err().contains("not running"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn asks_the_socket() {
        use std::os::unix::net::UnixListener;
        let dir = std::env::temp_dir().join(format!("kiri-gpulease-sock-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let sock = dir.join("lease.sock");
        let listener = UnixListener::bind(&sock).unwrap();
        let server = std::thread::spawn(move || {
            for answer in ["{\"ok\":true}\n", "{\"ok\":false,\"reason\":\"held by desktop\"}\n"] {
                let (s, _) = listener.accept().unwrap();
                let mut line = String::new();
                BufReader::new(&s).read_line(&mut line).unwrap();
                let v: Value = serde_json::from_str(&line).unwrap();
                assert_eq!(v["op"], "gpu-start");
                assert_eq!(v["devices"][0], "nvidia.com/gpu=all");
                (&s).write_all(answer.as_bytes()).unwrap();
            }
        });
        let dev = ["nvidia.com/gpu=all".to_owned()];
        let none = dir.join("absent");
        assert!(check_at(&sock, &none, "abc", &json!({"dev.kiri.sched.runtime": "maya"}), &dev).is_ok());
        assert!(check_at(&sock, &none, "abc", &json!({}), &dev).unwrap_err().contains("held by desktop"));
        server.join().unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
