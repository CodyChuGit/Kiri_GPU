//! The socket the Kiri daemon asks before it gives a container GPUs (see
//! `crates/kiri/src/daemon/gpulease.rs`): one line of JSON each way.

use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::time::timeout;

use crate::engine::{Engine, audit};
use crate::host::Host;

/// The answer to one question line.
pub fn answer<H: Host>(engine: &Engine<H>, line: &str) -> Value {
    let Ok(q) = serde_json::from_str::<Value>(line) else {
        return json!({"ok": false, "reason": "not JSON"});
    };
    if q["op"] != "gpu-start" {
        return json!({"ok": false, "reason": "unknown op"});
    }
    let verdict = engine.core().lease_ok(&q["labels"]);
    match verdict {
        Ok(()) => {
            audit(
                "lease-granted",
                json!({"container": q["id"], "devices": q["devices"]}),
            );
            json!({"ok": true})
        }
        Err(reason) => {
            audit(
                "lease-refused",
                json!({"container": q["id"], "labels": q["labels"], "reason": reason}),
            );
            json!({"ok": false, "reason": reason})
        }
    }
}

#[cfg(unix)]
pub async fn serve<H: Host>(engine: Arc<Engine<H>>, listener: tokio::net::UnixListener) {
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            continue;
        };
        let engine = engine.clone();
        tokio::spawn(async move {
            let (r, mut w) = stream.into_split();
            let mut line = String::new();
            let mut reader = BufReader::new(r.take(8192));
            let reply = match timeout(Duration::from_secs(3), reader.read_line(&mut line)).await {
                Ok(Ok(_)) => answer(&engine, line.trim()),
                _ => json!({"ok": false, "reason": "no question within 3 s"}),
            };
            let _ = w.write_all(format!("{reply}\n").as_bytes()).await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, EXAMPLE};
    use crate::host::fake::Fake;

    #[test]
    fn answers() {
        let e = Engine::new(
            Arc::new(Config::parse(EXAMPLE).unwrap()),
            Fake::default(),
            None,
        );
        assert_eq!(answer(&e, "garbage")["ok"], false);
        assert_eq!(answer(&e, r#"{"op":"other"}"#)["ok"], false);
        let q = r#"{"op":"gpu-start","id":"abc","labels":{"com.docker.compose.project":"maya"},"devices":["nvidia.com/gpu=all"]}"#;
        assert_eq!(answer(&e, q)["ok"], false, "nothing holds the GPUs yet");
        {
            let mut c = e.core();
            c.switch_begin("maya");
            c.switch_starting();
        }
        assert_eq!(answer(&e, q)["ok"], true);
    }
}
