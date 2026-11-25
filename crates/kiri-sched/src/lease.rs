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
    let devices: Vec<String> = q["devices"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|d| d.as_str().map(str::to_owned))
        .collect();
    // A daemon that can start a container on other devices than it asked
    // for says so; an older one only ever gets a yes or a no.
    let narrow = q["narrow"] == true;
    let verdict = engine.core().lease(&q["labels"], &devices, narrow);
    match verdict {
        Ok(None) => {
            audit(
                "lease-granted",
                json!({"container": q["id"], "devices": q["devices"]}),
            );
            json!({"ok": true})
        }
        Ok(Some(given)) => {
            audit(
                "lease-granted",
                json!({"container": q["id"], "asked": q["devices"], "devices": given,
                       "parent": q["labels"]["dev.kiri.sched.parent"]}),
            );
            json!({"ok": true, "devices": given})
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
            let p = c.want_switch(0).unwrap();
            c.switch_begin(&p);
            c.switch_starting();
        }
        assert_eq!(answer(&e, q)["ok"], true);
        let other = q.replace("maya", "comfyui");
        assert_eq!(answer(&e, &other)["ok"], false);
    }

    #[test]
    fn a_child_gets_its_parents_card() {
        let e = Engine::new(
            Arc::new(Config::parse(EXAMPLE).unwrap()),
            Fake::default(),
            None,
        );
        {
            let mut c = e.core();
            c.admit(0, "mocap", crate::config::Class::Interactive);
            let p = c.want_switch(0).unwrap();
            c.switch_begin(&p);
            c.switch_starting();
            c.switch_ready(0, "mocap", 0);
        }
        let card = e.core().status(0)["runtimes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["name"] == "mocap")
            .unwrap()["cards"][0]
            .as_u64()
            .unwrap();
        let q = r#"{"op":"gpu-start","id":"s1","labels":{"dev.kiri.sched.parent":"mocap"},"devices":["nvidia.com/gpu=all"],"narrow":true}"#;
        let a = answer(&e, q);
        assert_eq!(a["ok"], true, "{a}");
        assert_eq!(a["devices"], json!([format!("nvidia.com/gpu={card}")]));
        // An older daemon that cannot narrow is refused, not given all.
        let old = q.replace(r#","narrow":true"#, "");
        assert_eq!(answer(&e, &old)["ok"], false);
    }
}
