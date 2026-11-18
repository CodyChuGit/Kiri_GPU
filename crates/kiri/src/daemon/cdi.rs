//! GPUs and other vendor devices through CDI, the Container Device
//! Interface (<https://github.com/cncf-tags/container-device-interface>).
//!
//! A vendor's tool writes a spec that lists its devices and what a
//! container needs to use them: device nodes, files and libraries to bind
//! in, environment and hooks. NVIDIA's is `nvidia-ctk cdi generate`, which
//! writes `/etc/cdi/nvidia.yaml`. Kiri reads the specs on the Linux host at
//! every start (a driver update changes the library paths) and turns what
//! they ask for into what it already gives a container: `device` lines,
//! bind mounts, environment, and `cdi` lines that the guest agent carries
//! out itself. There is no runc to run OCI hooks, so the hooks NVIDIA
//! writes (symlinks, the ld.so cache, CUDA forward compatibility, device
//! node permissions) are done natively by the agent; any other hook is
//! refused rather than skipped.
//!
//! What asks for devices:
//! - `--gpus all`, `--gpus 2`, `--gpus device=0,1` and Compose's
//!   `deploy.resources.reservations.devices` (driver `nvidia` or none, with
//!   the `gpu` capability): `DeviceRequests`, made into `nvidia.com/gpu=…`.
//! - `--device nvidia.com/gpu=all`, or a request with driver `cdi`: any
//!   fully-qualified CDI name, `vendor.com/class=name`.

use std::path::{Path, PathBuf};

use serde_json::Value;

/// Where specs live, lowest priority first: a device named in both comes
/// from the later directory, as the CDI spec orders them.
pub const SPEC_DIRS: [&str; 2] = ["/etc/cdi", "/var/run/cdi"];

/// The kind `--gpus` asks for.
const NVIDIA_GPU: &str = "nvidia.com/gpu";

/// What a container asks for, before the specs are read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Want {
    /// A fully-qualified device name, `vendor.com/class=name`.
    Named(String),
    /// `--gpus N` (or `all`, -1): that many NVIDIA GPUs.
    Gpus(i64),
}

/// What the specs add to a container's start.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Edits {
    /// `KEY=value`, replacing the same key from the image or `-e`.
    pub env: Vec<String>,
    /// Bind mounts as the guest takes them: `source\ttarget\tflags`.
    pub mounts: Vec<String>,
    /// Guest `opts` lines: `device`, `group` and `cdi`.
    pub opts: Vec<String>,
    /// Hooks left out (an NVIDIA hook this version does not know), for the log.
    pub skipped: Vec<String>,
}

/// One spec file's kind, its own edits and its devices.
#[derive(Debug, Clone)]
pub struct Spec {
    pub kind: String,
    pub edits: Value,
    pub devices: Vec<(String, Value)>,
    pub file: PathBuf,
}

/// Whether `name` is a fully-qualified CDI device name:
/// `vendor/class=name`, with the spec's character sets.
pub fn is_qualified(name: &str) -> bool {
    let Some((kind, dev)) = name.split_once('=') else { return false };
    let Some((vendor, class)) = kind.split_once('/') else { return false };
    let ok = |s: &str, extra: &[u8]| !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || extra.contains(&b));
    ok(vendor, b"._-")
        && vendor.as_bytes()[0].is_ascii_alphanumeric()
        && ok(class, b"_-")
        && ok(dev, b"_.:-")
        && dev.as_bytes()[0].is_ascii_alphanumeric()
}

/// The devices a container's kept HostConfig settings ask for, or
/// Docker's error for a request no driver takes.
pub fn wanted(opts: &Value) -> Result<Vec<Want>, String> {
    let mut out = Vec::new();
    for r in opts["DeviceRequests"].as_array().into_iter().flatten() {
        let driver = r["Driver"].as_str().unwrap_or("");
        let ids: Vec<&str> = r["DeviceIDs"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        let caps: Vec<Vec<&str>> =
            r["Capabilities"].as_array().into_iter().flatten().map(|set| set.as_array().into_iter().flatten().filter_map(Value::as_str).collect()).collect();
        match driver {
            "cdi" => {
                for id in ids {
                    if !is_qualified(id) {
                        return Err(format!("invalid CDI device name {id:?}: want vendor.com/class=name"));
                    }
                    out.push(Want::Named(id.to_owned()));
                }
            }
            // Docker picks the NVIDIA driver for an empty driver when a
            // capability set asks for `gpu`.
            "nvidia" | "" if driver == "nvidia" || caps.iter().any(|set| set.contains(&"gpu")) => {
                if ids.is_empty() {
                    let count = r["Count"].as_i64().unwrap_or(0);
                    if count != 0 {
                        out.push(Want::Gpus(count));
                    }
                } else {
                    for id in ids {
                        // `--gpus device=0,1` arrives split; a name with its own
                        // vendor (from Compose) is taken as it is.
                        for one in id.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                            let name = if is_qualified(one) { one.to_owned() } else { format!("{NVIDIA_GPU}={one}") };
                            if !is_qualified(&name) {
                                return Err(format!("invalid GPU device {one:?}"));
                            }
                            out.push(Want::Named(name));
                        }
                    }
                }
            }
            _ => {
                let caps = caps.iter().map(|set| format!("{set:?}")).collect::<Vec<_>>().join(" ");
                return Err(format!("could not select device driver {driver:?} with capabilities: [{caps}]"));
            }
        }
    }
    // `--device nvidia.com/gpu=all`, when a client sends the name as a
    // device rather than as a `cdi` request.
    for d in opts["Devices"].as_array().into_iter().flatten() {
        if let Some(name) = d["PathOnHost"].as_str().filter(|p| is_qualified(p)) {
            out.push(Want::Named(name.to_owned()));
        }
    }
    Ok(out)
}

/// Reads every spec under `dirs`, lowest priority first. A file that does
/// not parse is an error naming it: a broken spec must not quietly drop a
/// GPU.
pub fn load(dirs: &[&Path]) -> Result<Vec<Spec>, String> {
    let mut specs = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        let mut files: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("json" | "yaml" | "yml")))
            .collect();
        files.sort();
        for file in files {
            let text = std::fs::read_to_string(&file).map_err(|e| format!("CDI spec {}: {e}", file.display()))?;
            let v = if file.extension().and_then(|e| e.to_str()) == Some("json") {
                serde_json::from_str(&text).map_err(|e| e.to_string())
            } else {
                super::yaml::parse(&text)
            }
            .map_err(|e| format!("CDI spec {}: {e}", file.display()))?;
            specs.push(parse_spec(v, file)?);
        }
    }
    Ok(specs)
}

fn parse_spec(v: Value, file: PathBuf) -> Result<Spec, String> {
    let kind = v["kind"].as_str().filter(|k| k.contains('/')).ok_or_else(|| format!("CDI spec {}: no kind", file.display()))?.to_owned();
    let mut devices = Vec::new();
    for d in v["devices"].as_array().into_iter().flatten() {
        let name = d["name"].as_str().ok_or_else(|| format!("CDI spec {}: a device has no name", file.display()))?;
        devices.push((name.to_owned(), d["containerEdits"].clone()));
    }
    Ok(Spec { kind, edits: v["containerEdits"].clone(), devices, file })
}

/// The edits for `wants`, from `specs` (as [`load`] returns them).
pub fn resolve(specs: &[Spec], wants: &[Want]) -> Result<Edits, String> {
    let mut names = Vec::new();
    for w in wants {
        match w {
            Want::Named(n) => names.push(n.clone()),
            Want::Gpus(count) => names.extend(gpus(specs, *count)?),
        }
    }
    let mut out = Edits::default();
    // Each spec's own edits once, then each device's, in the order asked.
    let mut used: Vec<&Path> = Vec::new();
    let mut done: Vec<&str> = Vec::new();
    for name in &names {
        if done.contains(&name.as_str()) {
            continue;
        }
        let (kind, dev) = name.split_once('=').expect("qualified");
        let (spec, edits) = specs
            .iter()
            .rev()
            .filter(|s| s.kind == kind)
            .find_map(|s| s.devices.iter().find(|(n, _)| n == dev).map(|(_, e)| (s, e)))
            .ok_or_else(|| unresolvable(specs, name))?;
        if !used.contains(&spec.file.as_path()) {
            used.push(&spec.file);
            apply(&spec.edits, &mut out)?;
        }
        apply(edits, &mut out)?;
        done.push(name);
    }
    Ok(out)
}

fn unresolvable(specs: &[Spec], name: &str) -> String {
    if specs.is_empty() {
        let hint = if name.starts_with("nvidia.com/") { "; run `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`" } else { "" };
        return format!("could not resolve CDI device {name}: no CDI specs in {}{hint}", SPEC_DIRS.join(" or "));
    }
    format!("could not resolve CDI device {name}: unresolvable CDI devices {name}")
}

/// `--gpus N`: `all` when the spec has it and every GPU is asked for, else
/// the first N GPUs by index.
fn gpus(specs: &[Spec], count: i64) -> Result<Vec<String>, String> {
    let mut indexes: Vec<u32> = specs.iter().filter(|s| s.kind == NVIDIA_GPU).flat_map(|s| s.devices.iter().filter_map(|(n, _)| n.parse().ok())).collect();
    indexes.sort_unstable();
    indexes.dedup();
    let has_all = specs.iter().any(|s| s.kind == NVIDIA_GPU && s.devices.iter().any(|(n, _)| n == "all"));
    if count < 0 {
        if has_all {
            return Ok(vec![format!("{NVIDIA_GPU}=all")]);
        }
        if indexes.is_empty() {
            return Err(unresolvable(specs, &format!("{NVIDIA_GPU}=all")));
        }
        return Ok(indexes.iter().map(|i| format!("{NVIDIA_GPU}={i}")).collect());
    }
    let n = count as usize;
    if indexes.len() < n {
        return Err(format!("could not select {count} GPUs: the CDI specs list {}", indexes.len()));
    }
    Ok(indexes[..n].iter().map(|i| format!("{NVIDIA_GPU}={i}")).collect())
}

/// A path the guest can take in a tab-separated line.
fn path_ok(p: &str) -> bool {
    p.starts_with('/') && !p.contains(['\t', '\n']) && !p.split('/').any(|c| c == "..")
}

fn strings(v: &Value) -> impl Iterator<Item = &str> {
    v.as_array().into_iter().flatten().filter_map(Value::as_str)
}

fn push_once(list: &mut Vec<String>, line: String) {
    if !list.contains(&line) {
        list.push(line);
    }
}

/// Adds one `containerEdits` object to `out`.
fn apply(edits: &Value, out: &mut Edits) -> Result<(), String> {
    for e in strings(&edits["env"]) {
        let Some((key, _)) = e.split_once('=') else { return Err(format!("CDI env {e:?} is not KEY=value")) };
        out.env.retain(|x| x.split_once('=').map(|(k, _)| k) != Some(key));
        out.env.push(e.to_owned());
    }
    for d in edits["deviceNodes"].as_array().into_iter().flatten() {
        let path = d["path"].as_str().unwrap_or("");
        let host = d["hostPath"].as_str().filter(|p| !p.is_empty()).unwrap_or(path);
        let perms = d["permissions"].as_str().filter(|p| !p.is_empty()).unwrap_or("rwm");
        if !path_ok(path) || !path_ok(host) {
            return Err(format!("CDI device node {path:?}: not an absolute path"));
        }
        if perms.is_empty() || perms.len() > 3 || !perms.chars().all(|c| "rwm".contains(c)) {
            return Err(format!("CDI device node {path}: invalid permissions {perms:?}"));
        }
        push_once(&mut out.opts, format!("device\t{host}\t{path}\t{perms}"));
    }
    for m in edits["mounts"].as_array().into_iter().flatten() {
        let host = m["hostPath"].as_str().unwrap_or("");
        let target = m["containerPath"].as_str().unwrap_or("");
        let kind = m["type"].as_str().unwrap_or("bind");
        let options: Vec<&str> = strings(&m["options"]).collect();
        if !matches!(kind, "" | "bind" | "rbind") && !options.iter().any(|o| *o == "bind" || *o == "rbind") {
            return Err(format!("CDI mount {target}: type {kind:?} is not supported, only bind mounts"));
        }
        if !path_ok(host) || !path_ok(target) {
            return Err(format!("CDI mount {host:?} on {target:?}: not an absolute path"));
        }
        let flags = if options.contains(&"ro") { "ro" } else { "" };
        push_once(&mut out.mounts, format!("{host}\t{target}\t{flags}"));
    }
    for h in edits["hooks"].as_array().into_iter().flatten() {
        hook(h, out)?;
    }
    for g in edits["additionalGids"].as_array().or_else(|| edits["additionalGIDs"].as_array()).into_iter().flatten() {
        let gid = g.as_u64().filter(|g| *g <= u32::MAX as u64).ok_or_else(|| format!("CDI additional GID {g} is not a group id"))?;
        push_once(&mut out.opts, format!("group\t{gid}"));
    }
    Ok(())
}

/// The values of `--name X` and `--name=X` in `args`.
fn flag<'a>(args: &[&'a str], name: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if let Some(v) = a.strip_prefix(name).and_then(|rest| rest.strip_prefix('=')) {
            out.push(v);
        } else if *a == name
            && let Some(v) = it.next()
        {
            out.push(*v);
        }
    }
    out
}

/// One OCI hook from a spec. NVIDIA's (`nvidia-cdi-hook <command>`, or the
/// older `nvidia-ctk hook <command>`) become `cdi` lines the agent carries
/// out; anything else would need an OCI runtime, and is refused.
fn hook(h: &Value, out: &mut Edits) -> Result<(), String> {
    let path = h["path"].as_str().unwrap_or("");
    let args: Vec<&str> = strings(&h["args"]).collect();
    let base = Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or("");
    let mut rest: &[&str] = args.get(1..).unwrap_or(&[]);
    if base == "nvidia-ctk" && rest.first() == Some(&"hook") {
        rest = &rest[1..];
    }
    let shown = || format!("{path} {}", args.get(1..).unwrap_or(&[]).join(" "));
    if !matches!(base, "nvidia-cdi-hook" | "nvidia-ctk") {
        return Err(format!("CDI hook {} is not supported: Kiri runs NVIDIA's hooks itself and has no OCI runtime for others", shown()));
    }
    let Some((command, flags)) = rest.split_first() else { return Ok(()) };
    let line = |out: &mut Edits, l: String| {
        if l.contains('\n') {
            return Err(format!("CDI hook {}: bad argument", shown()));
        }
        push_once(&mut out.opts, l);
        Ok(())
    };
    match *command {
        "create-symlinks" => {
            for link in flag(flags, "--link") {
                let (target, at) = link.split_once("::").ok_or_else(|| format!("CDI hook {}: bad --link {link:?}", shown()))?;
                if !path_ok(at) || target.is_empty() || target.contains('\t') {
                    return Err(format!("CDI hook {}: bad --link {link:?}", shown()));
                }
                line(out, format!("cdi\tlink\t{target}\t{at}"))?;
            }
        }
        "update-ldcache" => {
            for folder in flag(flags, "--folder") {
                if !path_ok(folder) {
                    return Err(format!("CDI hook {}: bad --folder {folder:?}", shown()));
                }
                line(out, format!("cdi\tfolder\t{folder}"))?;
            }
            line(out, "cdi\tldconfig".to_owned())?;
        }
        "enable-cuda-compat" => {
            if let Some(v) = flag(flags, "--host-driver-version").first().filter(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit() || b == b'.')) {
                line(out, format!("cdi\tcompat\t{v}"))?;
            }
        }
        "chmod" => {
            let mode = flag(flags, "--mode").first().and_then(|m| u32::from_str_radix(m, 8).ok()).filter(|m| *m <= 0o7777);
            let Some(mode) = mode else { return Err(format!("CDI hook {}: bad --mode", shown())) };
            for p in flag(flags, "--path") {
                if !path_ok(p) {
                    return Err(format!("CDI hook {}: bad --path {p:?}", shown()));
                }
                line(out, format!("cdi\tchmod\t{mode:o}\t{p}"))?;
            }
        }
        "disable-device-node-modification" => line(out, "cdi\tnomodify".to_owned())?,
        _ => out.skipped.push(shown()),
    }
    Ok(())
}

/// The edits for a container's kept HostConfig settings (`opts`, from
/// `hostopts::from_host`), read from the host's specs; None when it asks
/// for no devices. Only a Linux host has GPUs to give: in the Mac's VM
/// there are none.
pub fn for_container(opts: &Value) -> Result<Option<Edits>, String> {
    let wants = wanted(opts)?;
    if wants.is_empty() {
        return Ok(None);
    }
    if !cfg!(target_os = "linux") {
        return Err("Kiri gives containers GPUs and other CDI devices only on a Linux host".to_owned());
    }
    let dirs: Vec<&Path> = SPEC_DIRS.iter().map(Path::new).collect();
    let specs = load(&dirs)?;
    resolve(&specs, &wants).map(Some)
}

/// `env` with the edits' environment: a key set in both takes the spec's value.
pub fn merge_env(env: &[String], edits: &[String]) -> Vec<String> {
    let key = |e: &str| e.split_once('=').map_or(e, |(k, _)| k).to_owned();
    let mut out: Vec<String> = env.iter().filter(|e| !edits.iter().any(|x| key(x) == key(e))).cloned().collect();
    out.extend(edits.iter().cloned());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// What `nvidia-ctk cdi generate` (1.17) writes on a two-GPU box, cut down.
    const NVIDIA: &str = r#"---
cdiVersion: 0.5.0
containerEdits:
  deviceNodes:
  - path: /dev/nvidia-modeset
  - path: /dev/nvidia-uvm
  - path: /dev/nvidia-uvm-tools
  - path: /dev/nvidiactl
  env:
  - NVIDIA_VISIBLE_DEVICES=void
  hooks:
  - args:
    - nvidia-cdi-hook
    - create-symlinks
    - --link
    - libnvidia-allocator.so.580.95.05::/usr/lib/x86_64-linux-gnu/gbm/nvidia-drm_gbm.so
    hookName: createContainer
    path: /usr/bin/nvidia-cdi-hook
  - args:
    - nvidia-cdi-hook
    - enable-cuda-compat
    - --host-driver-version=580.95.05
    hookName: createContainer
    path: /usr/bin/nvidia-cdi-hook
  - args: [nvidia-cdi-hook, update-ldcache, --folder, /usr/lib/x86_64-linux-gnu]
    hookName: createContainer
    path: /usr/bin/nvidia-cdi-hook
  - args:
    - nvidia-cdi-hook
    - disable-device-node-modification
    hookName: createContainer
    path: /usr/bin/nvidia-cdi-hook
  mounts:
  - containerPath: /usr/lib/x86_64-linux-gnu/libcuda.so.580.95.05
    hostPath: /usr/lib/x86_64-linux-gnu/libcuda.so.580.95.05
    options:
    - ro
    - nosuid
    - nodev
    - rbind
    - rprivate
  - containerPath: /usr/bin/nvidia-smi
    hostPath: /usr/bin/nvidia-smi
    options: ["ro", "nosuid", "nodev", "rbind", "rprivate"]
devices:
- containerEdits:
    deviceNodes:
    - path: /dev/nvidia0
    - path: /dev/dri/card1
    - path: /dev/dri/renderD128
    hooks:
    - args:
      - nvidia-cdi-hook
      - create-symlinks
      - --link
      - ../card1::/dev/dri/by-path/pci-0000:21:00.0-card
      hookName: createContainer
      path: /usr/bin/nvidia-cdi-hook
    - args: [nvidia-cdi-hook, chmod, --mode, "755", --path, /dev/dri]
      hookName: createContainer
      path: /usr/bin/nvidia-cdi-hook
  name: "0"
- containerEdits:
    deviceNodes:
    - path: /dev/nvidia1
  name: "1"
- containerEdits:
    deviceNodes:
    - path: /dev/nvidia0
    - path: /dev/nvidia1
  name: all
- containerEdits:
    deviceNodes:
    - path: /dev/nvidia1
  name: GPU-4a1b2c3d-0000-1111-2222-333344445555
kind: nvidia.com/gpu
"#;

    fn specs() -> Vec<Spec> {
        vec![parse_spec(super::super::yaml::parse(NVIDIA).unwrap(), PathBuf::from("/etc/cdi/nvidia.yaml")).unwrap()]
    }

    #[test]
    fn qualified_names() {
        assert!(is_qualified("nvidia.com/gpu=all"));
        assert!(is_qualified("nvidia.com/gpu=0"));
        assert!(is_qualified("nvidia.com/gpu=GPU-4a1b2c3d-0000"));
        assert!(is_qualified("nvidia.com/mig=0:1"));
        assert!(!is_qualified("/dev/nvidia0"));
        assert!(!is_qualified("nvidia.com/gpu"));
        assert!(!is_qualified("gpu=0"));
        assert!(!is_qualified("nvidia.com/gpu="));
        assert!(!is_qualified("nvidia.com/g.pu=0"));
    }

    #[test]
    fn what_docker_sends_for_each_way_of_asking() {
        // docker run --gpus all
        let all = json!({"DeviceRequests": [{"Driver": "", "Count": -1, "DeviceIDs": null, "Capabilities": [["gpu"]], "Options": {}}]});
        assert_eq!(wanted(&all).unwrap(), [Want::Gpus(-1)]);
        // docker run --gpus 1
        assert_eq!(wanted(&json!({"DeviceRequests": [{"Count": 1, "Capabilities": [["gpu"]]}]})).unwrap(), [Want::Gpus(1)]);
        // docker run --gpus '"device=0,1"'
        let ids = json!({"DeviceRequests": [{"Driver": "", "DeviceIDs": ["0", "1"], "Capabilities": [["gpu"]]}]});
        assert_eq!(wanted(&ids).unwrap(), [Want::Named("nvidia.com/gpu=0".into()), Want::Named("nvidia.com/gpu=1".into())]);
        // Compose: driver nvidia, count all, capabilities [gpu]
        assert_eq!(wanted(&json!({"DeviceRequests": [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}]})).unwrap(), [Want::Gpus(-1)]);
        // docker run --device nvidia.com/gpu=all (Docker 25+ sends a cdi request)
        let cdi = json!({"DeviceRequests": [{"Driver": "cdi", "DeviceIDs": ["nvidia.com/gpu=all"]}]});
        assert_eq!(wanted(&cdi).unwrap(), [Want::Named("nvidia.com/gpu=all".into())]);
        // …or an older client, as a device.
        let dev = json!({"Devices": [{"PathOnHost": "nvidia.com/gpu=0", "PathInContainer": "nvidia.com/gpu=0", "CgroupPermissions": "rwm"}, {"PathOnHost": "/dev/fuse"}]});
        assert_eq!(wanted(&dev).unwrap(), [Want::Named("nvidia.com/gpu=0".into())]);
        assert!(wanted(&json!({})).unwrap().is_empty());
        assert!(wanted(&json!({"DeviceRequests": [{"Driver": "", "Count": 0, "Capabilities": [["gpu"]]}]})).unwrap().is_empty());
    }

    #[test]
    fn requests_no_driver_takes_get_dockers_error() {
        let e = wanted(&json!({"DeviceRequests": [{"Driver": "", "Count": -1, "Capabilities": [["tpu"]]}]})).unwrap_err();
        assert!(e.starts_with("could not select device driver \"\" with capabilities"), "{e}");
        assert!(wanted(&json!({"DeviceRequests": [{"Driver": "amd", "Count": -1}]})).is_err());
        assert!(wanted(&json!({"DeviceRequests": [{"Driver": "cdi", "DeviceIDs": ["/dev/nvidia0"]}]})).unwrap_err().contains("invalid CDI device name"));
    }

    #[test]
    fn all_gpus_from_nvidias_spec() {
        let e = resolve(&specs(), &[Want::Gpus(-1)]).unwrap();
        assert_eq!(e.env, ["NVIDIA_VISIBLE_DEVICES=void"]);
        assert_eq!(
            e.mounts,
            [
                "/usr/lib/x86_64-linux-gnu/libcuda.so.580.95.05\t/usr/lib/x86_64-linux-gnu/libcuda.so.580.95.05\tro",
                "/usr/bin/nvidia-smi\t/usr/bin/nvidia-smi\tro",
            ]
        );
        assert_eq!(
            e.opts,
            [
                "device\t/dev/nvidia-modeset\t/dev/nvidia-modeset\trwm",
                "device\t/dev/nvidia-uvm\t/dev/nvidia-uvm\trwm",
                "device\t/dev/nvidia-uvm-tools\t/dev/nvidia-uvm-tools\trwm",
                "device\t/dev/nvidiactl\t/dev/nvidiactl\trwm",
                "cdi\tlink\tlibnvidia-allocator.so.580.95.05\t/usr/lib/x86_64-linux-gnu/gbm/nvidia-drm_gbm.so",
                "cdi\tcompat\t580.95.05",
                "cdi\tfolder\t/usr/lib/x86_64-linux-gnu",
                "cdi\tldconfig",
                "cdi\tnomodify",
                "device\t/dev/nvidia0\t/dev/nvidia0\trwm",
                "device\t/dev/nvidia1\t/dev/nvidia1\trwm",
            ]
        );
        assert!(e.skipped.is_empty());
    }

    #[test]
    fn some_gpus_by_count_index_and_uuid() {
        let one = resolve(&specs(), &[Want::Gpus(1)]).unwrap();
        assert!(one.opts.contains(&"device\t/dev/nvidia0\t/dev/nvidia0\trwm".to_owned()));
        assert!(!one.opts.contains(&"device\t/dev/nvidia1\t/dev/nvidia1\trwm".to_owned()));
        assert!(one.opts.contains(&"device\t/dev/dri/renderD128\t/dev/dri/renderD128\trwm".to_owned()));
        assert!(one.opts.contains(&"cdi\tlink\t../card1\t/dev/dri/by-path/pci-0000:21:00.0-card".to_owned()));
        assert!(one.opts.contains(&"cdi\tchmod\t755\t/dev/dri".to_owned()));
        let uuid = resolve(&specs(), &[Want::Named("nvidia.com/gpu=GPU-4a1b2c3d-0000-1111-2222-333344445555".into())]).unwrap();
        assert!(uuid.opts.contains(&"device\t/dev/nvidia1\t/dev/nvidia1\trwm".to_owned()));
        assert!(!uuid.opts.contains(&"device\t/dev/nvidia0\t/dev/nvidia0\trwm".to_owned()));
        // Both by index: the shared nodes and mounts once each.
        let both = resolve(&specs(), &[Want::Named("nvidia.com/gpu=0".into()), Want::Named("nvidia.com/gpu=1".into()), Want::Named("nvidia.com/gpu=0".into())]).unwrap();
        assert_eq!(both.opts.iter().filter(|l| l.as_str() == "device\t/dev/nvidiactl\t/dev/nvidiactl\trwm").count(), 1);
        assert_eq!(both.mounts.len(), 2);
        assert_eq!(resolve(&specs(), &[Want::Gpus(3)]).unwrap_err(), "could not select 3 GPUs: the CDI specs list 2");
    }

    #[test]
    fn missing_devices_and_specs_say_what_to_do() {
        let e = resolve(&specs(), &[Want::Named("nvidia.com/gpu=7".into())]).unwrap_err();
        assert!(e.contains("unresolvable CDI devices nvidia.com/gpu=7"), "{e}");
        let e = resolve(&[], &[Want::Gpus(-1)]).unwrap_err();
        assert!(e.contains("nvidia-ctk cdi generate"), "{e}");
    }

    #[test]
    fn a_later_directory_wins() {
        let mut later = specs()[0].clone();
        later.file = PathBuf::from("/var/run/cdi/nvidia.yaml");
        later.edits = json!({});
        later.devices = vec![("0".into(), json!({"deviceNodes": [{"path": "/dev/nvidia9"}]}))];
        let e = resolve(&[specs().remove(0), later], &[Want::Named("nvidia.com/gpu=0".into())]).unwrap();
        assert_eq!(e.opts, ["device\t/dev/nvidia9\t/dev/nvidia9\trwm"]);
    }

    #[test]
    fn hooks_old_and_new_and_foreign() {
        let mut e = Edits::default();
        let old = json!({"path": "/usr/bin/nvidia-ctk", "args": ["nvidia-ctk", "hook", "update-ldcache", "--folder=/usr/lib64", "--folder", "/usr/lib"]});
        hook(&old, &mut e).unwrap();
        assert_eq!(e.opts, ["cdi\tfolder\t/usr/lib64", "cdi\tfolder\t/usr/lib", "cdi\tldconfig"]);
        let future = json!({"path": "/usr/bin/nvidia-cdi-hook", "args": ["nvidia-cdi-hook", "something-new", "--x"]});
        hook(&future, &mut e).unwrap();
        assert_eq!(e.skipped, ["/usr/bin/nvidia-cdi-hook something-new --x"]);
        let other = json!({"path": "/opt/vendor/hook", "args": ["hook", "go"]});
        assert!(hook(&other, &mut e).unwrap_err().contains("is not supported"));
        let bad = json!({"path": "/usr/bin/nvidia-cdi-hook", "args": ["nvidia-cdi-hook", "create-symlinks", "--link", "a::relative"]});
        assert!(hook(&bad, &mut e).is_err());
    }

    #[test]
    fn mounts_and_nodes_are_checked() {
        let mut e = Edits::default();
        assert!(apply(&json!({"mounts": [{"hostPath": "x", "containerPath": "/x"}]}), &mut e).is_err());
        assert!(apply(&json!({"mounts": [{"hostPath": "/x", "containerPath": "/x", "type": "tmpfs"}]}), &mut e).is_err());
        assert!(apply(&json!({"deviceNodes": [{"path": "/dev/../etc/shadow"}]}), &mut e).is_err());
        assert!(apply(&json!({"deviceNodes": [{"path": "/dev/x", "permissions": "rwx"}]}), &mut e).is_err());
        apply(&json!({"deviceNodes": [{"path": "/dev/x", "hostPath": "/dev/y", "permissions": "rw"}], "additionalGids": [44]}), &mut e).unwrap();
        assert_eq!(e.opts, ["device\t/dev/y\t/dev/x\trw", "group\t44"]);
    }

    #[test]
    fn spec_env_replaces_the_images() {
        let env = vec!["PATH=/usr/bin".to_owned(), "NVIDIA_VISIBLE_DEVICES=all".to_owned()];
        assert_eq!(merge_env(&env, &["NVIDIA_VISIBLE_DEVICES=void".to_owned()]), ["PATH=/usr/bin", "NVIDIA_VISIBLE_DEVICES=void"]);
    }

    #[test]
    fn specs_load_from_directories_in_order() {
        let dir = std::env::temp_dir().join(format!("kiri-cdi-{}", std::process::id()));
        let (a, b) = (dir.join("etc"), dir.join("run"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        std::fs::write(a.join("nvidia.yaml"), NVIDIA).unwrap();
        std::fs::write(b.join("other.json"), r#"{"cdiVersion":"0.6.0","kind":"example.com/dev","devices":[{"name":"x","containerEdits":{"env":["X=1"]}}]}"#).unwrap();
        std::fs::write(b.join("notes.txt"), "not a spec").unwrap();
        let specs = load(&[a.as_path(), b.as_path()]).unwrap();
        assert_eq!(specs.iter().map(|s| s.kind.as_str()).collect::<Vec<_>>(), ["nvidia.com/gpu", "example.com/dev"]);
        std::fs::write(b.join("broken.json"), "{").unwrap();
        assert!(load(&[a.as_path(), b.as_path()]).unwrap_err().contains("broken.json"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
