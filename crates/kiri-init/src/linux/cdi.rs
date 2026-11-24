//! What a CDI spec's hooks ask for, done here rather than by running them:
//! there is no runc, and so no OCI hook runner. The daemon turns NVIDIA's
//! hooks (`nvidia-cdi-hook create-symlinks`, `update-ldcache`,
//! `enable-cuda-compat`, `chmod`, `disable-device-node-modification`) into
//! `cdi` lines (see `kiri::daemon::cdi`); [`apply`] carries them out in the
//! container after pivot_root, before it execs, while it is still root
//! with every capability and its root is still writable. Paths are then
//! the container's own, so nothing here can reach the host's files.

use std::ffi::CString;
use std::fs;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

/// The `cdi` lines of a start.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Cdi {
    /// (target, link): `link` is made as a symlink to `target`.
    pub links: Vec<(String, String)>,
    /// Folders for the ld.so cache, and whether to rebuild it.
    pub folders: Vec<String>,
    pub ldconfig: bool,
    /// The host driver's version, for CUDA forward compatibility.
    pub compat: Option<String>,
    pub chmods: Vec<(u32, String)>,
    /// Keep the driver from making or changing device nodes in the container.
    pub no_modify: bool,
}

impl Cdi {
    pub fn is_empty(&self) -> bool {
        *self == Cdi::default()
    }

    /// One `cdi` line's fields after `cdi`, or None if it is malformed.
    pub fn add(&mut self, f: &[&str]) -> Option<()> {
        let abs = |p: &str| p.starts_with('/') && !p.split('/').any(|c| c == "..");
        match f {
            ["link", target, link] if !target.is_empty() && abs(link) => {
                self.links.push(((*target).to_owned(), (*link).to_owned()))
            }
            ["folder", dir] if abs(dir) => self.folders.push((*dir).to_owned()),
            ["ldconfig"] => self.ldconfig = true,
            ["compat", v]
                if !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit() || b == b'.') =>
            {
                self.compat = Some((*v).to_owned())
            }
            ["chmod", mode, path] if abs(path) => self.chmods.push((
                u32::from_str_radix(mode, 8).ok().filter(|m| *m <= 0o7777)?,
                (*path).to_owned(),
            )),
            ["nomodify"] => self.no_modify = true,
            _ => return None,
        }
        Some(())
    }
}

/// Where CUDA images keep the forward-compatibility driver.
const COMPAT_DIR: &str = "/usr/local/cuda/compat";

/// ld.so.conf.d files, named to sort before the image's own: the compat
/// driver first, so it wins over the host's libcuda when it is newer.
const COMPAT_CONF: &str = "/etc/ld.so.conf.d/00-kiri-0-cuda-compat.conf";
const CDI_CONF: &str = "/etc/ld.so.conf.d/00-kiri-1-cdi.conf";

/// The ldconfig to run, the container's own: Debian and Ubuntu's
/// /sbin/ldconfig is a shell script around ldconfig.real.
const LDCONFIG: [&str; 3] = [
    "/sbin/ldconfig.real",
    "/sbin/ldconfig",
    "/usr/sbin/ldconfig",
];

/// Carries out `c` in the container (after pivot_root).
pub fn apply(c: &Cdi) -> io::Result<()> {
    for (mode, path) in &c.chmods {
        match fs::set_permissions(path, fs::Permissions::from_mode(*mode)) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => {
                return Err(io::Error::new(e.kind(), format!("chmod {path}: {e}")));
            }
            _ => {}
        }
    }
    for (target, link) in &c.links {
        // As NVIDIA's hook: what is there already (the image's own link, or
        // a library) is kept.
        if fs::symlink_metadata(link).is_ok() {
            continue;
        }
        if let Some(parent) = Path::new(link).parent() {
            fs::create_dir_all(parent)
                .map_err(|e| io::Error::new(e.kind(), format!("symlink {link}: {e}")))?;
        }
        std::os::unix::fs::symlink(target, link)
            .map_err(|e| io::Error::new(e.kind(), format!("symlink {link} -> {target}: {e}")))?;
    }
    let compat = c
        .compat
        .as_deref()
        .filter(|host| compat_wanted(host, &compat_versions(Path::new(COMPAT_DIR))));
    if compat.is_some() || !c.folders.is_empty() {
        fs::create_dir_all("/etc/ld.so.conf.d")?;
    }
    if compat.is_some() {
        fs::write(COMPAT_CONF, format!("{COMPAT_DIR}\n"))?;
    }
    if !c.folders.is_empty() {
        fs::write(
            CDI_CONF,
            c.folders
                .iter()
                .map(|f| format!("{f}\n"))
                .collect::<String>(),
        )?;
    }
    if c.ldconfig || compat.is_some() {
        run_ldconfig()?;
    }
    Ok(())
}

/// Versions of the compat driver's libcuda (`libcuda.so.580.95.05`) in `dir`.
fn compat_versions(dir: &Path) -> Vec<Vec<u32>> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|e| {
            e.file_name()
                .to_str()
                .and_then(|n| n.strip_prefix("libcuda.so."))
                .and_then(version)
        })
        .filter(|v| v.len() > 1)
        .collect()
}

fn version(s: &str) -> Option<Vec<u32>> {
    s.split('.').map(|p| p.parse().ok()).collect()
}

/// Whether the image's compat driver should be used: as NVIDIA's hook
/// decides, only when its major version is newer than the host driver's.
fn compat_wanted(host: &str, compat: &[Vec<u32>]) -> bool {
    let Some(host) = version(host) else {
        return false;
    };
    compat.iter().any(|v| v[0] > host[0])
}

/// Rebuilds the container's ld.so cache, as `update-ldcache` does. An image
/// without ldconfig (musl, distroless) has no cache to update.
fn run_ldconfig() -> io::Result<()> {
    let Some(path) = LDCONFIG
        .iter()
        .find(|p| fs::metadata(p).is_ok_and(|m| m.is_file()))
    else {
        return Ok(());
    };
    let prog = CString::new(*path).expect("no NUL");
    // SAFETY: fork in the single-threaded container child; the new child
    // only calls async-signal-safe functions before execve or _exit.
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return Err(io::Error::last_os_error());
    }
    if pid == 0 {
        let argv = [prog.as_ptr(), std::ptr::null()];
        let envp = [std::ptr::null()];
        // SAFETY: valid NUL-terminated program and NULL-terminated vectors.
        unsafe {
            libc::execve(prog.as_ptr(), argv.as_ptr(), envp.as_ptr());
            libc::_exit(127);
        }
    }
    let mut status = 0;
    // SAFETY: waiting for the child just forked.
    while unsafe { libc::waitpid(pid, &mut status, 0) } < 0 {
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e);
        }
    }
    if libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0 {
        Ok(())
    } else {
        Err(io::Error::other(format!("{path} failed (status {status})")))
    }
}

/// The NVIDIA driver's parameters with `ModifyDeviceFiles: 0`, for binding
/// over /proc/driver/nvidia/params, as `disable-device-node-modification`
/// does; None if the host has no NVIDIA driver loaded.
pub fn nvidia_params(host: &str) -> Option<String> {
    if !host.lines().any(|l| l.starts_with("ModifyDeviceFiles:")) {
        return None;
    }
    Some(
        host.lines()
            .map(|l| {
                if l.starts_with("ModifyDeviceFiles:") {
                    "ModifyDeviceFiles: 0\n".to_owned()
                } else {
                    format!("{l}\n")
                }
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_parse() {
        let mut c = Cdi::default();
        assert!(c.is_empty());
        for f in [
            &[
                "link",
                "libcuda.so.1",
                "/usr/lib/x86_64-linux-gnu/libcuda.so",
            ][..],
            &["folder", "/usr/lib/x86_64-linux-gnu"],
            &["ldconfig"],
            &["compat", "580.95.05"],
            &["chmod", "755", "/dev/dri"],
            &["nomodify"],
        ] {
            assert_eq!(c.add(f), Some(()), "{f:?}");
        }
        assert_eq!(
            c.links,
            [(
                "libcuda.so.1".to_owned(),
                "/usr/lib/x86_64-linux-gnu/libcuda.so".to_owned()
            )]
        );
        assert_eq!(c.chmods, [(0o755, "/dev/dri".to_owned())]);
        assert!(c.ldconfig && c.no_modify && c.compat.as_deref() == Some("580.95.05"));
        for bad in [
            &["link", "x", "relative"][..],
            &["folder", "/a/../b"],
            &["chmod", "999", "/x"],
            &["compat", "abc"],
            &["what"],
        ] {
            assert_eq!(c.add(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn the_compat_driver_only_when_it_is_newer() {
        assert!(compat_wanted("570.86.15", &[vec![580, 95, 5]]));
        assert!(!compat_wanted("580.95.05", &[vec![580, 95, 5]]));
        assert!(!compat_wanted("580.65.06", &[vec![580, 95, 5]]));
        assert!(!compat_wanted("590.10", &[vec![580, 95, 5]]));
        assert!(!compat_wanted("580.95.05", &[]));
        let dir = std::env::temp_dir().join(format!("kiri-compat-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        for f in [
            "libcuda.so",
            "libcuda.so.1",
            "libcuda.so.580.95.05",
            "libnvidia-ptxjitcompiler.so.580.95.05",
        ] {
            fs::write(dir.join(f), "").unwrap();
        }
        assert_eq!(compat_versions(&dir), [vec![580, 95, 5]]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn params_keep_the_driver_from_touching_device_nodes() {
        let host = "ResmanDebugLevel: 4294967295\nModifyDeviceFiles: 1\nDeviceFileUID: 0\n";
        assert_eq!(
            nvidia_params(host).unwrap(),
            "ResmanDebugLevel: 4294967295\nModifyDeviceFiles: 0\nDeviceFileUID: 0\n"
        );
        assert_eq!(nvidia_params(""), None);
    }
}
