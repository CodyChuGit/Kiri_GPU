//! The front door: every API call and UI request for a GPU runtime comes
//! through here, waits for its runtime if it must, and is passed through
//! to it, response streaming included. Also the status, events and admin
//! endpoints.

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full, Limited};
use hyper::body::{Body, Frame, Incoming};
use hyper::header::{self, HeaderMap, HeaderName, HeaderValue};
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio::time::{Instant, timeout};

use crate::config::{Class, Config, parse_http_url};
use crate::core::{Admit, HoldFor};
use crate::engine::{Engine, audit};
use crate::host::Host;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Resp = Response<BoxBody<Bytes, Error>>;

/// Who is asking.
#[derive(Debug, Clone)]
pub struct Caller {
    pub name: String,
    pub class: Class,
    pub admin: bool,
}

impl Caller {
    /// The admin socket: whoever can open it (file permissions decide).
    pub fn local_admin() -> Caller {
        Caller {
            name: "admin-socket".into(),
            class: Class::Interactive,
            admin: true,
        }
    }
}

#[derive(Debug)]
pub struct Front<H: Host> {
    pub engine: Arc<Engine<H>>,
    /// Each runtime's concurrency limit; a permit is held until the
    /// response body is done.
    slots: BTreeMap<String, Arc<Semaphore>>,
}

impl<H: Host> Front<H> {
    pub fn new(engine: Arc<Engine<H>>) -> Arc<Front<H>> {
        let cfg = engine.config();
        let slots = cfg
            .runtimes
            .iter()
            .map(|r| (r.name.clone(), Arc::new(Semaphore::new(r.concurrency))))
            .collect();
        Arc::new(Front { engine, slots })
    }

    fn cfg(&self) -> Arc<Config> {
        self.engine.config()
    }

    /// The caller from `Authorization: Bearer <token>`; an `X-Box-Class`
    /// header may lower its class, never raise it.
    pub fn authenticate(&self, headers: &HeaderMap) -> Option<Caller> {
        let token = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()))?;
        let hash = hex(&Sha256::digest(token.trim().as_bytes()));
        let cfg = self.cfg();
        let c = cfg
            .clients
            .iter()
            .find(|c| c.token_sha256.eq_ignore_ascii_case(&hash))?;
        let mut class = c.class;
        if let Some(asked) = headers
            .get("x-box-class")
            .and_then(|v| v.to_str().ok())
            .and_then(Class::parse)
            && asked > class
        {
            class = asked;
        }
        Some(Caller {
            name: c.name.clone(),
            class,
            admin: c.admin,
        })
    }

    pub async fn handle(
        self: Arc<Self>,
        req: Request<Incoming>,
        caller: Option<Caller>,
    ) -> Result<Resp, Infallible> {
        let Some(caller) = caller.or_else(|| self.authenticate(req.headers())) else {
            return Ok(error(
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                "a valid bearer token is required",
                0,
                None,
            ));
        };
        let path = req.uri().path().to_owned();
        let method = req.method().clone();
        let resp = match (method.clone(), path.as_str()) {
            (Method::GET, "/v1/models") => {
                json_resp(StatusCode::OK, self.engine.core().models(self.engine.now()))
            }
            (Method::GET, "/box/status") => json_resp(StatusCode::OK, self.engine.status()),
            (Method::GET, "/box/events") => self.events(),
            (Method::POST, p) if p.starts_with("/box/") => {
                self.admin(&caller, p, req.uri().query().unwrap_or(""))
            }
            (Method::POST, p) if p.starts_with("/v1/") => self.api(req, &caller).await,
            (_, p) if p.starts_with("/rt/") => self.ui(req, &caller).await,
            _ => error(
                StatusCode::NOT_FOUND,
                "not_found",
                &format!("no route {method} {path}"),
                0,
                None,
            ),
        };
        Ok(resp)
    }

    fn admin(&self, caller: &Caller, path: &str, query: &str) -> Resp {
        if !caller.admin {
            return error(
                StatusCode::FORBIDDEN,
                "permission_error",
                "this client may not change the scheduler",
                0,
                None,
            );
        }
        let now = self.engine.now();
        let minutes = query
            .split('&')
            .find_map(|kv| kv.strip_prefix("minutes="))
            .and_then(|m| m.parse::<u64>().ok());
        let parts: Vec<&str> = path.trim_start_matches("/box/").split('/').collect();
        let param = |k: &str| {
            query
                .split('&')
                .find_map(|kv| kv.strip_prefix(k).and_then(|v| v.strip_prefix('=')))
        };
        let force = param("force").is_some_and(|v| v == "1" || v == "true");
        // Scripts on the box name themselves (`--by NAME`), so their holds
        // can be told apart; a front-door client is its configured name.
        let by = match param("by").filter(|n| {
            !n.is_empty()
                && n.len() <= 64
                && n.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
        }) {
            Some(n) if caller.name == "admin-socket" => format!("admin-socket:{n}"),
            _ => caller.name.clone(),
        };
        let result: Result<Value, (u16, String)> = {
            let mut core = self.engine.core();
            match parts.as_slice() {
                // `use`: switch now and keep it for at least the dwell time
                // (or `minutes`) counted from when it is ready.
                ["use", rt] => {
                    let secs = minutes.map_or(core.config().sched.dwell_secs, |m| m * 60);
                    core.set_hold(now, rt, HoldFor::AfterReady(secs), &by, force)
                        .map(|()| {
                            let state = core.state_of(rt, now);
                            json!({
                                "use": rt,
                                "by": by,
                                "secs_after_ready": secs,
                                "state": state,
                                "note": if state == "ready" { "ready now" } else { "switching: the hold's time starts once it is ready" },
                            })
                        })
                }
                ["hold", rt] => {
                    let until = minutes.filter(|&m| m > 0).map(|m| now + m * 60_000);
                    core.set_hold(now, rt, HoldFor::Until(until), &by, force)
                        .map(|()| json!({"hold": rt, "by": by, "until": until, "state": core.state_of(rt, now)}))
                }
                ["release"] => core
                    .release(now, None, &by, force)
                    .map(|h| json!({"released": h})),
                ["release", rt] => core
                    .release(now, Some(rt), &by, force)
                    .map(|h| json!({"released": h})),
                ["unfence", name] => {
                    let rt = core.unfence(name);
                    drop(core);
                    Ok(json!({"unfenced": rt || self.engine.unfence_service(name)}))
                }
                ["reset"] => Ok(json!({"reset": core.reset()})),
                _ => Err((400, format!("no admin action {path}"))),
            }
        };
        match result {
            Ok(v) => {
                audit(
                    "admin",
                    json!({"by": by, "action": path, "query": query}),
                );
                self.engine.commit();
                json_resp(StatusCode::OK, v)
            }
            Err((status, e)) => {
                audit(
                    "admin-refused",
                    json!({"by": by, "action": path, "status": status, "reason": e}),
                );
                error(
                    StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_REQUEST),
                    "invalid_request_error",
                    &e,
                    0,
                    None,
                )
            }
        }
    }

    /// An OpenAI- or Anthropic-style call: the runtime comes from `"model"`.
    async fn api(&self, req: Request<Incoming>, caller: &Caller) -> Resp {
        let cfg = self.cfg();
        let limit = (cfg.sched.max_body_mib * 1024 * 1024) as usize;
        let (parts, body) = req.into_parts();
        let bytes = match Limited::new(body, limit).collect().await {
            Ok(b) => b.to_bytes(),
            Err(_) => {
                return error(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "invalid_request_error",
                    "request body too large or cut off",
                    0,
                    None,
                );
            }
        };
        let model = serde_json::from_slice::<Value>(&bytes)
            .ok()
            .and_then(|v| v["model"].as_str().map(str::to_owned));
        let Some(model) = model else {
            return error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "the body needs a \"model\"",
                0,
                None,
            );
        };
        let Some(rt) = cfg.runtime_for_model(&model).map(|r| r.name.clone()) else {
            return error(
                StatusCode::NOT_FOUND,
                "not_found_error",
                &format!("no runtime serves model {model}"),
                0,
                None,
            );
        };
        let req = Request::from_parts(parts, Full::new(bytes).map_err(|e| match e {}).boxed());
        self.forward(req, &rt, None, caller).await
    }

    /// A browser UI or other API under `/rt/<runtime>/…`.
    async fn ui(&self, mut req: Request<Incoming>, caller: &Caller) -> Resp {
        let path = req.uri().path().to_owned();
        let rest = &path["/rt/".len()..];
        let (rt, tail) = rest
            .split_once('/')
            .map_or((rest, "/"), |(r, t)| (r, &path[path.len() - t.len() - 1..]));
        let rt = rt.to_owned();
        if self.cfg().runtime(&rt).is_none() {
            return error(
                StatusCode::NOT_FOUND,
                "not_found_error",
                &format!("no runtime {rt}"),
                0,
                None,
            );
        }
        let tail = match req.uri().query() {
            Some(q) => format!("{tail}?{q}"),
            None => tail.to_owned(),
        };
        let upgrade = req
            .headers()
            .contains_key(header::UPGRADE)
            .then(|| hyper::upgrade::on(&mut req));
        let (parts, body) = req.into_parts();
        let req = Request::from_parts(parts, body.map_err(|e| -> Error { Box::new(e) }).boxed());
        self.forward(req, &rt, Some((tail, upgrade)), caller).await
    }

    /// Waits for `rt` to be ready and to have a free slot, then passes the
    /// request through.
    async fn forward(
        &self,
        req: Request<BoxBody<Bytes, Error>>,
        rt: &str,
        ui: Option<(String, Option<hyper::upgrade::OnUpgrade>)>,
        caller: &Caller,
    ) -> Resp {
        let cfg = self.cfg();
        let wait = Duration::from_secs(cfg.wait_secs(caller.class));
        let deadline = Instant::now() + wait;
        let slot = self
            .slots
            .get(rt)
            .cloned()
            .expect("every runtime has slots");
        let (permit, guard) = loop {
            if let Err(resp) = self.wait_ready(rt, caller, deadline).await {
                return resp;
            }
            let permit = match timeout(
                deadline.saturating_duration_since(Instant::now()),
                slot.clone().acquire_owned(),
            )
            .await
            {
                Ok(Ok(p)) => p,
                _ => {
                    return error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "overloaded_error",
                        &format!("{rt} is busy with other requests"),
                        5,
                        Some(("ready", None)),
                    );
                }
            };
            // Ready may have ended while waiting for the slot (a drain).
            let mut core = self.engine.core();
            if core.is_ready(rt) {
                core.begin(rt);
                break (
                    permit,
                    Inflight {
                        engine: self.engine.clone(),
                        rt: rt.to_owned(),
                    },
                );
            }
        };
        let r = cfg.runtime(rt).expect("checked").clone();
        let (path, upgrade) = match ui {
            Some((p, u)) => (p, u),
            None => (
                req.uri()
                    .path_and_query()
                    .map_or("/".to_owned(), |p| p.to_string()),
                None,
            ),
        };
        match proxy(&r.upstream, &path, req, upgrade.is_some()).await {
            Ok(mut resp) => {
                if resp.status() == StatusCode::SWITCHING_PROTOCOLS
                    && let Some(client) = upgrade
                {
                    let upstream = hyper::upgrade::on(&mut resp);
                    tokio::spawn(async move {
                        let _hold = (permit, guard);
                        if let (Ok(a), Ok(b)) = (client.await, upstream.await) {
                            let _ = tokio::io::copy_bidirectional(
                                &mut TokioIo::new(a),
                                &mut TokioIo::new(b),
                            )
                            .await;
                        }
                    });
                    let (parts, _) = resp.into_parts();
                    return Response::from_parts(parts, empty());
                }
                let (mut parts, body) = resp.into_parts();
                strip_hop(&mut parts.headers, false);
                parts
                    .headers
                    .insert("x-box-state", HeaderValue::from_static("ready"));
                Response::from_parts(
                    parts,
                    Tracked {
                        inner: body.map_err(|e| -> Error { Box::new(e) }).boxed(),
                        _permit: permit,
                        _guard: guard,
                    }
                    .boxed(),
                )
            }
            Err(e) => {
                audit("upstream-error", json!({"runtime": rt, "error": e}));
                error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    &format!("{rt} did not answer: {e}"),
                    5,
                    Some(("ready", None)),
                )
            }
        }
    }

    /// Admission: straight through when ready, else wait (holding the
    /// connection) until ready or until refused.
    async fn wait_ready(&self, rt: &str, caller: &Caller, deadline: Instant) -> Result<(), Resp> {
        let mut changes = self.engine.subscribe();
        let admit = self
            .engine
            .core()
            .admit(self.engine.now(), rt, caller.class);
        let id = match admit {
            Admit::Go => return Ok(()),
            Admit::Refuse {
                status,
                reason,
                retry_after,
            } => return Err(self.busy(status, &reason, retry_after, None, caller, rt)),
            Admit::Wait(id) => id,
        };
        // Wakes the main loop, which may start a switch for this request.
        self.engine.notify();
        let mut waiter = Waiter {
            engine: self.engine.clone(),
            id: Some(id),
        };
        loop {
            let answer = self.engine.core().poll(self.engine.now(), id);
            match answer {
                Some(Admit::Go) => {
                    waiter.id = None;
                    return Ok(());
                }
                Some(Admit::Refuse {
                    status,
                    reason,
                    retry_after,
                }) => {
                    waiter.id = None;
                    return Err(self.busy(status, &reason, retry_after, None, caller, rt));
                }
                Some(Admit::Wait(_)) | None => {}
            }
            if Instant::now() >= deadline {
                let pos = self.engine.core().position(id);
                return Err(self.busy(
                    503,
                    &format!("still waiting for {rt}"),
                    30,
                    pos,
                    caller,
                    rt,
                ));
            }
            let _ = timeout(
                Duration::from_secs(1).min(deadline.saturating_duration_since(Instant::now())),
                changes.changed(),
            )
            .await;
        }
    }

    fn busy(
        &self,
        status: u16,
        reason: &str,
        retry_after: u64,
        position: Option<usize>,
        caller: &Caller,
        rt: &str,
    ) -> Resp {
        audit(
            "refused",
            json!({"client": caller.name, "runtime": rt, "status": status, "reason": reason}),
        );
        let phase = self.engine.core().phase_name();
        let status = StatusCode::from_u16(status).unwrap_or(StatusCode::SERVICE_UNAVAILABLE);
        let kind = if status == StatusCode::TOO_MANY_REQUESTS {
            "rate_limit_error"
        } else {
            "overloaded_error"
        };
        error(status, kind, reason, retry_after, Some((phase, position)))
    }

    /// Server-sent events: the status, at most once a second, on change.
    fn events(&self) -> Resp {
        let (tx, rx) = mpsc::channel::<Bytes>(4);
        let engine = self.engine.clone();
        tokio::spawn(async move {
            let mut changes = engine.subscribe();
            loop {
                let line = format!("data: {}\n\n", engine.status());
                if tx.send(Bytes::from(line)).await.is_err() {
                    return;
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
                // A comment every 15 s keeps idle connections open.
                if timeout(Duration::from_secs(15), changes.changed())
                    .await
                    .is_err()
                    && tx
                        .send(Bytes::from_static(b": keep-alive\n\n"))
                        .await
                        .is_err()
                {
                    return;
                }
            }
        });
        let mut resp = Response::new(Channel(rx).boxed());
        resp.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        resp.headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        resp
    }
}

/// Ends an in-flight request's count when its response is done.
#[derive(Debug)]
struct Inflight<H: Host> {
    engine: Arc<Engine<H>>,
    rt: String,
}

impl<H: Host> Drop for Inflight<H> {
    fn drop(&mut self) {
        self.engine.core().end(&self.rt);
        self.engine.notify();
    }
}

/// A waiting request that goes away (client hung up) leaves the line.
struct Waiter<H: Host> {
    engine: Arc<Engine<H>>,
    id: Option<u64>,
}

impl<H: Host> Drop for Waiter<H> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            self.engine.core().forget(id);
        }
    }
}

/// A response body that holds its slot and in-flight count until done.
struct Tracked<H: Host> {
    inner: BoxBody<Bytes, Error>,
    _permit: OwnedSemaphorePermit,
    _guard: Inflight<H>,
}

impl<H: Host> Body for Tracked<H> {
    type Data = Bytes;
    type Error = Error;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Error>>> {
        Pin::new(&mut self.inner).poll_frame(cx)
    }
    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> hyper::body::SizeHint {
        self.inner.size_hint()
    }
}

struct Channel(mpsc::Receiver<Bytes>);

impl Body for Channel {
    type Data = Bytes;
    type Error = Error;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Error>>> {
        self.0.poll_recv(cx).map(|o| o.map(|b| Ok(Frame::data(b))))
    }
}

fn empty() -> BoxBody<Bytes, Error> {
    Full::new(Bytes::new()).map_err(|e| match e {}).boxed()
}

fn json_resp(status: StatusCode, v: Value) -> Resp {
    let mut r = Response::new(
        Full::new(Bytes::from(v.to_string()))
            .map_err(|e| match e {})
            .boxed(),
    );
    *r.status_mut() = status;
    r.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    r
}

/// An error in the shape both OpenAI and Anthropic clients read, with the
/// box's state alongside and `Retry-After` when retrying makes sense.
fn error(
    status: StatusCode,
    kind: &str,
    message: &str,
    retry_after: u64,
    state: Option<(&str, Option<usize>)>,
) -> Resp {
    let mut body = json!({"type": "error", "error": {"type": kind, "message": message}});
    if let Some((phase, pos)) = state {
        body["box"] = json!({"state": phase, "queue_position": pos, "retry_after": retry_after});
    }
    let mut r = json_resp(status, body);
    if retry_after > 0 {
        r.headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from(retry_after));
    }
    if let Some((phase, pos)) = state {
        if let Ok(v) = HeaderValue::from_str(phase) {
            r.headers_mut().insert("x-box-state", v);
        }
        if let Some(p) = pos {
            r.headers_mut()
                .insert("x-box-queue-position", HeaderValue::from(p));
        }
    }
    r
}

const HOP: [&str; 8] = [
    "connection",
    "keep-alive",
    "proxy-connection",
    "transfer-encoding",
    "te",
    "trailer",
    "upgrade",
    "proxy-authorization",
];

fn strip_hop(h: &mut HeaderMap, keep_upgrade: bool) {
    let named: Vec<HeaderName> = h
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .filter_map(|n| HeaderName::from_bytes(n.trim().as_bytes()).ok())
        .collect();
    for n in HOP {
        if keep_upgrade && (n == "connection" || n == "upgrade") {
            continue;
        }
        h.remove(n);
    }
    if !keep_upgrade {
        for n in named {
            h.remove(n);
        }
    }
}

/// Passes one request to a runtime over a fresh HTTP/1.1 connection. The
/// client's token never reaches the runtime.
async fn proxy(
    upstream: &str,
    path: &str,
    req: Request<BoxBody<Bytes, Error>>,
    upgrade: bool,
) -> Result<Response<Incoming>, String> {
    let (host, port, prefix) = parse_http_url(upstream)?;
    let stream = timeout(
        Duration::from_secs(5),
        tokio::net::TcpStream::connect((host.as_str(), port)),
    )
    .await
    .map_err(|_| "connect timed out".to_owned())?
    .map_err(|e| e.to_string())?;
    let _ = stream.set_nodelay(true);
    let (mut send, conn) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
        .await
        .map_err(|e| e.to_string())?;
    tokio::spawn(conn.with_upgrades());
    let (mut parts, body) = req.into_parts();
    let full = format!("{}{}", prefix.trim_end_matches('/'), path);
    parts.uri = full.parse().map_err(|_| format!("bad path {full}"))?;
    strip_hop(&mut parts.headers, upgrade);
    for h in ["authorization", "x-api-key", "x-box-class", "cookie-box"] {
        parts.headers.remove(h);
    }
    parts.headers.insert(
        header::HOST,
        HeaderValue::from_str(&format!("{host}:{port}")).map_err(|e| e.to_string())?,
    );
    let fut = send.send_request(Request::from_parts(parts, body));
    // Headers must come within 5 minutes (a long prompt read); the body may
    // stream for as long as the runtime keeps sending.
    timeout(Duration::from_secs(300), fut)
        .await
        .map_err(|_| "no response headers within 300 s".to_owned())?
        .map_err(|e| e.to_string())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Serves the front door on TCP; each client authenticates with a token.
pub async fn serve_tcp<H: Host>(front: Arc<Front<H>>, listener: tokio::net::TcpListener) {
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            continue;
        };
        let _ = stream.set_nodelay(true);
        let front = front.clone();
        tokio::spawn(async move {
            let svc = hyper::service::service_fn(move |req| front.clone().handle(req, None));
            let _ = hyper::server::conn::http1::Builder::new()
                .serve_connection(TokioIo::new(stream), svc)
                .with_upgrades()
                .await;
        });
    }
}

/// Serves the same routes on the admin socket, as an admin.
#[cfg(unix)]
pub async fn serve_unix<H: Host>(front: Arc<Front<H>>, listener: tokio::net::UnixListener) {
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            continue;
        };
        let front = front.clone();
        tokio::spawn(async move {
            let svc = hyper::service::service_fn(move |req| {
                front.clone().handle(req, Some(Caller::local_admin()))
            });
            let _ = hyper::server::conn::http1::Builder::new()
                .serve_connection(TokioIo::new(stream), svc)
                .with_upgrades()
                .await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EXAMPLE;
    use crate::engine::Health;
    use crate::host::fake::Fake;

    /// A fake runtime API: echoes the path and whether a token leaked.
    async fn upstream() -> u16 {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let (s, _) = l.accept().await.unwrap();
                tokio::spawn(async move {
                    let svc = hyper::service::service_fn(|req: Request<Incoming>| async move {
                        let leaked = req.headers().contains_key("authorization");
                        let path = req.uri().to_string();
                        let body = req.into_body().collect().await.unwrap().to_bytes();
                        Ok::<_, Infallible>(Response::new(Full::new(Bytes::from(
                            json!({"path": path, "leaked": leaked, "len": body.len()}).to_string(),
                        ))))
                    });
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(TokioIo::new(s), svc)
                        .await;
                });
            }
        });
        port
    }

    async fn setup() -> (Arc<Front<Fake>>, u16) {
        let maya = upstream().await;
        let comfy = upstream().await;
        let text = EXAMPLE
            .replace("http://127.0.0.1:8080", &format!("http://127.0.0.1:{maya}"))
            .replace(
                "http://127.0.0.1:8188",
                &format!("http://127.0.0.1:{comfy}"),
            );
        let cfg = Arc::new(Config::parse(&text).unwrap());
        let fake = Fake::with(|s| {
            s.url_stack
                .insert(format!("http://127.0.0.1:{maya}"), "maya".into());
            s.url_stack.insert(
                format!("http://127.0.0.1:{comfy}/system_stats"),
                "comfyui".into(),
            );
        });
        let engine = Engine::new(cfg, fake, None);
        let front = Front::new(engine.clone());
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        tokio::spawn(serve_tcp(front.clone(), l));
        tokio::spawn(async move {
            let mut h = Health::default();
            loop {
                engine.step(&mut h).await;
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        });
        (front, port)
    }

    async fn call(
        port: u16,
        method: &str,
        path: &str,
        token: &str,
        body: &str,
    ) -> (u16, HeaderMap, Value) {
        let s = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .unwrap();
        let (mut send, conn) = hyper::client::conn::http1::handshake(TokioIo::new(s))
            .await
            .unwrap();
        tokio::spawn(conn);
        let req = Request::builder()
            .method(method)
            .uri(path)
            .header("host", "x")
            .header("authorization", format!("Bearer {token}"))
            .body(Full::new(Bytes::from(body.to_owned())))
            .unwrap();
        let resp = send.send_request(req).await.unwrap();
        let status = resp.status().as_u16();
        let headers = resp.headers().clone();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (
            status,
            headers,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    #[tokio::test]
    async fn end_to_end() {
        let (_front, port) = setup().await;
        // No or wrong token.
        assert_eq!(call(port, "GET", "/box/status", "nope", "").await.0, 401);
        // "test" is the interactive admin client in the example config.
        let (st, _, v) = call(
            port,
            "POST",
            "/v1/chat/completions",
            "test",
            r#"{"model":"glm-5.3-flash","messages":[]}"#,
        )
        .await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(v["path"], "/v1/chat/completions");
        assert_eq!(
            v["leaked"], false,
            "the client token must not reach the runtime"
        );
        // A UI path on another runtime: the box switches (maya is idle).
        let (st, _, v) = call(port, "GET", "/rt/comfyui/api/queue?x=1", "test", "").await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(v["path"], "/api/queue?x=1");
        let (_, _, status) = call(port, "GET", "/box/status", "test", "").await;
        assert_eq!(status["phase"]["rt"], "comfyui");
        // Unknown model, and a hold that refuses with Retry-After.
        assert_eq!(
            call(
                port,
                "POST",
                "/v1/chat/completions",
                "test",
                r#"{"model":"nope"}"#
            )
            .await
            .0,
            404
        );
        assert_eq!(
            call(port, "POST", "/box/hold/comfyui?minutes=10", "test", "")
                .await
                .0,
            200
        );
        let (st, h, v) = call(
            port,
            "POST",
            "/v1/chat/completions",
            "test",
            r#"{"model":"glm-5.3-flash"}"#,
        )
        .await;
        assert_eq!(st, 503);
        assert!(h.get("retry-after").is_some());
        assert_eq!(v["error"]["type"], "overloaded_error");
        let (_, _, models) = call(port, "GET", "/v1/models", "test", "").await;
        assert_eq!(models["data"][0]["id"], "glm-5.3-flash");
        assert_eq!(models["data"][0]["status"], "stopped");
    }

    #[tokio::test]
    async fn non_admins_cannot_steer() {
        let (_front, port) = setup().await;
        // "hello" is the agent client in the example config: not an admin.
        assert_eq!(
            call(port, "POST", "/box/hold/comfyui", "hello", "").await.0,
            403
        );
        assert_eq!(call(port, "POST", "/box/release", "hello", "").await.0, 403);
        assert_eq!(call(port, "GET", "/box/status", "hello", "").await.0, 200);
    }

    #[test]
    fn class_can_only_be_lowered() {
        let cfg = Arc::new(Config::parse(EXAMPLE).unwrap());
        let front = Front::new(Engine::new(cfg, Fake::default(), None));
        let mut h = HeaderMap::new();
        h.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer test"),
        );
        assert_eq!(front.authenticate(&h).unwrap().class, Class::Interactive);
        h.insert("x-box-class", HeaderValue::from_static("batch"));
        assert_eq!(front.authenticate(&h).unwrap().class, Class::Batch);
        // An agent cannot raise itself to interactive.
        let mut h = HeaderMap::new();
        h.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer hello"),
        );
        h.insert("x-box-class", HeaderValue::from_static("interactive"));
        assert_eq!(front.authenticate(&h).unwrap().class, Class::Agent);
    }

    #[test]
    fn hop_headers_go() {
        let mut h = HeaderMap::new();
        h.insert(
            header::CONNECTION,
            HeaderValue::from_static("keep-alive, x-secret"),
        );
        h.insert("x-secret", HeaderValue::from_static("1"));
        h.insert("x-fine", HeaderValue::from_static("1"));
        strip_hop(&mut h, false);
        assert!(h.get("x-secret").is_none() && h.get(header::CONNECTION).is_none());
        assert!(h.get("x-fine").is_some());
    }
}
