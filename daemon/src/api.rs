//! The daemon API: localhost only, JSON over HTTP, SSE for streams.
//!
//! One thread per connection; anything that needs the kernel goes through
//! the worker channel, everything else (journal, store, settings, status
//! snapshot) is served directly off the shared state.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use serde_json::{json, Value};

use aios::http::find_subslice;

use crate::state::{now_ms, now_ts, Req, Settings, Shared, MODE_INCOGNITO, MODE_PAUSED, MODE_PERSISTENT};

pub fn handle(mut stream: TcpStream, shared: Arc<Shared>) {
    let Some((method, raw_path, body)) = read_request(&mut stream) else { return };
    let (path, query) = split_query(&raw_path);

    if method == "OPTIONS" {
        let _ = write!(
            stream,
            "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n\
             Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n\
             Access-Control-Allow-Headers: Content-Type\r\nContent-Length: 0\r\n\r\n"
        );
        return;
    }

    // The turn endpoint streams; it writes its own response.
    if method == "POST" && path == "/v1/turn" {
        turn_sse(&mut stream, &shared, &body);
        return;
    }

    let (status, out) = match (method.as_str(), path.as_str()) {
        ("POST", "/v1/turn/cancel") => cancel_turn(&shared, &body),
        ("GET", "/v1/timeline") => timeline(&shared, &query),
        ("GET", "/v1/memory/search") => memory_search(&shared, &query),
        ("GET", "/v1/memory/browse") => memory_browse(&shared),
        ("POST", "/v1/memory/correct") => memory_correct(&shared, &body),
        ("POST", "/v1/memory/delete") => memory_delete(&shared, &body),
        ("GET", "/v1/status") => ("200 OK", shared.status.lock().unwrap().clone()),
        ("GET", "/v1/models") => ("200 OK", models(&shared)),
        ("GET", "/v1/settings") => settings_get(&shared),
        ("PUT", "/v1/settings") => settings_put(&shared, &body),
        ("GET", "/v1/digest") => ("200 OK", digest(&shared)),
        ("POST", "/v1/kv/save") => kv(&shared, true),
        ("POST", "/v1/kv/restore") => kv(&shared, false),
        ("GET", p) if p.starts_with("/v1/media/") => return serve_media(&mut stream, &shared, p),
        ("GET", _) => return serve_static(&mut stream, &shared, &path),
        _ => ("404 Not Found", json!({"error": "not found"})),
    };
    respond_json(&mut stream, status, &out);
}

fn respond_json(stream: &mut TcpStream, status: &str, v: &Value) {
    let out = v.to_string();
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{out}",
        out.len()
    );
}

// --- /v1/turn ------------------------------------------------------------------

fn turn_sse(stream: &mut TcpStream, shared: &Arc<Shared>, body: &str) {
    let parsed = serde_json::from_str::<Value>(body).unwrap_or(Value::Null);
    let mut text = parsed["text"].as_str().map(|s| s.trim().to_string()).unwrap_or_default();
    let images: Vec<String> = parsed["images"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    if text.is_empty() && images.is_empty() {
        respond_json(stream, "400 Bad Request", &json!({"error": "empty text"}));
        return;
    }
    if text.is_empty() {
        text = "(shared an image)".into();
    }
    let image_files = save_media(shared, &images);

    let id = shared.turn_counter.fetch_add(1, Ordering::SeqCst) + 1;
    let cancel = Arc::new(AtomicBool::new(false));
    shared.cancels.lock().unwrap().insert(id, cancel.clone());

    let (etx, erx) = mpsc::channel::<Value>();
    if shared.tx.send(Req::Turn { id, text, images, image_files, cancel: cancel.clone(), events: etx }).is_err() {
        respond_json(stream, "500 Internal Server Error", &json!({"error": "worker gone"}));
        return;
    }

    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
         Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
    );
    while let Ok(event) = erx.recv() {
        let done = event["t"] == "done";
        if write!(stream, "data: {event}\n\n").is_err() || stream.flush().is_err() {
            // Client went away: that's a cancel.
            cancel.store(true, Ordering::Relaxed);
            break;
        }
        if done {
            break;
        }
    }
    shared.cancels.lock().unwrap().remove(&id);
}

/// Persist attached data:-URL images under ~/.aios/media; returns filenames.
fn save_media(shared: &Arc<Shared>, images: &[String]) -> Vec<String> {
    use base64::Engine;
    let mut out = Vec::new();
    let dir = shared.dirs.media_dir();
    for (i, data_url) in images.iter().enumerate().take(6) {
        let Some((head, b64)) = data_url.split_once(";base64,") else { continue };
        let ext = match head.strip_prefix("data:").unwrap_or("") {
            "image/jpeg" => "jpg",
            "image/webp" => "webp",
            "image/gif" => "gif",
            _ => "png",
        };
        let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) else { continue };
        let name = format!("{}_{i}.{ext}", now_ms());
        if std::fs::write(dir.join(&name), bytes).is_ok() {
            out.push(name);
        }
    }
    out
}

fn cancel_turn(shared: &Arc<Shared>, body: &str) -> (&'static str, Value) {
    let id = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v["turn_id"].as_u64())
        .unwrap_or(0);
    match shared.cancels.lock().unwrap().get(&id) {
        Some(flag) => {
            flag.store(true, Ordering::Relaxed);
            ("200 OK", json!({"cancelled": id}))
        }
        None => ("404 Not Found", json!({"error": "no such active turn"})),
    }
}

// --- Timeline & digest -------------------------------------------------------------

fn timeline(shared: &Arc<Shared>, query: &HashMap<String, String>) -> (&'static str, Value) {
    let before = query.get("before").and_then(|v| v.parse().ok()).unwrap_or(0u64);
    let limit = query.get("limit").and_then(|v| v.parse().ok()).unwrap_or(50usize).min(500);
    let j = shared.journal.lock().unwrap();
    let entries: Vec<Value> = j
        .page(before, limit)
        .into_iter()
        .map(|e| serde_json::to_value(e).unwrap_or(Value::Null))
        .collect();
    ("200 OK", json!({"entries": entries}))
}

/// A short daemon-composed digest for first render after boot. Template
/// only, no model call: greeting, recent topics from memory events, and an
/// opening question. The first conversation IS onboarding.
fn digest(shared: &Arc<Shared>) -> Value {
    let hour = ((now_ms() / 3_600_000 + local_utc_offset_hours()) % 24) as i64;
    let greeting = match hour {
        5..=11 => "Good morning.",
        12..=17 => "Good afternoon.",
        _ => "Good evening.",
    };

    let j = shared.journal.lock().unwrap();
    if j.len() == 0 {
        return json!({
            "text": format!(
                "{greeting}\nI'm your companion — one continuous conversation, one memory.\n\
                 I don't know you yet. Tell me who you are and what you're working on, \
                 and I'll remember."
            ),
            "fresh": true,
        });
    }

    let mut topics: Vec<String> = Vec::new();
    let mut last_activity_ms = 0u64;
    for e in j.recent(200) {
        if e.kind == "user" || e.kind == "assistant" {
            last_activity_ms = last_activity_ms.max(e.ts_ms);
        }
        if e.kind == "memory" {
            if let Some(branch) = e.meta.get("branch").and_then(|b| b.as_str()) {
                if !branch.is_empty() && !topics.iter().any(|t| t == branch) {
                    topics.push(branch.to_string());
                }
            }
        }
    }
    drop(j);

    let mut lines = vec![greeting.to_string()];
    if last_activity_ms > 0 {
        let days = (now_ms().saturating_sub(last_activity_ms)) / 86_400_000;
        let when = match days {
            0 => "earlier today".to_string(),
            1 => "yesterday".to_string(),
            n => format!("{n} days ago"),
        };
        lines.push(format!("We last talked {when}."));
    }
    let recent: Vec<&String> = topics.iter().rev().take(3).collect();
    if !recent.is_empty() {
        lines.push(format!(
            "Recently on our plate: {}.",
            recent.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
        ));
    }
    lines.push("What would you like to work on?".to_string());
    json!({"text": lines.join("\n"), "fresh": false})
}

/// Local timezone offset in hours, derived once from the difference between
/// libc-free sources we have: none. Darwin daemons run for one user; read
/// the TZ offset from `date +%z` once per call (cheap, no deps).
fn local_utc_offset_hours() -> u64 {
    let out = std::process::Command::new("date").arg("+%z").output().ok();
    let s = out
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    let s = s.trim();
    if s.len() >= 5 {
        let sign = if s.starts_with('-') { -1i64 } else { 1 };
        let hours: i64 = s[1..3].parse().unwrap_or(0);
        return ((24 + sign * hours) % 24) as u64;
    }
    0
}

// --- Memory ----------------------------------------------------------------------

fn memory_search(shared: &Arc<Shared>, query: &HashMap<String, String>) -> (&'static str, Value) {
    let q = query.get("q").cloned().unwrap_or_default();
    if q.trim().is_empty() {
        return ("400 Bad Request", json!({"error": "missing q"}));
    }
    let (rtx, rrx) = mpsc::channel();
    if shared.tx.send(Req::Search { q, resp: rtx }).is_err() {
        return ("500 Internal Server Error", json!({"error": "worker gone"}));
    }
    match rrx.recv() {
        Ok(v) => ("200 OK", v),
        Err(_) => ("500 Internal Server Error", json!({"error": "worker died"})),
    }
}

fn versions_json(vv: &aios::store::VersionedValue) -> Value {
    json!({
        "current": vv.current(),
        "last_updated": vv.last_updated(),
        "versions": vv.history.iter().map(|v| json!({
            "value": v.value, "timestamp": v.timestamp, "source": v.source
        })).collect::<Vec<_>>(),
    })
}

/// Identity at top, then branches with summary, recent facts, archive count,
/// and per-value version history (the store is copy-on-write; surface it).
fn memory_browse(shared: &Arc<Shared>) -> (&'static str, Value) {
    let store = shared.store.lock().unwrap();
    let branches: Vec<Value> = store
        .all_branches()
        .map(|b| {
            json!({
                "name": b.name,
                "created_at": b.created_at,
                "summary": versions_json(&b.summary),
                "details": b.details.iter().map(versions_json).collect::<Vec<_>>(),
                "archive": b.archive.len(),
                "tags": b.tags,
            })
        })
        .collect();
    ("200 OK", json!({
        "identity": versions_json(&store.identity),
        "branches": branches,
    }))
}

/// Correction writes a new version (copy-on-write), never destroys.
/// Body: {target: "identity" | "summary" | "detail", branch?, index?, value}
fn memory_correct(shared: &Arc<Shared>, body: &str) -> (&'static str, Value) {
    let Ok(v) = serde_json::from_str::<Value>(body) else {
        return ("400 Bad Request", json!({"error": "bad json"}));
    };
    let target = v["target"].as_str().unwrap_or("");
    let value = v["value"].as_str().unwrap_or("").trim();
    if value.is_empty() {
        return ("400 Bad Request", json!({"error": "empty value"}));
    }
    let branch = v["branch"].as_str().unwrap_or("");
    let index = v["index"].as_u64().map(|i| i as usize);
    let now = now_ts();

    let mut store = shared.store.lock().unwrap();
    let ok = match target {
        "identity" => {
            store.set_identity(value, "user_correction", now);
            true
        }
        "summary" => match store.get_branch_mut(branch) {
            Some(b) => {
                b.update_summary(value, "user_correction", now);
                true
            }
            None => false,
        },
        "detail" => match (store.get_branch_mut(branch), index) {
            (Some(b), Some(i)) if i < b.details.len() => {
                b.details[i].update(value, "user_correction", now);
                true
            }
            _ => false,
        },
        _ => false,
    };
    if !ok {
        return ("404 Not Found", json!({"error": "no such memory"}));
    }
    store.save(&shared.dirs.store_path()).ok();
    ("200 OK", json!({"corrected": true}))
}

/// Deletion is the one true delete and requires confirmation.
/// Body: {branch, target?: "branch" | "detail", index?, confirm: true}
fn memory_delete(shared: &Arc<Shared>, body: &str) -> (&'static str, Value) {
    let Ok(v) = serde_json::from_str::<Value>(body) else {
        return ("400 Bad Request", json!({"error": "bad json"}));
    };
    if v["confirm"].as_bool() != Some(true) {
        return ("400 Bad Request", json!({"error": "deletion requires confirm: true"}));
    }
    let branch = v["branch"].as_str().unwrap_or("");
    let target = v["target"].as_str().unwrap_or("branch");
    let index = v["index"].as_u64().map(|i| i as usize);

    let mut store = shared.store.lock().unwrap();
    let ok = match target {
        "detail" => match (store.get_branch_mut(branch), index) {
            (Some(b), Some(i)) if i < b.details.len() => {
                b.details.remove(i);
                true
            }
            _ => false,
        },
        "branch" => store.branches.remove(&aios::store::to_key(branch)).is_some(),
        "identity" => {
            store.identity = Default::default();
            true
        }
        _ => false,
    };
    if !ok {
        return ("404 Not Found", json!({"error": "no such memory"}));
    }
    store.save(&shared.dirs.store_path()).ok();
    ("200 OK", json!({"deleted": true}))
}

// --- Settings -----------------------------------------------------------------------

fn settings_get(shared: &Arc<Shared>) -> (&'static str, Value) {
    let s = shared.settings.lock().unwrap().clone();
    let mut v = serde_json::to_value(&s).unwrap_or(Value::Null);
    // Keys never leave the daemon; report presence only.
    v["keys_present"] = json!(shared.keys().present());
    ("200 OK", v)
}

fn settings_put(shared: &Arc<Shared>, body: &str) -> (&'static str, Value) {
    let Ok(patch) = serde_json::from_str::<Value>(body) else {
        return ("400 Bad Request", json!({"error": "bad json"}));
    };
    let Some(patch_obj) = patch.as_object() else {
        return ("400 Bad Request", json!({"error": "expected object"}));
    };

    let mut settings = shared.settings.lock().unwrap();
    let mut merged = serde_json::to_value(&*settings).unwrap_or(Value::Null);
    if let Some(base) = merged.as_object_mut() {
        for (k, v) in patch_obj {
            if base.contains_key(k) {
                base.insert(k.clone(), v.clone());
            }
        }
    }
    let new: Settings = match serde_json::from_value(merged) {
        Ok(s) => s,
        Err(e) => return ("400 Bad Request", json!({"error": format!("invalid settings: {e}")})),
    };
    if ![MODE_PERSISTENT, MODE_INCOGNITO, MODE_PAUSED].contains(&new.privacy_mode.as_str()) {
        return ("400 Bad Request", json!({"error": "privacy_mode must be persistent | incognito | paused"}));
    }

    let left_incognito = settings.privacy_mode == MODE_INCOGNITO && new.privacy_mode != MODE_INCOGNITO;
    *settings = new.clone();
    settings.save(&shared.dirs);
    drop(settings);

    if left_incognito {
        shared.journal.lock().unwrap().purge_ephemeral();
    }
    let _ = shared.tx.send(Req::SettingsChanged);
    let mut v = serde_json::to_value(&new).unwrap_or(Value::Null);
    v["keys_present"] = json!(shared.keys().present());
    ("200 OK", v)
}

/// What can answer right now: hosted presets gated on key presence, plus a
/// live inventory of self-hosted models. Bedrock is on the roadmap, not
/// integrated; it appears as unavailable so the UI can say so honestly.
fn models(shared: &Arc<Shared>) -> Value {
    let s = shared.settings.lock().unwrap().clone();
    let keys = shared.keys();
    let has_anthropic = keys.get("anthropic").is_some();
    let has_openai = keys.get("openai").is_some();

    let hosted = json!([
        {"provider": "claude", "model": "claude-sonnet-5", "label": "Claude Sonnet 5",
         "note": "fast frontier", "available": has_anthropic, "needs_key": "anthropic"},
        {"provider": "claude", "model": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5",
         "note": "cheap + quick", "available": has_anthropic, "needs_key": "anthropic"},
        {"provider": "claude", "model": "claude-opus-4-8", "label": "Claude Opus 4.8",
         "note": "deep work", "available": has_anthropic, "needs_key": "anthropic"},
        {"provider": "openai_compat", "model": s.model, "label": "OpenAI-compatible",
         "note": "any endpoint: OpenAI, OpenRouter, vLLM, LM Studio", "available": true,
         "needs_key": if has_openai { "" } else { "openai" }, "custom": true},
    ]);

    // Live local inventory straight from Ollama.
    let mut self_hosted: Vec<Value> = Vec::new();
    if let Ok(resp) = ureq::get("http://127.0.0.1:11434/api/tags").timeout(std::time::Duration::from_secs(2)).call() {
        if let Ok(v) = resp.into_json::<Value>() {
            for m in v["models"].as_array().unwrap_or(&Vec::new()) {
                if let Some(name) = m["name"].as_str() {
                    if name.contains("embed") {
                        continue; // embedding models don't chat
                    }
                    let short = name.strip_suffix(":latest").unwrap_or(name);
                    let tuned = short.starts_with("aios-ft");
                    self_hosted.push(json!({
                        "provider": "ollama", "model": short, "label": short,
                        "note": if tuned { "tuned for the memory protocol" } else { "local via Ollama" },
                        "available": true,
                    }));
                }
            }
        }
    }
    let llama_up = aios::llamaserver::LlamaServer::new(8080).healthy();
    self_hosted.push(json!({
        "provider": "llamaserver", "model": "llama-server", "label": "llama-server",
        "note": "attention-state paging to disk", "available": llama_up,
        "needs": if llama_up { "" } else { "start llama-server on :8080" },
    }));

    // Claude through the user's own AWS account. Real inference profiles
    // when the CLI session is live; an honest pointer when it is not.
    let aws_ok = crate::bedrock::credentials().is_ok();
    if aws_ok {
        let region = crate::bedrock::default_region();
        let profiles = crate::bedrock::list_claude_profiles(&region);
        if profiles.is_empty() {
            self_hosted.push(json!({
                "provider": "bedrock", "model": "", "label": "AWS Bedrock",
                "note": format!("no Claude inference profiles visible in {region}"),
                "available": false,
            }));
        }
        for id in profiles {
            let short = id.trim_start_matches("us.").trim_start_matches("eu.").trim_start_matches("apac.");
            self_hosted.push(json!({
                "provider": "bedrock", "model": id, "label": format!("bedrock · {short}"),
                "note": format!("your AWS account, {region}"), "available": true,
            }));
        }
    } else {
        self_hosted.push(json!({
            "provider": "bedrock", "model": "", "label": "AWS Bedrock",
            "note": "run `aws login` first", "available": false,
        }));
    }

    json!({
        "current": {"provider": s.provider, "model": s.model},
        "hosted": hosted,
        "self_hosted": self_hosted,
    })
}

fn kv(shared: &Arc<Shared>, save: bool) -> (&'static str, Value) {
    let (rtx, rrx) = mpsc::channel();
    let req = if save { Req::KvSave { resp: rtx } } else { Req::KvRestore { resp: rtx } };
    if shared.tx.send(req).is_err() {
        return ("500 Internal Server Error", json!({"error": "worker gone"}));
    }
    match rrx.recv() {
        Ok(v) => ("200 OK", v),
        Err(_) => ("500 Internal Server Error", json!({"error": "worker died"})),
    }
}

// --- Static UI ------------------------------------------------------------------------

/// Attached images, straight off ~/.aios/media.
fn serve_media(stream: &mut TcpStream, shared: &Arc<Shared>, path: &str) {
    let name = path.trim_start_matches("/v1/media/");
    if name.contains('/') || name.contains("..") {
        respond_json(stream, "400 Bad Request", &json!({"error": "bad path"}));
        return;
    }
    let file = shared.dirs.media_dir().join(name);
    match std::fs::read(&file) {
        Ok(bytes) => {
            let mime = match file.extension().and_then(|e| e.to_str()).unwrap_or("") {
                "jpg" => "image/jpeg",
                "webp" => "image/webp",
                "gif" => "image/gif",
                _ => "image/png",
            };
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nCache-Control: max-age=31536000\r\n\
                 Access-Control-Allow-Origin: *\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                bytes.len()
            );
            let _ = stream.write_all(&bytes);
        }
        Err(_) => respond_json(stream, "404 Not Found", &json!({"error": "no such media"})),
    }
}

/// Serve the built frontend (app/dist) when present, with an SPA fallback to
/// index.html. Without a build, serve a pointer to the dev server.
fn serve_static(stream: &mut TcpStream, shared: &Arc<Shared>, path: &str) {
    let rel = path.trim_start_matches('/');
    if rel.contains("..") {
        respond_json(stream, "400 Bad Request", &json!({"error": "bad path"}));
        return;
    }
    let base = std::path::Path::new(&shared.ui_dir);
    let mut file = if rel.is_empty() { base.join("index.html") } else { base.join(rel) };
    if !file.is_file() {
        file = base.join("index.html"); // SPA fallback
    }
    match std::fs::read(&file) {
        Ok(bytes) => {
            let mime = match file.extension().and_then(|e| e.to_str()).unwrap_or("") {
                "html" => "text/html; charset=utf-8",
                "js" => "application/javascript",
                "css" => "text/css",
                "svg" => "image/svg+xml",
                "png" => "image/png",
                "ico" => "image/x-icon",
                "woff2" => "font/woff2",
                _ => "application/octet-stream",
            };
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nAccess-Control-Allow-Origin: *\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                bytes.len()
            );
            let _ = stream.write_all(&bytes);
        }
        Err(_) => {
            let msg = "aios daemon is running. No UI build found — run `npm run build` in app/, \
                       or use the dev server (`npm run dev` in app/).";
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{msg}",
                msg.len()
            );
        }
    }
}

// --- HTTP plumbing -----------------------------------------------------------------------

fn split_query(raw: &str) -> (String, HashMap<String, String>) {
    let (path, qs) = raw.split_once('?').unwrap_or((raw, ""));
    let mut map = HashMap::new();
    for pair in qs.split('&').filter(|p| !p.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        map.insert(url_decode(k), url_decode(v));
    }
    (path.to_string(), map)
}

fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(b) => {
                        out.push(b);
                        i += 2;
                    }
                    Err(_) => out.push(bytes[i]),
                }
            }
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Minimal HTTP request reader: request line, headers, Content-Length body.
fn read_request(stream: &mut TcpStream) -> Option<(String, String, String)> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end;
    loop {
        let n = stream.read(&mut tmp).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            header_end = pos;
            break;
        }
        if buf.len() > 4_194_304 {
            return None;
        }
    }
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut first = head.lines().next()?.split_whitespace();
    let method = first.next()?.to_string();
    let path = first.next()?.to_string();

    let content_length: usize = head
        .lines()
        .find(|l| l.to_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);

    let mut body_bytes = buf[header_end + 4..].to_vec();
    while body_bytes.len() < content_length {
        let n = stream.read(&mut tmp).ok()?;
        if n == 0 {
            break;
        }
        body_bytes.extend_from_slice(&tmp[..n]);
    }
    Some((method, path, String::from_utf8_lossy(&body_bytes).to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_parsing_decodes() {
        let (path, q) = split_query("/v1/memory/search?q=when%20did%20I+decide&limit=5");
        assert_eq!(path, "/v1/memory/search");
        assert_eq!(q["q"], "when did I decide");
        assert_eq!(q["limit"], "5");
    }
}
