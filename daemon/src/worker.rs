//! The kernel actor: one long-lived thread owning the kernel, the eviction
//! window, and the fault loop. Requests come in over a channel, events go
//! out over per-turn channels, so a slow generation never blocks status
//! endpoints and cancellation is possible.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;

use serde_json::{json, Value};

use continuum::eviction::ContextWindow;
use continuum::hierarchical::{today_timestamp, HierarchicalTopicDriver};
use continuum::kernel::{detect_page_fault, Kernel, KernelConfig};
use continuum::llamaserver::LlamaServer;
use continuum::ollama::{ChatMessage, Ollama};

use crate::mcp::{self, McpServer};
use crate::providers;
use crate::state::{now_ts, EventTx, Req, Settings, Shared, MODE_INCOGNITO, MODE_PAUSED};
use crate::websearch;

const KV_SESSION: &str = "daemon_session.kv";

/// The companion voice. The kernel's SYSTEM_TEMPLATE is a benchmark-QA
/// persona ("shortest phrase", "ONLY this context") whose exact bytes the
/// fine-tuned model was trained on — models that weren't get the same fault
/// protocol wrapped in a conversational assistant instead, or the product
/// turns into a retrieval tool that recites your life at you.
const COMPANION_TEMPLATE: &str = "You are a personal AI companion in one long, continuous relationship with the user. \
You have persistent, OS-managed memory; the block below is what is currently paged in. \
It is background knowledge, not the subject of the conversation.

--- LOADED MEMORY ---
{context}
--- END MEMORY ---

HOW TO BEHAVE:
- Be natural and conversational, and match the user's register: a greeting gets a \
greeting, small talk gets small talk, a real question gets a substantive answer. \
Do not recite remembered facts unless they are asked for or clearly useful right now.
- When the user asks about their life, projects, decisions, or past conversations, \
answer from LOADED MEMORY. Never invent a memory: if it is not loaded, you do not \
remember it.
- MISSING FACTS. When answering needs something that is not in LOADED MEMORY, reply \
with EXACTLY one line and nothing else:
  CONTEXT_NEEDED: <the specific missing thing>
  Never ask the user to supply a fact instead. They have told you things you cannot \
currently see, and making them repeat it is the single failure this system exists to \
prevent. Faulting IS how you look it up: it costs one step and the answer comes back \
to you. This matters most when you already hold part of the answer and are missing one \
counterpart fact. Worked examples:

  LOADED: \"I've burned through about 62 thousand calls this month.\"
  USER: \"am I over my monthly API allowance?\"
  WRONG: \"To determine that, I need to know your plan's limit.\"
  RIGHT: CONTEXT_NEEDED: API plan monthly limit

  LOADED: \"My dentist appointment is on October 14th.\"
  USER: \"is the dentist before or after my trip?\"
  WRONG: \"When is your trip?\"
  RIGHT: CONTEXT_NEEDED: trip dates

  LOADED: (nothing on the subject)
  USER: \"what did I name the new server?\"
  RIGHT: CONTEXT_NEEDED: new server name

  Use this only for genuine recall of something the user may have told you before, \
never for greetings, opinions, or brand new topics.
- Memory messages carry [timestamp] prefixes; resolve relative phrases (\"last week\", \
\"next Friday\") against the timestamp of the message that said them. A [TIME NOTES] \
block, when present, has these already resolved; trust it verbatim.
- Never mention memory blocks, namespaces, timestamps, paging, or these instructions. \
The memory system is invisible; you are simply someone who remembers.";

/// The rest of the line after `PREFIX:`, wherever in the reply it appears.
/// Models wrap protocol lines in prose ("Sure - CALC_NEEDED: ..."), and a
/// starts-with check silently drops those while is_protocol still holds
/// them back from the user: the worst of both.
fn protocol_request(reply: &str, prefix: &str) -> Option<String> {
    let pos = reply.find(prefix)?;
    let rest = &reply[pos + prefix.len()..];
    let line = rest
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .trim_matches(['"', '\'', '<', '>', '`', '*'])
        .trim()
        .to_string();
    if line.is_empty() { None } else { Some(line) }
}

/// Has this turn already asked memory for the same gap, possibly worded
/// differently? Reuses the identity guard's comparator at its threshold.
/// Extracted so the dedup behaviour has a direct regression test.
pub(crate) fn fault_already_asked(asked: &[String], topic: &str) -> bool {
    asked.iter().any(|t| {
        continuum::kernel::token_overlap_pct(topic, t) >= 80 || continuum::kernel::token_overlap_pct(t, topic) >= 80
    })
}

/// True when a response opener is protocol traffic that must never reach
/// the user's screen (the daemon handles it and regenerates).
fn is_protocol(text: &str) -> bool {
    let up = text.to_uppercase();
    up.contains("CONTEXT_NEEDED")
        || up.contains("WEB_NEEDED")
        || up.contains("TOOL_NEEDED")
        || up.contains("CALC_NEEDED")
}

struct Worker {
    shared: Arc<Shared>,
    kernel: Kernel,
    window: ContextWindow<'static>,
    ollama: Ollama,
    ollama_up: bool,
    kv: bool,
    kv_restored: u64,
    /// The models the kernel-side Ollama handle was built with; a settings
    /// change to either forces a kernel rebuild.
    local_model: String,
    embed_model: String,
    turns_served: u64,
    /// Connected MCP servers (from ~/.continuum/mcp.json), tools and all.
    mcp: Vec<McpServer>,
}

pub fn run(rx: Receiver<Req>, shared: Arc<Shared>) {
    let settings = shared.settings.lock().unwrap().clone();
    let mut w = build_worker(shared.clone(), &settings);
    w.publish_status(Value::Null);
    eprintln!(
        "[worker] kernel up: {} messages indexed, ollama {}",
        w.kernel.driver().map(|d| d.namespace().to_string()).unwrap_or_default(),
        if w.ollama_up { "up" } else { "DOWN (memory formation degraded)" }
    );

    while let Ok(req) = rx.recv() {
        match req {
            Req::Turn { id, text, images, image_files, cancel, events } => {
                w.turn(id, &text, &images, &image_files, &cancel, &events)
            }
            Req::Search { q, resp } => {
                let out = w.search(&q);
                let _ = resp.send(out);
            }
            Req::KvSave { resp } => {
                let out = match w.kernel.save_kv(KV_SESSION) {
                    Ok(s) => json!({"saved_tokens": s.tokens, "bytes": s.bytes}),
                    Err(e) => json!({"error": e}),
                };
                let _ = resp.send(out);
            }
            Req::KvRestore { resp } => {
                let out = match w.kernel.restore_kv(KV_SESSION) {
                    Ok(n) => json!({"restored_tokens": n}),
                    Err(e) => json!({"error": e}),
                };
                let _ = resp.send(out);
            }
            Req::SettingsChanged => w.apply_settings(),
        }
    }
}

fn build_worker(shared: Arc<Shared>, s: &Settings) -> Worker {
    let ollama = Ollama::new(&s.local_model, &s.embed_model);
    let ollama_up = ollama.healthy();

    let mut driver = HierarchicalTopicDriver::load(&shared.dirs.driver_path())
        .unwrap_or_else(|_| HierarchicalTopicDriver::new("/social"));
    if ollama_up {
        driver.set_embedder(ollama.clone());
    }
    driver.route_cfg.max_load = s.max_retrieved.max(1);

    let config = KernelConfig {
        num_ctx: s.num_ctx,
        max_response_tokens: s.max_response_tokens,
        ..KernelConfig::default()
    };
    let mut kernel = Kernel::new(ollama.clone(), config);
    if s.social_enabled {
        kernel.mount(Box::new(driver));
    }

    // KV backend is opportunistic: mounted only when llama-server is already
    // up. Text is the source of truth; KV is a per-model cache tier.
    let kv_server = LlamaServer::new(8080);
    let kv = kv_server.healthy();
    let mut kv_restored = 0;
    if kv {
        kernel.set_kv_backend(kv_server);
        kv_restored = kernel.restore_kv(KV_SESSION).unwrap_or(0);
    }

    let mcp = mcp::start_all(&shared.dirs.mcp_path());

    Worker {
        shared,
        kernel,
        window: ContextWindow::new(s.window_budget, None),
        ollama,
        ollama_up,
        kv,
        kv_restored,
        local_model: s.local_model.clone(),
        embed_model: s.embed_model.clone(),
        turns_served: 0,
        mcp,
    }
}

impl Worker {
    /// Settings changed. Config knobs apply in place; a change to the
    /// kernel-side models (classifier/embedder) rebuilds the kernel from the
    /// persisted driver state. The KV tier is model-locked and simply not
    /// carried across; it rebuilds silently.
    fn apply_settings(&mut self) {
        let s = self.shared.settings.lock().unwrap().clone();
        if s.local_model != self.local_model || s.embed_model != self.embed_model {
            let evicted = std::mem::take(&mut self.window.evicted_summary);
            let old_slots = std::mem::take(&mut self.window.slots);
            *self = build_worker(self.shared.clone(), &s);
            self.window.evicted_summary = evicted;
            self.window.slots = old_slots;
        } else {
            self.kernel.config.num_ctx = s.num_ctx;
            self.kernel.config.max_response_tokens = s.max_response_tokens;
            if let Some(d) = self.kernel.driver_mut() {
                d.set_max_load(s.max_retrieved.max(1));
            }
            self.window.budget_tokens = s.window_budget;
        }
        self.publish_status(Value::Null);
    }

    fn send(&self, events: &EventTx, v: Value) {
        let _ = events.send(v);
    }

    /// The system prompt for this turn. Tuned models get SYSTEM_TEMPLATE's
    /// exact bytes (and no live actions — they weren't trained for them);
    /// everyone else gets the companion voice plus whatever the daemon can
    /// actually do right now: web search, MCP tools.
    fn build_template(&self, s: &Settings) -> String {
        if s.model.starts_with("aios-ft") {
            return continuum::kernel::SYSTEM_TEMPLATE.to_string();
        }
        let mut t = COMPANION_TEMPLATE.to_string();
        let mut actions = String::new();
        actions.push_str(
            "- NEVER do arithmetic on remembered numbers or dates in your head (sums, \
             differences, comparisons against limits, date shifts). Respond with EXACTLY\n  \
             CALC_NEEDED: <expression>\n  and nothing else, e.g. `CALC_NEEDED: 1800 + 200` \
             or `CALC_NEEDED: October 14 + 7 days`. The exact result comes back to you.\n",
        );
        if s.web_enabled {
            actions.push_str(
                "- The user asks about something current (news, prices, weather, releases, \
                 anything after your training): respond with EXACTLY\n  WEB_NEEDED: <search query>\n  \
                 and nothing else. The results will be handed back to you.\n",
            );
        }
        let tools: Vec<String> = self
            .mcp
            .iter()
            .flat_map(|srv| srv.tools.iter())
            .map(|t| format!("    {}.{} — {}", t.server, t.name, t.description))
            .collect();
        if !tools.is_empty() {
            actions.push_str(
                "- You can invoke these tools by responding with EXACTLY\n  \
                 TOOL_NEEDED: <server.tool> <json arguments>\n  and nothing else:\n",
            );
            actions.push_str(&tools.join("\n"));
            actions.push('\n');
        }
        if !actions.is_empty() {
            t.push_str("\n\nLIVE ACTIONS — real, executed by the memory system; use them only when the conversation genuinely needs them:\n");
            t.push_str(&actions);
        }
        t
    }

    /// One conversation turn: page in, stream generation (holding back
    /// protocol openers), memory fault-retry, then the action loop (web
    /// search, tools), memory formation, eviction, journal.
    fn turn(&mut self, id: u64, text: &str, images: &[String], image_files: &[String], cancel: &Arc<AtomicBool>, events: &EventTx) {
        let s = self.shared.settings.lock().unwrap().clone();
        let incognito = s.privacy_mode == MODE_INCOGNITO;
        let paused = s.privacy_mode == MODE_PAUSED;
        let mem_writes_allowed = !incognito && !paused;

        // The user turn goes to the journal first, so the timeline is
        // append-ordered even if generation dies.
        let user_entry = {
            let mut j = self.shared.journal.lock().unwrap();
            j.append("user", text, json!({"turn_id": id, "images": image_files}), incognito)
        };
        self.send(events, json!({"t": "turn", "id": id, "user_entry": user_entry}));

        // Identity always rides along; the store owns it.
        let identity = self.shared.store.lock().unwrap().get_identity().to_string();
        self.kernel.set_identity(&identity);

        // The store-into-context experiment: page query-relevant topics
        // (summary + current facts, latest values only) in alongside the
        // driver's raw messages. The tuned models never see this; they were
        // not trained with a [MEMORY TOPICS] block.
        let store_topics = if s.store_context && !s.model.starts_with("aios-ft") {
            let block = self.store_topics_block(text);
            let n = if block.is_empty() { 0 } else { block.matches("• ").count() };
            self.kernel.set_store_block(&block);
            n
        } else {
            self.kernel.set_store_block("");
            0
        };

        // Session RAM -> kernel session messages.
        let mut session: Vec<ChatMessage> = Vec::new();
        if !self.window.evicted_summary.is_empty() {
            session.push(ChatMessage::new("system", format!("[PREVIOUS CONTEXT] {}", self.window.evicted_summary)));
        }
        for slot in &self.window.slots {
            if let Some((role, content)) = slot.content.split_once(": ") {
                if role == "user" || role == "assistant" {
                    session.push(ChatMessage::new(role, content));
                }
            }
        }

        let template = self.build_template(&s);
        let t0 = std::time::Instant::now();
        let (mut messages, meta) = self.kernel.prepare_with(text, &session, &template);
        if !images.is_empty() {
            if let Some(last) = messages.last_mut() {
                last.images = Some(images.to_vec());
            }
        }
        let retrieval_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.send(events, json!({
            "t": "route",
            "loaded": meta.messages_loaded,
            "namespace": meta.namespace,
            "budget": meta.memory_budget_tokens,
            "retrieval_ms": retrieval_ms,
            "store_topics": store_topics,
        }));

        let keys = self.shared.keys();
        let provider = match providers::build(&s, &keys) {
            Ok(p) => p,
            Err(e) => {
                self.send(events, json!({"t": "err", "message": e}));
                self.send(events, json!({"t": "done", "error": true}));
                return;
            }
        };

        let t1 = std::time::Instant::now();
        let first = self.generate(provider.as_ref(), &messages, &s, cancel, events);
        let mut reply = first;
        let mut faulted = false;
        let mut fault_topic = String::new();

        // The action loop. Every protocol line the model can raise lands
        // here: memory faults (which may CHAIN, so a missing counterpart
        // fact triggers a second targeted re-page), web, tools, and exact
        // arithmetic. A dedup set stops a chain from asking for the same
        // thing twice in different words.
        let mut fault_topics_asked: Vec<String> = Vec::new();
        let mut actions: Vec<Value> = Vec::new();
        let mut loop_trace: Vec<Value> = Vec::new();
        for round in 0..4 {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let block = if let Some(topic) = detect_page_fault(&reply) {
                if topic == "unknown" {
                    loop_trace.push(json!({"round": round, "proto": "fault", "arg": topic, "outcome": "break: unnamed topic"}));
                    break; // soft refusal with no named topic: nothing to page
                }
                if fault_already_asked(&fault_topics_asked, &topic) {
                    loop_trace.push(json!({"round": round, "proto": "fault", "arg": topic, "outcome": "break: dedup, same gap re-asked"}));
                    break;
                }
                if !faulted {
                    faulted = true;
                    fault_topic = topic.clone();
                }
                self.send(events, json!({"t": "fault", "topic": topic}));
                fault_topics_asked.push(topic.clone());
                let paged = if s.fault_semantic_expansion {
                    self.kernel.fault_block_semantic(&topic, meta.memory_budget_tokens)
                } else {
                    self.kernel.fault_block(&topic, meta.memory_budget_tokens)
                };
                match paged {
                    Some(b) => {
                        actions.push(json!({"type": "repage", "topic": topic, "semantic": s.fault_semantic_expansion}));
                        loop_trace.push(json!({
                            "round": round, "proto": "fault", "arg": topic,
                            "outcome": format!("repage {} chars{}", b.len(), if s.fault_semantic_expansion { ", semantic" } else { "" }),
                            "block_preview": b.chars().take(110).collect::<String>(),
                        }));
                        format!(
                            "[ADDITIONAL MEMORY: {topic}]\n{b}\n\nUse this together with what \
                             was already loaded. If one specific needed fact is STILL missing, \
                             fault for exactly that fact; otherwise answer now."
                        )
                    }
                    None => {
                        loop_trace.push(json!({"round": round, "proto": "fault", "arg": topic, "outcome": "break: nothing paged in"}));
                        break; // nothing pages in: fall to the honest voice below
                    }
                }
            } else if let Some(expr) = protocol_request(&reply, "CALC_NEEDED:") {
                if actions.iter().any(|a| a["expr"] == expr.as_str()) {
                    loop_trace.push(json!({"round": round, "proto": "calc", "arg": expr, "outcome": "break: same expression re-asked"}));
                    break;
                }
                self.send(events, json!({"t": "tool", "name": "calc"}));
                match crate::calc::eval(&expr) {
                    Ok(v) => {
                        loop_trace.push(json!({"round": round, "proto": "calc", "arg": expr, "outcome": format!("= {v}")}));
                        actions.push(json!({"type": "calc", "expr": expr, "result": v.clone()}));
                        {
                            let mut j = self.shared.journal.lock().unwrap();
                            j.append("tool", &format!("calc: {expr} = {v}"), json!({"turn_id": id}), incognito);
                        }
                        format!("[CALC RESULT] {expr} = {v}\nUse this exact value in your answer.")
                    }
                    Err(e) => {
                        loop_trace.push(json!({"round": round, "proto": "calc", "arg": expr, "outcome": format!("error: {e}")}));
                        actions.push(json!({"type": "calc", "expr": expr, "error": e.clone()}));
                        format!("[CALC ERROR] {e}\nState the calculation in words instead of guessing a number.")
                    }
                }
            } else if let Some(query) = protocol_request(&reply, "WEB_NEEDED:") {
                if !s.web_enabled {
                    loop_trace.push(json!({"round": round, "proto": "web", "arg": query, "outcome": "break: web disabled"}));
                    break;
                }
                self.send(events, json!({"t": "web", "query": query}));
                let keys = self.shared.keys();
                let brave = keys.get("brave").map(|k| k.to_string());
                match websearch::search(&query, brave.as_deref()) {
                    Ok(hits) => {
                        actions.push(json!({"type": "web", "query": query, "results": hits.len()}));
                        {
                            let mut j = self.shared.journal.lock().unwrap();
                            j.append("web", &query, json!({"turn_id": id, "results": hits.len()}), incognito);
                        }
                        loop_trace.push(json!({"round": round, "proto": "web", "arg": query, "outcome": format!("{} results", hits.len())}));
                        websearch::render_block(&query, &hits)
                    }
                    Err(e) => {
                        actions.push(json!({"type": "web", "query": query, "error": e}));
                        format!("[WEB ERROR] {e}\nAnswer from what you already know, and say plainly that you couldn't search.")
                    }
                }
            } else if let Some(rest) = protocol_request(&reply, "TOOL_NEEDED:") {
                let Some((server_name, tool, args)) = mcp::parse_tool_request(&rest) else {
                    loop_trace.push(json!({"round": round, "proto": "tool", "arg": rest, "outcome": "break: unparseable tool request"}));
                    break;
                };
                self.send(events, json!({"t": "tool", "name": format!("{server_name}.{tool}")}));
                let outcome = self
                    .mcp
                    .iter_mut()
                    .find(|m| m.name == server_name)
                    .ok_or_else(|| format!("no MCP server named '{server_name}'"))
                    .and_then(|m| m.call(&tool, args));
                match outcome {
                    Ok(out) => {
                        actions.push(json!({"type": "tool", "name": format!("{server_name}.{tool}")}));
                        {
                            let mut j = self.shared.journal.lock().unwrap();
                            j.append("tool", &format!("{server_name}.{tool}"), json!({"turn_id": id}), incognito);
                        }
                        let capped: String = out.chars().take(4000).collect();
                        format!("[TOOL RESULT: {server_name}.{tool}]\n{capped}\n\nAnswer the user using this result.")
                    }
                    Err(e) => {
                        actions.push(json!({"type": "tool", "name": format!("{server_name}.{tool}"), "error": e.clone()}));
                        format!("[TOOL ERROR: {server_name}.{tool}] {e}\nTell the user plainly that the tool failed.")
                    }
                }
            } else {
                break; // a real answer: the loop's job is done
            };
            messages.push(ChatMessage::new("assistant", reply.trim()));
            messages.push(ChatMessage::new("user", block));
            reply = self.generate(provider.as_ref(), &messages, &s, cancel, events);
        }
        let retried = faulted && detect_page_fault(&reply).is_none() && !is_protocol(&reply);

        // A protocol line the loop couldn't satisfy must never be the reply.
        if detect_page_fault(&reply).is_some() {
            // A memory fault the chain couldn't resolve: the honest voice,
            // never raw protocol text.
            let topic = fault_topic.trim_start_matches("/social/").replace('_', " ");
            reply = if topic.is_empty() || topic == "unknown" {
                "That isn't in my memory yet. Tell me and I'll remember it.".to_string()
            } else {
                format!("I don't have anything about {topic} in memory yet. Tell me and I'll remember it.")
            };
            self.send(events, json!({"t": "tok", "v": reply.as_str()}));
        } else if is_protocol(&reply) {
            loop_trace.push(json!({"proto": "wedge", "unresolved": reply.chars().take(160).collect::<String>()}));
            reply = "I tried to reach for outside help there and couldn't. Ask me again, or rephrase?".into();
            self.send(events, json!({"t": "tok", "v": reply.as_str()}));
        }
        let generation_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let reply = reply.trim().to_string();
        let was_cancelled = cancel.load(Ordering::Relaxed);
        // A provider error is not a conversation. Show it, journal it, but
        // never let it form memory or advance the session window.
        let errored = reply.starts_with("[ERROR:");

        // Memory formation. Classification is always local (kernel-side
        // model); provenance points at the journal entry of the user turn.
        let mut writes: Vec<Value> = Vec::new();
        if mem_writes_allowed && !was_cancelled && !errored && !reply.is_empty() {
            let source = format!("journal:{user_entry}");
            let wbs = self.classify(&s, provider.as_ref(), text, &reply);
            let mut store = self.shared.store.lock().unwrap();
            Kernel::apply_write_backs_from(&mut store, &wbs, &source, now_ts());
            drop(store);
            if let Some(driver) = self.kernel.driver_mut() {
                driver.ingest_turn("user", text, &today_timestamp());
                if !reply.to_uppercase().contains("CONTEXT_NEEDED") {
                    driver.ingest_turn("assistant", &reply, &today_timestamp());
                }
            }
            for wb in wbs.iter().filter(|w| w.kind != "EPHEMERAL" && !w.content.is_empty()) {
                let v = json!({"kind": wb.kind, "content": wb.content, "branch": wb.branch});
                self.send(events, json!({"t": "mem", "kind": wb.kind, "content": wb.content, "branch": wb.branch}));
                let mut j = self.shared.journal.lock().unwrap();
                j.append("memory", &wb.content, json!({"turn_id": id, "kind": wb.kind, "branch": wb.branch}), false);
                writes.push(v);
            }
        }

        // Session RAM bookkeeping. The window always advances (otherwise the
        // conversation loses its own thread); demotions only reach the store
        // when writes are allowed.
        if !was_cancelled && !errored && !reply.is_empty() {
            self.window.load_message("user", text, false);
            self.window.load_message("assistant", &reply, false);
            if self.window.pressure_level() != "OK" {
                let before = self.window.total_evictions;
                self.window.evict_messages(4);
                let evicted = self.window.total_evictions - before;
                if evicted > 0 {
                    self.send(events, json!({"t": "evict", "n": evicted}));
                    let mut j = self.shared.journal.lock().unwrap();
                    j.append("evict", &format!("{evicted} messages demoted"), json!({"turn_id": id, "n": evicted}), incognito);
                }
            }
            let demotions = self.window.drain_demotions();
            if mem_writes_allowed {
                let mut store = self.shared.store.lock().unwrap();
                for (branch, role, content) in demotions {
                    store.add_archive(&branch, &role, &content, now_ts());
                }
            }
        }

        if mem_writes_allowed {
            let store = self.shared.store.lock().unwrap();
            store.save(&self.shared.dirs.store_path()).ok();
            drop(store);
            if let Some(d) = self.kernel.driver() {
                d.persist(&self.shared.dirs.driver_path()).ok();
            }
        }

        let inspector = json!({
            "turn_id": id,
            "namespace": meta.namespace,
            "loaded": meta.messages_loaded,
            "budget": meta.memory_budget_tokens,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "store_topics": store_topics,
            "faulted": faulted,
            "fault_topic": fault_topic,
            "retried": retried,
            "provider": provider.label(),
            "writes": writes,
            "actions": actions,
            "loop_trace": loop_trace,
            "cancelled": was_cancelled,
            "errored": errored,
            "privacy_mode": s.privacy_mode,
        });
        let assistant_entry = {
            let mut j = self.shared.journal.lock().unwrap();
            j.append("assistant", &reply, inspector.clone(), incognito)
        };

        self.turns_served += 1;
        self.publish_status(inspector.clone());
        self.send(events, json!({
            "t": "done",
            "turn_id": id,
            "entry": assistant_entry,
            "reply": reply,
            "inspector": inspector,
            "pressure": self.pressure(),
        }));
    }

    /// The store block for one turn: the topics whose text shares words
    /// with the query, rendered as summary plus latest fact values. Current
    /// values only; history stays in the store. Empty when nothing matches,
    /// so unrelated turns carry no store baggage.
    fn store_topics_block(&self, query: &str) -> String {
        let terms: Vec<String> = query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() > 2)
            .map(String::from)
            .collect();
        if terms.is_empty() {
            return String::new();
        }
        let store = self.shared.store.lock().unwrap();
        let mut scored: Vec<(usize, String)> = store
            .all_branches()
            .filter_map(|b| {
                let hay = b.all_text().to_lowercase();
                let hits = terms.iter().filter(|t| hay.contains(t.as_str())).count();
                if hits == 0 {
                    return None;
                }
                let mut lines = format!("• {}: {}", b.name, b.summary.current());
                for d in b.details.iter().rev().take(4) {
                    let cur = d.current();
                    let body = cur.split_once("] ").map(|(_, t)| t).unwrap_or(cur);
                    lines.push_str(&format!("\n  - {body}"));
                }
                Some((hits, lines))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        let mut block = String::new();
        for (_, lines) in scored.into_iter().take(3) {
            if block.len() + lines.len() > 900 {
                break;
            }
            if !block.is_empty() {
                block.push('\n');
            }
            block.push_str(&lines);
        }
        block
    }

    /// Memory formation classification. Local by default (private); with
    /// memory_model = "answer" the active provider classifies instead —
    /// a frontier model beats any amount of chain-of-thought coaxing on an
    /// 8B here, at the cost of the exchange leaving the machine twice.
    fn classify(&self, s: &Settings, provider: &dyn providers::Provider, user: &str, reply: &str) -> Vec<continuum::kernel::WriteBack> {
        let store = self.shared.store.lock().unwrap();
        if s.memory_model == "answer" && s.provider != "ollama" {
            let branches = serde_json::to_string(&store.list_branches()).unwrap_or_else(|_| "[]".into());
            drop(store);
            let prompt = continuum::kernel::WRITEBACK_PROMPT
                .replace("{branches}", &branches)
                .replace("{user_msg}", user)
                .replace("{response}", reply);
            let msgs = [
                ChatMessage::new("system", "Output only JSON."),
                ChatMessage::new("user", prompt),
            ];
            let never = AtomicBool::new(false);
            match provider.chat_stream(&msgs, 300, 0.0, &mut |_| {}, &never) {
                Ok(raw) => continuum::kernel::parse_write_backs(&raw),
                Err(_) => Vec::new(),
            }
        } else {
            self.kernel.classify_write_back(&store, user, reply)
        }
    }

    /// Stream one completion, holding the first ~24 chars back so a
    /// CONTEXT_NEEDED opener can be intercepted before the user sees it.
    fn generate(
        &self,
        provider: &dyn providers::Provider,
        messages: &[ChatMessage],
        s: &Settings,
        cancel: &Arc<AtomicBool>,
        events: &EventTx,
    ) -> String {
        let mut held = String::new();
        let mut flushed = false;
        let out = provider.chat_stream(
            messages,
            s.max_response_tokens,
            s.temperature,
            &mut |piece| {
                if flushed {
                    let _ = events.send(json!({"t": "tok", "v": piece}));
                    return;
                }
                held.push_str(piece);
                if held.len() >= 24 || held.contains('\n') {
                    if !is_protocol(&held) {
                        let _ = events.send(json!({"t": "tok", "v": held.as_str()}));
                        flushed = true;
                    }
                    // A protocol opener stays held; the caller handles it.
                }
            },
            cancel,
        );
        match out {
            Ok(full) => {
                if !flushed && detect_page_fault(&full).is_none() && !is_protocol(&full) && !full.is_empty() {
                    let _ = events.send(json!({"t": "tok", "v": held.as_str()}));
                }
                full
            }
            Err(e) => {
                let _ = events.send(json!({"t": "err", "message": e.clone()}));
                format!("[ERROR: {e}]")
            }
        }
    }

    /// Ranked memories with sources: driver-routed turns plus matching store
    /// branches (current value + version history).
    fn search(&self, q: &str) -> Value {
        let t0 = std::time::Instant::now();
        let mut turns: Vec<Value> = Vec::new();
        if let Some(driver) = self.kernel.driver() {
            let embedding = self.ollama.embed(q).unwrap_or_default();
            let indices = driver.route_query(q, &embedding);
            for idx in indices.into_iter().take(20) {
                if let Some((speaker, text, timestamp)) = driver.get_message(idx) {
                    turns.push(json!({"idx": idx, "speaker": speaker, "text": text, "timestamp": timestamp}));
                }
            }
        }

        let terms: Vec<String> = q
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() > 2)
            .map(|t| t.to_string())
            .collect();
        let store = self.shared.store.lock().unwrap();
        let mut memories: Vec<Value> = Vec::new();
        for b in store.all_branches() {
            let hay = b.all_text().to_lowercase();
            let hits = terms.iter().filter(|t| hay.contains(t.as_str())).count();
            if hits == 0 {
                continue;
            }
            let details: Vec<Value> = b
                .details
                .iter()
                .enumerate()
                .filter(|(_, d)| {
                    let dl = d.current().to_lowercase();
                    terms.iter().any(|t| dl.contains(t.as_str()))
                })
                .map(|(i, d)| {
                    json!({
                        "index": i,
                        "value": d.current(),
                        "last_updated": d.last_updated(),
                        "versions": d.history.iter().map(|v| json!({
                            "value": v.value, "timestamp": v.timestamp, "source": v.source
                        })).collect::<Vec<_>>(),
                    })
                })
                .collect();
            memories.push(json!({
                "branch": b.name,
                "score": hits,
                "summary": b.summary.current(),
                "details": details,
            }));
        }
        memories.sort_by(|a, b| b["score"].as_u64().cmp(&a["score"].as_u64()));

        json!({
            "query": q,
            "turns": turns,
            "memories": memories,
            "search_ms": t0.elapsed().as_secs_f64() * 1000.0,
        })
    }

    fn pressure(&self) -> Value {
        json!({
            "used": self.window.used_tokens(),
            "budget": self.window.budget_tokens,
            "level": self.window.pressure_level(),
            "evictions": self.window.total_evictions,
        })
    }

    fn publish_status(&self, last_turn: Value) {
        let s = self.shared.settings.lock().unwrap().clone();
        let store_stats = {
            let store = self.shared.store.lock().unwrap();
            serde_json::to_value(store.stats()).unwrap_or(Value::Null)
        };
        let journal_len = self.shared.journal.lock().unwrap().len();
        let drivers: Vec<Value> = self
            .kernel
            .driver()
            .map(|d| vec![json!({"namespace": d.namespace()})])
            .unwrap_or_default();
        let snapshot = json!({
            "provider": s.provider,
            "model": s.model,
            "local_model": s.local_model,
            "privacy_mode": s.privacy_mode,
            "ollama_up": self.ollama_up,
            "kv": {"mounted": self.kv, "restored_tokens": self.kv_restored},
            "web": {
                "enabled": s.web_enabled,
                "provider": websearch::provider_name(self.shared.keys().get("brave")),
            },
            "mcp": self.mcp.iter().map(|m| json!({
                "name": m.name,
                "tools": m.tools.iter().map(|t| t.name.clone()).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "drivers": drivers,
            "pressure": self.pressure(),
            "counters": {
                "turns_served": self.turns_served,
                "journal_entries": journal_len,
                "store": store_stats,
            },
            "last_turn": last_turn,
        });
        *self.shared.status.lock().unwrap() = snapshot;
    }
}

#[cfg(test)]
mod tests {
    use super::fault_already_asked;

    #[test]
    fn fault_dedup_catches_the_same_gap_in_different_words() {
        let asked = vec!["API plan monthly limit".to_string()];
        // Same gap, reworded: suppressed.
        assert!(fault_already_asked(&asked, "monthly limit of the API plan"));
        assert!(fault_already_asked(&asked, "the API plan's monthly limit"));
        // A genuinely different gap: allowed through, the chain may continue.
        assert!(!fault_already_asked(&asked, "current API usage this month"));
        assert!(!fault_already_asked(&asked, "dentist appointment date"));
        // Second ask joins the set and is itself suppressed thereafter.
        let asked = vec!["API plan monthly limit".into(), "current API usage this month".into()];
        assert!(fault_already_asked(&asked, "this month's current usage of the API"));
    }
}
