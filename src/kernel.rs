//! The AIOS Kernel, domain-agnostic (spec §3.1).
//!
//! Responsibilities, and *only* these:
//!   - Track VRAM (token) pressure and compute a memory budget.
//!   - Route a query to the right Volume/Namespace driver.
//!   - Assemble context with strict VFS namespace boundaries (spec §4.3).
//!   - Intercept the LLM's `CONTEXT_NEEDED` page fault and re-route.
//!
//! The kernel knows nothing about trees, ASTs, embeddings, or BM25, that all
//! lives behind `MemoryIndexDriver`.

use crate::driver::MemoryIndexDriver;
use crate::ollama::{ChatMessage, Ollama};
use crate::store::{MemoryStore, Timestamp};

pub const SYSTEM_TEMPLATE: &str = "You are a personal AI assistant with persistent, OS-managed memory.
Below is the memory currently paged into your context. Answer using ONLY this context.

RULES:
- Answer with the shortest phrase that fully answers the question — no preamble,
  no \"Based on the context\", no restating the question. \"7 May 2023\" beats
  \"According to the conversation, it was on 7 May 2023.\"
- If the answer is in your loaded context, answer directly and concisely.
- If the user asks about something NOT in your loaded context, respond with EXACTLY:
  CONTEXT_NEEDED: <topic>
  and nothing else. Do NOT guess, infer, or invent facts that are not present.
- Memory blocks are namespaced (e.g. /social, /workspace). Do not mix rules across namespaces.
- Messages carry [timestamp] prefixes. For \"when\" questions, derive the date from those
  timestamps. Relative phrases inside a message (\"last week\", \"yesterday\") are relative
  to THAT message's timestamp — resolve them (e.g. \"last week\" said on 9 June 2023 means
  the week before 9 June 2023). Answer with the resolved date.
- A [TIME NOTES] block may follow the messages with relative dates ALREADY RESOLVED
  by the memory system. Trust those resolutions verbatim for \"when\" questions.

--- LOADED MEMORY ---
{context}
--- END MEMORY ---";

pub struct KernelConfig {
    pub num_ctx: usize,
    pub max_response_tokens: usize,
    pub system_overhead_tokens: usize,
    pub enable_page_faults: bool,
    pub max_fault_retries: usize,
}

impl Default for KernelConfig {
    fn default() -> Self {
        KernelConfig {
            num_ctx: 4096,
            max_response_tokens: 300,
            system_overhead_tokens: 150,
            enable_page_faults: true,
            max_fault_retries: 1,
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct QueryResult {
    pub response: String,
    pub page_faulted: bool,
    pub fault_topic: String,
    pub fault_retried: bool,
    pub namespace: String,
    pub route_path: String,
    pub messages_loaded: usize,
    pub memory_budget_tokens: usize,
    /// Time spent on the memory side: query embedding, routing, block load.
    pub retrieval_ms: f64,
    /// Time spent generating (including any fault retry).
    pub generation_ms: f64,
}

/// The kernel owns a set of drivers and the LLM handle. Each driver is a Volume.
pub struct Kernel {
    pub config: KernelConfig,
    ollama: Ollama,
    /// Optional KV-paging backend (llama-server). When mounted, chat runs
    /// through it, and attention states can be saved/restored to disk.
    /// Embeddings always stay on Ollama.
    kv: Option<crate::llamaserver::LlamaServer>,
    drivers: Vec<Box<dyn MemoryIndexDriver>>,
    identity: String,
    /// Optional per-turn block of store content (topic summaries and
    /// current facts) to page in alongside the driver's messages. Empty
    /// means the store stays out of the prompt, which was the only
    /// behavior before the store-into-context experiment.
    store_block: String,
}

impl Kernel {
    pub fn new(ollama: Ollama, config: KernelConfig) -> Self {
        Kernel { config, ollama, kv: None, drivers: Vec::new(), identity: String::new(), store_block: String::new() }
    }

    /// Mount the KV-paging inference backend.
    pub fn set_kv_backend(&mut self, server: crate::llamaserver::LlamaServer) {
        self.kv = Some(server);
    }

    pub fn has_kv_backend(&self) -> bool {
        self.kv.is_some()
    }

    /// Page-out: persist the current slot's attention states to disk.
    pub fn save_kv(&self, filename: &str) -> Result<crate::llamaserver::SlotSave, String> {
        self.kv.as_ref().ok_or("no KV backend mounted")?.save_slot(0, filename)
    }

    /// Page-in: map attention states back from disk. Returns tokens restored.
    pub fn restore_kv(&self, filename: &str) -> Result<u64, String> {
        self.kv.as_ref().ok_or("no KV backend mounted")?.restore_slot(0, filename)
    }

    /// Public wrapper for surfaces that assemble their own message lists.
    pub fn complete_messages(&self, messages: &[ChatMessage]) -> Result<String, String> {
        self.complete(messages)
    }

    /// One chat completion via whichever backend is mounted.
    fn complete(&self, messages: &[ChatMessage]) -> Result<String, String> {
        match &self.kv {
            Some(server) => server.chat(messages, self.config.max_response_tokens),
            None => self.ollama.chat(messages, self.config.num_ctx, self.config.max_response_tokens),
        }
    }

    pub fn mount(&mut self, driver: Box<dyn MemoryIndexDriver>) {
        self.drivers.push(driver);
    }

    pub fn set_identity(&mut self, identity: &str) {
        self.identity = identity.to_string();
    }

    /// Set (or clear, with "") the store content paged in for the next turn.
    pub fn set_store_block(&mut self, block: &str) {
        self.store_block = block.to_string();
    }

    fn compute_budget(&self, session: &[ChatMessage]) -> usize {
        let mut used = self.config.system_overhead_tokens + self.config.max_response_tokens;
        for m in session.iter().rev().take(6) {
            used += m.content.len() / 4;
        }
        self.config.num_ctx.saturating_sub(used).max(200)
    }

    /// Route through the (single) mounted driver and assemble a namespaced block.
    /// Returns (context, namespace, route_path, messages_loaded).
    fn page_in(&self, topic: &str, budget: usize) -> (String, String, String, usize) {
        // For now a single Volume; a full VFS would pick by namespace prefix.
        let Some(driver) = self.drivers.first() else {
            return (String::new(), String::new(), "no driver".into(), 0);
        };
        let embedding = self.ollama.embed(topic).unwrap_or_default();
        let indices = driver.route_query(topic, &embedding);
        let (body, _tokens) = driver.load_messages(&indices, budget);
        let ns = driver.namespace().trim_start_matches('/');
        let topic_key = to_slug(topic);
        let block = if body.is_empty() {
            String::new()
        } else {
            // VFS namespace boundary (spec §4.3).
            format!("[MEMORY_BLOCK: /{ns}/{topic_key}]\n{body}")
        };
        (block, driver.namespace().to_string(), format!("{}", indices.len()), indices.len())
    }

    fn assemble_prompt(&self, context: &str, template: &str) -> String {
        let mut ctx = String::new();
        if !self.identity.is_empty() {
            ctx.push_str(&format!("[IDENTITY]\n{}\n\n", self.identity));
        }
        if !self.store_block.is_empty() {
            ctx.push_str(&format!("[MEMORY TOPICS]\n{}\n\n", self.store_block));
        }
        ctx.push_str(context);
        template.replace("{context}", &ctx)
    }

    /// Page in memory for a query and build the full message list, without
    /// calling the model. Lets callers that stream generation themselves (the
    /// web server) reuse the exact routing and assembly the kernel uses.
    pub fn prepare(&self, user_message: &str, session: &[ChatMessage]) -> (Vec<ChatMessage>, QueryResult) {
        self.prepare_with(user_message, session, SYSTEM_TEMPLATE)
    }

    /// Same, with a caller-supplied system template (`{context}` placeholder).
    /// The tuned local model needs SYSTEM_TEMPLATE's exact bytes; other
    /// surfaces may want the same routing under a different voice.
    pub fn prepare_with(&self, user_message: &str, session: &[ChatMessage], template: &str) -> (Vec<ChatMessage>, QueryResult) {
        let mut result = QueryResult::default();
        let budget = self.compute_budget(session);
        result.memory_budget_tokens = budget;

        let (context, ns, route_path, loaded) = self.page_in(user_message, budget);
        result.namespace = ns;
        result.route_path = route_path;
        result.messages_loaded = loaded;

        let system = self.assemble_prompt(&context, template);
        let mut messages = vec![ChatMessage::new("system", system)];
        for m in session.iter().rev().take(6).rev() {
            messages.push(m.clone());
        }
        messages.push(ChatMessage::new("user", user_message));
        (messages, result)
    }

    /// Same as prepare, but pages in on the FAULT topic instead of the user
    /// message. Returns None when nothing pages in for that topic.
    pub fn prepare_fault(&self, topic: &str, user_msg: &str, session: &[ChatMessage], budget: usize) -> Option<Vec<ChatMessage>> {
        self.prepare_fault_with(topic, user_msg, session, budget, SYSTEM_TEMPLATE)
    }

    /// Page in the driver block for a fault topic, without rebuilding the
    /// message list. The daemon's action loop appends it as new context so
    /// faults can chain within one turn.
    pub fn fault_block(&self, topic: &str, budget: usize) -> Option<String> {
        let (context, _ns, _path, _n) = self.page_in(topic, budget);
        if context.trim().is_empty() {
            None
        } else {
            Some(context)
        }
    }

    /// fault_block plus semantic expansion: union the lexical route with the
    /// fault topic's pure dense neighbours, gate bypassed. For gaps the
    /// model names in different words than the user spoke ("engineers
    /// hired" vs "people on the platform team").
    pub fn fault_block_semantic(&self, topic: &str, budget: usize) -> Option<String> {
        let driver = self.drivers.first()?;
        let embedding = self.ollama.embed(topic).unwrap_or_default();
        let mut indices = driver.route_query(topic, &embedding);
        for idx in driver.semantic_neighbors(&embedding, 8) {
            if !indices.contains(&idx) {
                indices.push(idx);
            }
        }
        if indices.is_empty() {
            return None;
        }
        let (body, _tokens) = driver.load_messages(&indices, budget);
        if body.trim().is_empty() {
            return None;
        }
        let ns = driver.namespace().trim_start_matches('/');
        Some(format!("[MEMORY_BLOCK: /{ns}/{}]\n{body}", to_slug(topic)))
    }

    pub fn prepare_fault_with(&self, topic: &str, user_msg: &str, session: &[ChatMessage], budget: usize, template: &str) -> Option<Vec<ChatMessage>> {
        let (context, _ns, _path, _n) = self.page_in(topic, budget);
        if context.trim().is_empty() {
            return None;
        }
        let system = self.assemble_prompt(&context, template);
        let mut messages = vec![ChatMessage::new("system", system)];
        for m in session.iter().rev().take(4).rev() {
            messages.push(m.clone());
        }
        messages.push(ChatMessage::new("user", user_msg));
        Some(messages)
    }

    /// Full inference loop with the page-fault retry.
    pub fn query(&self, user_message: &str, session: &[ChatMessage]) -> QueryResult {
        let t0 = std::time::Instant::now();
        let (messages, mut result) = self.prepare(user_message, session);
        result.retrieval_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        result.response = self.complete(&messages).unwrap_or_else(|e| format!("[ERROR: {e}]"));
        result.generation_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // Page fault interception (spec §3.1).
        if self.config.enable_page_faults {
            if let Some(topic) = detect_page_fault(&result.response) {
                result.page_faulted = true;
                result.fault_topic = topic.clone();
                if self.config.max_fault_retries > 0 {
                    if let Some(retry) = self.handle_fault(&topic, user_message, session, result.memory_budget_tokens) {
                        result.response = retry;
                        result.fault_retried = true;
                    }
                }
            }
        }
        result
    }

    fn handle_fault(&self, topic: &str, user_msg: &str, session: &[ChatMessage], budget: usize) -> Option<String> {
        let messages = self.prepare_fault(topic, user_msg, session, budget)?;
        let resp = self.complete(&messages).ok()?;
        if resp.to_uppercase().contains("CONTEXT_NEEDED") {
            None
        } else {
            Some(resp)
        }
    }
}

// --- Write-back: memory formation (ported from the Python runtime) --------

pub const WRITEBACK_PROMPT: &str = "Classify memory updates from this exchange.
Existing branches: {branches}

User: {user_msg}
Assistant: {response}

For each new piece of information worth remembering, output a JSON object:
{\"type\":\"<TYPE>\",\"content\":\"<what>\",\"branch\":\"<where>\"}

Types: BRANCH_UPDATE, NEW_BRANCH, DECISION, PREFERENCE_CHANGE, IDENTITY_UPDATE, EPHEMERAL

Remember only NEW facts the User stated about themselves, their life, their
decisions, or their work. Greetings, questions, chit-chat, and things the
Assistant said are EPHEMERAL. Facts already covered by an existing branch
are EPHEMERAL unless they changed.
IDENTITY_UPDATE is only for who the user is: name, role, lasting traits.
Appointments, possessions, times, numbers, and events belong in branches.
Corrections and things the user did (lent, booked, moved) are worth
remembering too.
Output ONLY a JSON array. If nothing new: []";

#[derive(Debug, Clone)]
pub struct WriteBack {
    pub kind: String,
    pub content: String,
    pub branch: String,
}

/// Parse the classifier's raw output into write-backs. Tolerates ```json
/// fences and skips malformed entries. Pure function, unit-testable.
pub fn parse_write_backs(raw: &str) -> Vec<WriteBack> {
    let cleaned = raw.replace("```json", "").replace("```", "");
    let start = cleaned.find('[');
    let end = cleaned.rfind(']');
    let (Some(s), Some(e)) = (start, end) else { return Vec::new() };
    if e <= s {
        return Vec::new();
    }
    let Ok(items) = serde_json::from_str::<Vec<serde_json::Value>>(&cleaned[s..=e]) else {
        return Vec::new();
    };
    items
        .into_iter()
        .filter_map(|v| {
            let kind = v.get("type")?.as_str()?.to_string();
            let content = v.get("content")?.as_str().unwrap_or("").to_string();
            let branch = v.get("branch").and_then(|b| b.as_str()).unwrap_or("").to_string();
            Some(WriteBack { kind, content, branch })
        })
        .collect()
}

impl Kernel {
    /// Classify what this exchange should write to memory (one LLM call).
    pub fn classify_write_back(&self, store: &MemoryStore, user_msg: &str, response: &str) -> Vec<WriteBack> {
        let branches = serde_json::to_string(&store.list_branches()).unwrap_or_else(|_| "[]".into());
        let prompt = WRITEBACK_PROMPT
            .replace("{branches}", &branches)
            .replace("{user_msg}", user_msg)
            .replace("{response}", response);
        let msgs = [
            ChatMessage::new("system", "Output only JSON."),
            ChatMessage::new("user", prompt),
        ];
        match self.ollama.chat(&msgs, 2048, 200) {
            Ok(raw) => parse_write_backs(&raw),
            Err(_) => Vec::new(),
        }
    }

    /// Apply write-backs to the store. Returns how many changed the store.
    pub fn apply_write_backs(store: &mut MemoryStore, wbs: &[WriteBack], now: Timestamp) -> usize {
        Self::apply_write_backs_from(store, wbs, "write_back", now)
    }

    /// Same, with an explicit provenance string (e.g. `turn:42`) so surfaces
    /// can link a memory back to the exchange that formed it.
    ///
    /// Deduplicates as it applies: small classifiers restate the same fact
    /// under two labels in one batch, and re-remember loaded context on later
    /// turns. A fact whose content already lives in the target branch (or in
    /// the identity) is a no-op, not a thirteenth copy.
    ///
    /// Two more disciplines, both measured needs, not speculation:
    /// classifiers use IDENTITY_UPDATE as a catch-all, so value-bearing
    /// facts (two or more digits) reroute to a branch detail where they get
    /// versioning and provenance instead of a string merge; and a detail
    /// that narrowly restates the branch summary supersedes it, so a
    /// summary can't keep advertising the dentist's old date after the
    /// details learned the new one.
    pub fn apply_write_backs_from(store: &mut MemoryStore, wbs: &[WriteBack], source: &str, now: Timestamp) -> usize {
        let norm = |s: &str| s.to_lowercase().trim().trim_end_matches(['.', '!']).to_string();
        let mut seen: Vec<String> = Vec::new();
        let mut changed = 0;
        for wb in wbs {
            if wb.kind == "EPHEMERAL" || wb.content.is_empty() {
                continue;
            }
            let key = format!("{}|{}", norm(&wb.branch), norm(&wb.content));
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            let is_factual = wb.content.chars().filter(|c| c.is_ascii_digit()).count() >= 2;
            let kind = if wb.kind == "IDENTITY_UPDATE" && is_factual { "BRANCH_UPDATE" } else { wb.kind.as_str() };
            match kind {
                "IDENTITY_UPDATE" => {
                    let id_now = norm(store.get_identity());
                    if id_now.contains(&norm(&wb.content)) || token_overlap_pct(&wb.content, store.get_identity()) >= 80 {
                        continue;
                    }
                    let merged = format!("{} {}", store.get_identity(), wb.content).trim().to_string();
                    store.set_identity(&merged, source, now);
                    changed += 1;
                }
                "NEW_BRANCH" => {
                    let name = if wb.branch.is_empty() { &wb.content } else { &wb.branch };
                    // "New" branch the store already knows, restating a fact
                    // it already holds anywhere: a no-op, not a summary bump.
                    if let Some(b) = store.get_branch(name) {
                        let c = norm(&wb.content);
                        let known = norm(b.summary.current()) == c
                            || b.details.iter().any(|d| {
                                let cur = d.current();
                                let body = cur.split_once("] ").map(|(_, t)| t).unwrap_or(cur);
                                norm(body) == c
                            });
                        if known {
                            continue;
                        }
                    }
                    store.create_branch(name, &wb.content, source, now);
                    changed += 1;
                }
                "BRANCH_UPDATE" | "DECISION" | "PREFERENCE_CHANGE" => {
                    let name = if wb.branch.is_empty() { "general" } else { &wb.branch };
                    let duplicate = store.get_branch(name).map_or(false, |b| {
                        let c = norm(&wb.content);
                        norm(b.summary.current()) == c
                            || b.details.iter().any(|d| {
                                let cur = d.current();
                                let body = cur.split_once("] ").map(|(_, t)| t).unwrap_or(cur);
                                norm(body) == c
                            })
                    });
                    if duplicate {
                        continue;
                    }
                    store.add_detail(name, &format!("[{kind}] {}", wb.content), source, now);
                    changed += 1;
                    // Supersede a narrow summary this detail restates. Broad
                    // topical summaries (low overlap) are left alone.
                    if let Some(b) = store.get_branch_mut(name) {
                        let summary = b.summary.current().to_string();
                        if !summary.is_empty()
                            && norm(&summary) != norm(&wb.content)
                            && token_overlap_pct(&summary, &wb.content) >= 60
                        {
                            b.update_summary(&wb.content, source, now);
                        }
                    }
                }
                _ => {}
            }
        }
        changed
    }

    /// Full write path for one exchange: classify, apply to the store, and
    /// ingest both turns into the driver index so the session itself becomes
    /// retrievable memory. Returns the write-backs that changed the store so
    /// callers can show the user what was remembered.
    pub fn write_back(
        &mut self,
        store: &mut MemoryStore,
        user_msg: &str,
        response: &str,
        timestamp: &str,
        now: Timestamp,
    ) -> Vec<WriteBack> {
        let mut wbs = self.classify_write_back(store, user_msg, response);
        Self::apply_write_backs(store, &wbs, now);
        wbs.retain(|w| w.kind != "EPHEMERAL" && !w.content.is_empty());
        if let Some(driver) = self.drivers.first_mut() {
            driver.ingest_turn("user", user_msg, timestamp);
            if !response.to_uppercase().contains("CONTEXT_NEEDED") {
                driver.ingest_turn("assistant", response, timestamp);
            }
        }
        wbs
    }

    /// Borrow the mounted driver, for surfaces that need to inspect or persist it.
    pub fn driver(&self) -> Option<&dyn MemoryIndexDriver> {
        self.drivers.first().map(|d| d.as_ref())
    }

    /// Mutably borrow the mounted driver, for surfaces that ingest turns
    /// themselves (e.g. to control write-back provenance).
    pub fn driver_mut(&mut self) -> Option<&mut Box<dyn MemoryIndexDriver>> {
        self.drivers.first_mut()
    }
}

/// What share of `a`'s informative tokens (lowercase, alphanumeric, longer
/// than three chars) also appear in `b`. 0 when `a` has none. Public so
/// surfaces that need "same thing, different words" (the daemon's fault
/// dedup) reuse this comparator instead of growing their own.
pub fn token_overlap_pct(a: &str, b: &str) -> usize {
    let tokens = |s: &str| -> std::collections::HashSet<String> {
        s.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() > 3)
            // Classifier voice, not information: "the user's cat" and "my
            // cat" must compare equal.
            .filter(|t| !matches!(*t, "user" | "users" | "they" | "their" | "them"))
            .map(String::from)
            .collect()
    };
    let ta = tokens(a);
    if ta.is_empty() {
        return 0;
    }
    let tb = tokens(b);
    ta.intersection(&tb).count() * 100 / ta.len()
}

/// Extract the fault topic from a `CONTEXT_NEEDED: <topic>` signal, or detect
/// softer "I don't have that" phrasings and return "unknown".
pub fn detect_page_fault(response: &str) -> Option<String> {
    let upper = response.to_uppercase();
    if let Some(pos) = upper.find("CONTEXT_NEEDED:") {
        let after = &response[pos + "CONTEXT_NEEDED:".len()..];
        let topic = after.lines().next().unwrap_or("").trim().trim_matches(['<', '>', '.', ' ']);
        return Some(if topic.is_empty() { "unknown".into() } else { topic.to_string() });
    }
    let low = response.to_lowercase();
    let soft = [
        "i don't have that information",
        "i don't have enough context",
        "not currently in my",
        "not loaded",
    ];
    if soft.iter().any(|p| low.contains(p)) {
        return Some("unknown".into());
    }
    None
}

fn to_slug(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>()
        .split('_')
        .filter(|s| !s.is_empty())
        .take(5)
        .collect::<Vec<_>>()
        .join("_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_explicit_page_fault() {
        assert_eq!(detect_page_fault("CONTEXT_NEEDED: adoption journey").as_deref(), Some("adoption journey"));
        assert_eq!(detect_page_fault("Sure, here is the answer."), None);
        assert_eq!(detect_page_fault("I don't have that information loaded.").as_deref(), Some("unknown"));
    }

    #[test]
    fn write_back_parsing_and_application() {
        let raw = "```json\n[{\"type\":\"NEW_BRANCH\",\"content\":\"training a pottery model\",\"branch\":\"pottery ai\"},\n{\"type\":\"EPHEMERAL\",\"content\":\"said hi\",\"branch\":\"\"},\n{\"type\":\"IDENTITY_UPDATE\",\"content\":\"user is a potter\",\"branch\":\"\"}]\n```";
        let wbs = parse_write_backs(raw);
        assert_eq!(wbs.len(), 3);
        let mut store = MemoryStore::new();
        let changed = Kernel::apply_write_backs(&mut store, &wbs, 1.0);
        assert_eq!(changed, 2, "EPHEMERAL must not change the store");
        assert!(store.get_branch("pottery ai").is_some());
        assert!(store.get_identity().contains("potter"));
        // Garbage input degrades to empty, never panics.
        assert!(parse_write_backs("no json here").is_empty());
        assert!(parse_write_backs("[{\"type\":42}]").is_empty());
    }

    #[test]
    fn write_backs_deduplicate() {
        let mut store = MemoryStore::new();
        let wbs = vec![
            WriteBack { kind: "BRANCH_UPDATE".into(), content: "Building a kernel in Rust.".into(), branch: "project".into() },
            WriteBack { kind: "PREFERENCE_CHANGE".into(), content: "building a kernel in Rust".into(), branch: "project".into() },
        ];
        // Same fact under two labels in one batch: one detail, not two.
        assert_eq!(Kernel::apply_write_backs(&mut store, &wbs, 1.0), 1);
        // Re-remembered on a later turn: no thirteenth copy.
        assert_eq!(Kernel::apply_write_backs(&mut store, &wbs[..1], 2.0), 0);
        assert_eq!(store.get_branch("project").unwrap().details.len(), 1);

        // Identity fragments already present don't re-append forever.
        store.set_identity("Name: Abhi", "user", 3.0);
        let id = vec![WriteBack { kind: "IDENTITY_UPDATE".into(), content: "name: abhi".into(), branch: String::new() }];
        assert_eq!(Kernel::apply_write_backs(&mut store, &id, 4.0), 0);
        assert_eq!(store.get_identity(), "Name: Abhi");
    }

    #[test]
    fn store_block_rides_into_the_prompt() {
        let mut k = Kernel::new(Ollama::new("m", "e"), KernelConfig::default());
        k.set_identity("Abhi");
        k.set_store_block("• dentist: appointment on October 21st");
        let (msgs, _) = k.prepare("when is my appointment?", &[]);
        let system = &msgs[0].content;
        assert!(system.contains("[IDENTITY]\nAbhi"));
        assert!(system.contains("[MEMORY TOPICS]\n• dentist: appointment on October 21st"));
        // Cleared block leaves no empty header behind.
        k.set_store_block("");
        let (msgs, _) = k.prepare("hi", &[]);
        assert!(!msgs[0].content.contains("[MEMORY TOPICS]"));
    }

    #[test]
    fn narrow_summary_superseded_by_correcting_detail() {
        let mut store = MemoryStore::new();
        store.create_branch("dentist", "Dentist appointment on October 14th", "wb", 1.0);
        let wbs = vec![WriteBack {
            kind: "BRANCH_UPDATE".into(),
            content: "dentist appointment moved to October 21st".into(),
            branch: "dentist".into(),
        }];
        Kernel::apply_write_backs(&mut store, &wbs, 2.0);
        let b = store.get_branch("dentist").unwrap();
        // The summary revised (copy-on-write, old version kept)...
        assert!(b.summary.current().contains("21st"), "summary: {}", b.summary.current());
        assert_eq!(b.summary.version_count(), 2);
        // ...and the detail landed too.
        assert!(b.details.iter().any(|d| d.current().contains("21st")));

        // A broad topical summary is NOT clobbered by one narrow detail.
        store.create_branch("aios", "A memory kernel for LLMs, written in Rust, defense in August", "wb", 3.0);
        let wbs = vec![WriteBack {
            kind: "DECISION".into(),
            content: "defense moved to September 1st".into(),
            branch: "aios".into(),
        }];
        Kernel::apply_write_backs(&mut store, &wbs, 4.0);
        assert!(store.get_branch("aios").unwrap().summary.current().contains("memory kernel"));
    }

    #[test]
    fn identity_rejects_value_facts_and_near_duplicates() {
        let mut store = MemoryStore::new();
        store.set_identity("Abhi, building a memory kernel", "user", 1.0);
        // A value-bearing "identity" fact reroutes to a branch detail.
        let wbs = vec![WriteBack {
            kind: "IDENTITY_UPDATE".into(),
            content: "My gym locker combination is 7-31-19".into(),
            branch: String::new(),
        }];
        Kernel::apply_write_backs(&mut store, &wbs, 2.0);
        assert!(!store.get_identity().contains("7-31-19"));
        assert!(store.get_branch("general").unwrap().details[0].current().contains("7-31-19"));
        // A rephrased near-duplicate does not re-append forever.
        store.set_identity("My cat is called Biscuit", "wb", 3.0);
        let id_before = store.get_identity().to_string();
        let wbs = vec![WriteBack {
            kind: "IDENTITY_UPDATE".into(),
            content: "the user's cat is called Biscuit".into(),
            branch: String::new(),
        }];
        Kernel::apply_write_backs(&mut store, &wbs, 4.0);
        assert_eq!(store.get_identity(), id_before);
    }

    #[test]
    fn slug_is_namespace_safe() {
        assert_eq!(to_slug("Adoption Journey!"), "adoption_journey");
        assert_eq!(to_slug("When did Caroline go?"), "when_did_caroline_go");
    }
}
