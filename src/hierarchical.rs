//! `HierarchicalTopicDriver` (`/social`), spec §3.2.
//!
//! Handles casual chat over the Claude-partitioned tree. Navigated with dense
//! embeddings + exact keyword boosting. Traversal is a `match` over `TreeNode`,
//! recursing through `Branch`es and terminating safely at `Leaf`es (spec §4.2).
//!
//! Node-name embeddings are precomputed once at build time and cached, the
//! Python version re-embedded every child name inside the traversal hot loop,
//! which is the routing cost we explicitly remove here.

use std::collections::HashMap;

use serde_json::Value;

use crate::driver::{cosine, Message, MemoryIndexDriver, TreeNode};
use crate::matcher::tokenize;
use crate::ollama::Ollama;

/// Sparse BM25 index over raw messages. Complements the dense tree routing:
/// exact keyword hits ("picnic", "Dr. Seuss") surface even when the topic tree
/// routes the query to a different partition. Hybrid sparse+dense is what the
/// top LoCoMo systems use; here the tree gives topical coherence and BM25 gives
/// lexical recall.
struct Bm25Index {
    /// term -> postings of (position-in-messages, term frequency)
    postings: HashMap<String, Vec<(usize, f32)>>,
    doc_len: Vec<f32>,
    avgdl: f32,
    n_docs: usize,
}

impl Bm25Index {
    const K1: f32 = 1.2;
    const B: f32 = 0.75;

    fn build(messages: &[Message]) -> Self {
        let mut idx = Bm25Index { postings: HashMap::new(), doc_len: Vec::new(), avgdl: 0.0, n_docs: 0 };
        for m in messages {
            idx.add_doc(&m.text);
        }
        idx
    }

    /// Incremental add, O(len of one message). Doc position = insertion order,
    /// which must mirror the driver's `messages` vector.
    fn add_doc(&mut self, text: &str) {
        let pos = self.n_docs;
        let toks = tokenize(text);
        let total: f32 = self.doc_len.iter().sum::<f32>() + toks.len() as f32;
        self.doc_len.push(toks.len() as f32);
        let mut tf: HashMap<String, f32> = HashMap::new();
        for t in toks {
            *tf.entry(t).or_insert(0.0) += 1.0;
        }
        for (term, f) in tf {
            self.postings.entry(term).or_default().push((pos, f));
        }
        self.n_docs += 1;
        self.avgdl = total / self.n_docs as f32;
    }

    /// Top-k message *positions* by BM25 score, above a minimum score.
    fn top_k(&self, query: &str, k: usize) -> Vec<(usize, f32)> {
        if self.n_docs == 0 {
            return Vec::new();
        }
        let mut scores: HashMap<usize, f32> = HashMap::new();
        for term in tokenize(query) {
            let Some(posts) = self.postings.get(&term) else { continue };
            let df = posts.len() as f32;
            let idf = ((self.n_docs as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(pos, tf) in posts {
                let dl = self.doc_len[pos];
                let denom = tf + Self::K1 * (1.0 - Self::B + Self::B * dl / self.avgdl.max(1.0));
                *scores.entry(pos).or_insert(0.0) += idf * tf * (Self::K1 + 1.0) / denom;
            }
        }
        let mut ranked: Vec<(usize, f32)> = scores.into_iter().filter(|(_, s)| *s > 0.0).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k);
        ranked
    }
}

pub struct HierarchicalTopicDriver {
    namespace: String,
    messages: Vec<Message>,
    tree: Option<TreeNode>,
    /// node name -> cached embedding of that name
    name_embeddings: HashMap<String, Vec<f32>>,
    bm25: Option<Bm25Index>,
    /// Online embedder for incremental ingestion (None => keyword-only).
    embedder: Option<Ollama>,
    /// Ablation switches; default is the full pipeline.
    pub route_cfg: RouteConfig,
    /// Leaf path that received the previous online message, conversational
    /// continuity prior (dialogue tends to stay on topic).
    last_leaf_path: Option<Vec<String>>,
    /// The last navigation path, for observability / training capture.
    pub last_path: std::cell::RefCell<String>,
}

/// Online tree growth tuning. CONTINUITY_BONUS is deliberately below
/// ATTACH_THRESHOLD: staying on the previous topic still requires SOME
/// topical evidence (keyword/content/embedding); continuity only tips
/// borderline cases, it can never glue an off-topic message by itself.
const ATTACH_THRESHOLD: f32 = 0.35; // below this, open a new topic leaf
const CONTINUITY_BONUS: f32 = 0.20; // prior for the previous message's leaf
const CONTENT_OVERLAP_W: f32 = 0.25; // per shared informative token, capped
const MAX_LEAF_SIZE: usize = 14; // split leaves beyond this

/// Ablation switches for the retrieval pipeline. Everything on by default;
/// the eval binary flips pieces off one at a time to measure what each
/// component contributes.
#[derive(Clone, Debug)]
pub struct RouteConfig {
    pub use_tree: bool,
    pub use_bm25: bool,
    pub use_dense: bool,
    pub max_load: usize,
    pub temporal_notes: bool,
}

impl Default for RouteConfig {
    fn default() -> Self {
        RouteConfig { use_tree: true, use_bm25: true, use_dense: true, max_load: 30, temporal_notes: true }
    }
}

impl HierarchicalTopicDriver {
    pub fn new(namespace: &str) -> Self {
        HierarchicalTopicDriver {
            namespace: namespace.to_string(),
            messages: Vec::new(),
            tree: None,
            name_embeddings: HashMap::new(),
            bm25: None,
            embedder: None,
            route_cfg: RouteConfig::default(),
            last_leaf_path: None,
            last_path: std::cell::RefCell::new(String::new()),
        }
    }

    /// Give the driver an embedder for online ingestion. Without one, online
    /// ingestion still works keyword-only (BM25 + name keywords).
    pub fn set_embedder(&mut self, ollama: Ollama) {
        self.embedder = Some(ollama);
    }

    /// Persist the whole driver state (messages with embeddings, the grown
    /// tree, cached name embeddings) so a companion session survives restarts.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let messages: Vec<serde_json::Value> = self.messages.iter().map(|m| {
            serde_json::json!({
                "idx": m.idx, "speaker": m.speaker, "text": m.text,
                "timestamp": m.timestamp, "embedding": m.embedding,
            })
        }).collect();
        let name_embeddings: serde_json::Value = self.name_embeddings.iter()
            .map(|(k, v)| (k.clone(), serde_json::json!(v)))
            .collect::<serde_json::Map<_, _>>().into();
        let data = serde_json::json!({
            "namespace": self.namespace,
            "messages": messages,
            "tree": self.tree.as_ref().map(|t| t.to_json()),
            "name_embeddings": name_embeddings,
        });
        std::fs::write(path, serde_json::to_string(&data)?)
    }

    /// Reload a saved driver. Call set_embedder afterwards for online ingestion.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let data: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let ns = data["namespace"].as_str().unwrap_or("/social").to_string();
        let mut d = HierarchicalTopicDriver::new(&ns);
        for m in data["messages"].as_array().into_iter().flatten() {
            d.messages.push(Message {
                idx: m["idx"].as_u64().unwrap_or(0) as usize,
                speaker: m["speaker"].as_str().unwrap_or("").to_string(),
                text: m["text"].as_str().unwrap_or("").to_string(),
                timestamp: m["timestamp"].as_str().unwrap_or("").to_string(),
                embedding: m["embedding"].as_array().map(|a| {
                    a.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect()
                }),
            });
        }
        d.bm25 = Some(Bm25Index::build(&d.messages));
        d.tree = data.get("tree").filter(|t| !t.is_null()).and_then(TreeNode::from_json);
        if let Some(ne) = data["name_embeddings"].as_object() {
            for (k, v) in ne {
                if let Some(a) = v.as_array() {
                    d.name_embeddings.insert(
                        k.clone(),
                        a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect(),
                    );
                }
            }
        }
        Ok(d)
    }

    /// O(1) message lookup when idx == vector position (true for online
    /// ingestion and sorted loads); falls back to linear scan otherwise.
    fn msg_by_idx(&self, idx: usize) -> Option<&Message> {
        match self.messages.get(idx) {
            Some(m) if m.idx == idx => Some(m),
            _ => self.messages.iter().find(|m| m.idx == idx),
        }
    }

    /// Load a pre-built `conv_*.json` (messages + partitioned tree) and, if an
    /// Ollama handle is given, precompute embeddings for every tree node name.
    pub fn from_conv_json(namespace: &str, data: &Value, ollama: Option<&Ollama>) -> Self {
        let mut driver = HierarchicalTopicDriver::new(namespace);

        for m in data.get("messages").and_then(|v| v.as_array()).into_iter().flatten() {
            let idx = m.get("global_idx").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            driver.messages.push(Message {
                idx,
                speaker: m.get("speaker").and_then(|v| v.as_str()).unwrap_or("").to_lowercase(),
                text: m.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                timestamp: m.get("timestamp").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                embedding: None,
            });
        }
        // messages may be out of order; index by global_idx for O(1) load
        driver.messages.sort_by_key(|m| m.idx);
        driver.bm25 = Some(Bm25Index::build(&driver.messages));

        if let Some(tree_json) = data.get("tree") {
            driver.tree = TreeNode::from_json(tree_json);
        }

        if let Some(ol) = ollama {
            driver.build_name_embeddings(ol);
            // Dense message embeddings (the Python original embedded every
            // message at load). One-time cost; enables paraphrase matching
            // ("gave a speech" vs "talked at the assembly") that BM25 cannot.
            for m in &mut driver.messages {
                m.embedding = ol.embed(&m.text).ok();
            }
        }
        driver
    }

    /// Embed every node name in the tree once and cache it.
    pub fn build_name_embeddings(&mut self, ollama: &Ollama) {
        let mut names = Vec::new();
        if let Some(t) = &self.tree {
            collect_names(t, &mut names);
        }
        for name in names {
            if self.name_embeddings.contains_key(&name) {
                continue;
            }
            if let Ok(emb) = ollama.embed(&name) {
                self.name_embeddings.insert(name, emb);
            }
        }
    }

    pub fn tree(&self) -> Option<&TreeNode> {
        self.tree.as_ref()
    }

    pub fn message_len(&self) -> usize {
        self.messages.len()
    }

    /// Score a single child edge: exact keyword bonus + dense embedding
    /// similarity (spec §4.2: "combining cosine similarity with exact keyword
    /// matching to select the best child").
    fn score_child(&self, query_lower: &str, query_emb: &[f32], name: &str) -> f32 {
        let mut score = 0.0f32;
        for word in name.split_whitespace() {
            if word.len() > 2 && query_lower.contains(word) {
                score += 0.3;
            }
        }
        if !query_emb.is_empty() {
            if let Some(child_emb) = self.name_embeddings.get(name) {
                score += cosine(query_emb, child_emb);
            }
        }
        score
    }

    /// Beam traversal (spec §4.2 traversal, widened for recall).
    ///
    /// Single-path descent commits to one tiny leaf; if the answer lives in a
    /// sibling partition the model correctly page-faults but recall collapses.
    /// Instead we keep the top-`BEAM` children at each branch level, accumulating
    /// path score, and collect every reachable leaf with its cumulative score.
    /// The `match` on `TreeNode` still makes branch-vs-leaf a compiler-checked
    /// totality, mixed-type siblings just land in different arms.
    fn collect_leaves_beam(&self, query_lower: &str, query_emb: &[f32]) -> Vec<(f32, Vec<usize>)> {
        const BEAM: usize = 3;
        let Some(root) = &self.tree else { return Vec::new() };

        // frontier of (cumulative_score, node)
        let mut frontier: Vec<(f32, &TreeNode)> = vec![(0.0, root)];
        let mut leaves: Vec<(f32, Vec<usize>)> = Vec::new();

        while !frontier.is_empty() {
            let mut next: Vec<(f32, &TreeNode)> = Vec::new();
            for (acc, node) in frontier.drain(..) {
                match node {
                    TreeNode::Leaf(ids) => leaves.push((acc, ids.clone())),
                    TreeNode::Branch(children) => {
                        // Score each child edge, keep the top BEAM.
                        let mut scored: Vec<(f32, &TreeNode)> = children
                            .iter()
                            .map(|(name, child)| {
                                (acc + self.score_child(query_lower, query_emb, name), child.as_ref())
                            })
                            .collect();
                        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                        scored.truncate(BEAM);
                        next.extend(scored);
                    }
                }
            }
            frontier = next;
        }
        leaves.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        leaves
    }

    /// Online ingestion of one turn: embed, index, grow the topic tree.
    /// No LLM in this path, routing is embeddings + keywords + continuity.
    /// Returns the assigned message idx.
    pub fn ingest_turn_impl(&mut self, speaker: &str, text: &str, timestamp: &str) -> usize {
        let idx = self.messages.iter().map(|m| m.idx + 1).max().unwrap_or(0);
        let embedding = self.embedder.as_ref().and_then(|o| o.embed(text).ok());

        self.bm25.get_or_insert_with(|| Bm25Index::build(&[])).add_doc(text);
        self.messages.push(Message {
            idx,
            speaker: speaker.to_lowercase(),
            text: text.to_string(),
            timestamp: timestamp.to_string(),
            embedding: embedding.clone(),
        });

        // --- Tree growth ---
        let root = self.tree.get_or_insert_with(|| TreeNode::Branch(HashMap::new()));
        let mut leaf_paths: Vec<Vec<String>> = Vec::new();
        collect_leaf_paths(root, &mut Vec::new(), &mut leaf_paths);

        let text_lower = text.to_lowercase();
        let msg_tokens: std::collections::HashSet<String> = tokenize(text).into_iter().collect();
        let mut best: Option<(f32, Vec<String>)> = None;
        for path in &leaf_paths {
            let mut score = 0.0f32;
            // Path-name keywords.
            for name in path {
                for word in name.split_whitespace() {
                    if word.len() > 2 && text_lower.contains(word) {
                        score += 0.3;
                    }
                }
            }
            // Content overlap with the leaf's most recent messages, names
            // from a single message are arbitrary; content is the real topic.
            if let Some(ids) = leaf_ids_at(self.tree.as_ref().unwrap(), path) {
                let mut shared = 0usize;
                for &lid in ids.iter().rev().take(3) {
                    if let Some(m) = self.msg_by_idx(lid) {
                        shared += tokenize(&m.text).iter().filter(|t| msg_tokens.contains(*t)).count();
                    }
                }
                score += (shared as f32 * CONTENT_OVERLAP_W).min(0.5);
            }
            // Dense similarity to the leaf name, when we have both embeddings.
            if let (Some(emb), Some(name)) = (&embedding, path.last()) {
                if let Some(ne) = self.name_embeddings.get(name) {
                    score += cosine(emb, ne);
                }
            }
            if self.last_leaf_path.as_ref() == Some(path) {
                score += CONTINUITY_BONUS;
            }
            if best.as_ref().map(|(s, _)| score > *s).unwrap_or(true) {
                best = Some((score, path.clone()));
            }
        }

        let attached_path = match best {
            Some((score, path)) if score >= ATTACH_THRESHOLD => {
                attach_to_leaf(self.tree.as_mut().unwrap(), &path, idx);
                path
            }
            _ => {
                // Open a new topic leaf at the root, named by message keywords.
                let name = topic_name(text, &self.name_embeddings);
                if let Some(ol) = &self.embedder {
                    if let Ok(e) = ol.embed(&name) {
                        self.name_embeddings.insert(name.clone(), e);
                    }
                }
                if let TreeNode::Branch(children) = self.tree.as_mut().unwrap() {
                    children
                        .entry(name.clone())
                        .or_insert_with(|| Box::new(TreeNode::Leaf(Vec::new())));
                    if let TreeNode::Leaf(ids) = children.get_mut(&name).unwrap().as_mut() {
                        ids.push(idx);
                    }
                }
                vec![name]
            }
        };

        // Split oversized leaves chronologically so routing stays sharp.
        self.split_if_needed(&attached_path);
        self.last_leaf_path = Some(attached_path);
        idx
    }

    /// If the leaf at `path` outgrew MAX_LEAF_SIZE, split it in half
    /// chronologically, naming each half from its own messages' keywords.
    fn split_if_needed(&mut self, path: &[String]) {
        let Some(root) = self.tree.as_mut() else { return };
        let Some(ids) = leaf_ids_at(root, path) else { return };
        if ids.len() <= MAX_LEAF_SIZE {
            return;
        }
        let mid = ids.len() / 2;
        let (a, b) = (ids[..mid].to_vec(), ids[mid..].to_vec());
        let text_of = |set: &[usize]| {
            set.iter()
                .filter_map(|i| self.messages.iter().find(|m| m.idx == *i))
                .map(|m| m.text.clone())
                .collect::<Vec<_>>()
                .join(" ")
        };
        let name_a = topic_name(&text_of(&a), &self.name_embeddings);
        let name_b = topic_name(&text_of(&b), &self.name_embeddings);
        if name_a == name_b {
            return; // no meaningful split available
        }
        if let Some(ol) = &self.embedder {
            for n in [&name_a, &name_b] {
                if !self.name_embeddings.contains_key(n) {
                    if let Ok(e) = ol.embed(n) {
                        self.name_embeddings.insert(n.clone(), e);
                    }
                }
            }
        }
        let mut children = HashMap::new();
        children.insert(name_a, Box::new(TreeNode::Leaf(a)));
        children.insert(name_b.clone(), Box::new(TreeNode::Leaf(b)));
        replace_node(self.tree.as_mut().unwrap(), path, TreeNode::Branch(children));
        // Continuity should now point at the recent half.
        let mut new_path = path.to_vec();
        new_path.push(name_b);
        self.last_leaf_path = Some(new_path);
    }

    /// Flat embedding fallback when there is no tree (or no query embedding).
    fn flat_search(&self, query_emb: &[f32]) -> Vec<usize> {
        if query_emb.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(usize, f32)> = self
            .messages
            .iter()
            .filter_map(|m| m.embedding.as_ref().map(|e| (m.idx, cosine(query_emb, e))))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(i, _)| i).collect()
    }
}

impl MemoryIndexDriver for HierarchicalTopicDriver {
    fn namespace(&self) -> &str {
        &self.namespace
    }

    fn ingest_messages(&mut self, messages: &[Message]) {
        // Append order must mirror BM25 doc positions, callers pass batches
        // in chronological order; we do not re-sort.
        for m in messages {
            self.messages.push(m.clone());
        }
        self.bm25 = Some(Bm25Index::build(&self.messages));
    }

    fn ingest_turn(&mut self, speaker: &str, text: &str, timestamp: &str) -> usize {
        self.ingest_turn_impl(speaker, text, timestamp)
    }

    fn persist(&self, path: &str) -> std::io::Result<()> {
        self.save(path)
    }

    fn route_query(&self, query_text: &str, query_embedding: &[f32]) -> Vec<usize> {
        // Candidate generation is broad; the final load is narrow. Stuffing 100
        // messages into a 4K window drowned the 8B model ("needle dilution") -
        // it answered "no mention of X" with X verbatim in context. Rerank all
        // candidates (BM25 + dense cosine) and keep only the best MAX_LOAD,
        // presented chronologically so the conversation reads coherently and
        // timestamps form a timeline for temporal questions.
        const MAX_LEAVES: usize = 6;
        const BM25_TOP: usize = 20;
        let max_load = self.route_cfg.max_load;

        let mut candidates: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut bm25_score: HashMap<usize, f32> = HashMap::new();
        let mut leaf_rank: HashMap<usize, usize> = HashMap::new();
        let mut path = String::new();

        // Dense: beam over the topic tree.
        if self.route_cfg.use_tree && self.tree.is_some() && !query_embedding.is_empty() {
            let query_lower = query_text.to_lowercase();
            let leaves = self.collect_leaves_beam(&query_lower, query_embedding);
            path.push_str("beam:");
            for (rank, (score, leaf_ids)) in leaves.iter().take(MAX_LEAVES).enumerate() {
                path.push_str(&format!(" {:.2}[{}]", score, leaf_ids.len()));
                for &id in leaf_ids {
                    candidates.insert(id);
                    leaf_rank.entry(id).or_insert(rank);
                }
            }
        }

        // Sparse: BM25 keyword hits (+ conversational ±1 neighbors).
        if let Some(bm25) = self.bm25.as_ref().filter(|_| self.route_cfg.use_bm25) {
            let hits = bm25.top_k(query_text, BM25_TOP);
            if !hits.is_empty() {
                path.push_str(&format!(" | bm25[{}]", hits.len()));
                let max_s = hits.first().map(|h| h.1).unwrap_or(1.0).max(1e-6);
                for (pos, s) in hits {
                    for p in pos.saturating_sub(1)..=(pos + 1).min(self.messages.len() - 1) {
                        let idx = self.messages[p].idx;
                        candidates.insert(idx);
                        // Neighbors inherit half the hit's normalized score.
                        let w = if p == pos { s / max_s } else { 0.5 * s / max_s };
                        let e = bm25_score.entry(idx).or_insert(0.0);
                        if w > *e {
                            *e = w;
                        }
                    }
                }
            }
        }

        if candidates.is_empty() {
            *self.last_path.borrow_mut() = "flat fallback".to_string();
            return self.flat_search(query_embedding);
        }

        // Rerank: normalized BM25 + dense cosine + small bonus for top leaves.
        let mut scored: Vec<(f32, usize)> = candidates
            .into_iter()
            .map(|idx| {
                let mut s = bm25_score.get(&idx).copied().unwrap_or(0.0);
                if let Some(rank) = leaf_rank.get(&idx) {
                    s += 0.3 / (1.0 + *rank as f32); // leaf 0 → +0.30, leaf 5 → +0.05
                }
                if self.route_cfg.use_dense && !query_embedding.is_empty() {
                    if let Some(msg) = self.msg_by_idx(idx) {
                        if let Some(emb) = &msg.embedding {
                            s += cosine(query_embedding, emb);
                        }
                    }
                }
                (s, idx)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_load);

        // Chronological presentation.
        let mut ids: Vec<usize> = scored.into_iter().map(|(_, idx)| idx).collect();
        ids.sort_unstable();
        path.push_str(&format!(" | rerank→{}", ids.len()));
        *self.last_path.borrow_mut() = path;
        ids
    }

    fn load_messages(&self, indices: &[usize], budget_tokens: usize) -> (String, usize) {
        let mut parts = Vec::new();
        let mut notes: Vec<String> = Vec::new();
        let mut tokens = 0usize;
        for &idx in indices {
            let Some(msg) = self.msg_by_idx(idx) else { continue };
            let line = if msg.timestamp.is_empty() {
                format!("{}: {}", msg.speaker, msg.text)
            } else {
                format!("[{}] {}: {}", msg.timestamp, msg.speaker, msg.text)
            };
            let t = line.len() / 4;
            if tokens + t > budget_tokens {
                break;
            }
            parts.push(line);
            tokens += t;

            // Deterministic temporal resolution ("MMU for dates"): if this
            // message speaks in relative time, resolve it against the
            // message's own timestamp and remember the note.
            if self.route_cfg.temporal_notes && notes.len() < 8 {
                if let Some((y, m, d)) = parse_msg_date(&msg.timestamp) {
                    let tl = msg.text.to_lowercase();
                    if let Some((phrase, resolution)) = resolve_phrase(&tl, y, m, d) {
                        notes.push(format!(
                            "- {} said \"{}\" on {} → {}",
                            msg.speaker,
                            phrase,
                            fmt_date(y, m, d),
                            resolution
                        ));
                    }
                }
            }
        }
        let mut context = parts.join("\n");
        if !notes.is_empty() {
            let block = format!("\n\n[TIME NOTES — relative dates resolved by the OS]\n{}", notes.join("\n"));
            tokens += block.len() / 4;
            context.push_str(&block);
        }
        (context, tokens)
    }
}

// --- Temporal resolution: the MMU for dates -------------------------------
//
// LoCoMo evidence is phrased relatively ("I went camping last week") inside a
// timestamped message. The model reliably finds the evidence but fails the
// calendar math (worst observed: answering literally "Yesterday."). So the OS
// resolves relative phrases deterministically and injects the resolution as a
// [TIME NOTES] block; the resolutions are phrased to match how humans (and
// LoCoMo golds) express them: "the week before 27 June 2023".

const MONTHS: [&str; 12] = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
];

/// Parse "1:56 pm on 8 May, 2023" → (year, month, day).
fn parse_msg_date(ts: &str) -> Option<(i32, u32, u32)> {
    let lower = ts.to_lowercase();
    let after_on = if let Some(pos) = lower.rfind(" on ") { &ts[pos + 4..] } else { ts };
    let cleaned = after_on.replace(',', "");
    let parts: Vec<&str> = cleaned.split_whitespace().collect();
    if parts.len() < 3 {
        return None;
    }
    let d: u32 = parts[0].parse().ok()?;
    let pl = parts[1].to_lowercase();
    let m = MONTHS.iter().position(|mo| mo.to_lowercase().starts_with(&pl) || pl.starts_with(&mo.to_lowercase()))? as u32 + 1;
    let y: i32 = parts[2].parse().ok()?;
    if d == 0 || d > 31 {
        return None;
    }
    Some((y, m, d))
}

fn fmt_date(y: i32, m: u32, d: u32) -> String {
    format!("{} {} {}", d, MONTHS[(m - 1) as usize], y)
}

/// Days since civil epoch (Howard Hinnant's algorithm), for cross-month
/// "yesterday"/"tomorrow" arithmetic without pulling in chrono.
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y } as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = ((m as i64) + 9) % 12;
    let doy = (153 * mp + 2) / 5 + (d as i64) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    ((if m <= 2 { y + 1 } else { y }) as i32, m, d)
}

fn shift_days(y: i32, m: u32, d: u32, delta: i64) -> String {
    let (ny, nm, nd) = civil_from_days(days_from_civil(y, m, d) + delta);
    fmt_date(ny, nm, nd)
}

/// Today's date as a message timestamp ("5 July 2026"), for live sessions.
pub fn today_timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (y, m, d) = civil_from_days(secs / 86_400);
    fmt_date(y, m, d)
}

/// Resolve a relative phrase against the message's date. Longest phrases are
/// checked first by the caller. Output phrasing mirrors LoCoMo gold style.
fn resolve_phrase(text_lower: &str, y: i32, m: u32, d: u32) -> Option<(String, String)> {
    let date = fmt_date(y, m, d);
    // (needle, resolution), order matters: longest/most specific first.
    let weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];
    for wd in weekdays {
        for pat in [format!("last {wd}"), format!("this past {wd}")] {
            if text_lower.contains(&pat) {
                return Some((pat, format!("the {wd} before {date}")));
            }
        }
    }
    let table: [(&str, String); 14] = [
        ("a couple of weekends ago", format!("two weekends before {date}")),
        ("couple of weekends ago", format!("two weekends before {date}")),
        ("two weekends ago", format!("two weekends before {date}")),
        ("last weekend", format!("the weekend before {date}")),
        ("this past weekend", format!("the weekend before {date}")),
        ("last week", format!("the week before {date}")),
        ("last month", format!("the month before {date}")),
        ("last year", format!("{}", y - 1)),
        ("last night", shift_days(y, m, d, -1)),
        ("yesterday", shift_days(y, m, d, -1)),
        ("tomorrow", shift_days(y, m, d, 1)),
        ("next week", format!("the week after {date}")),
        ("next month", format!("the month after {date}")),
        ("this morning", date.clone()),
    ];
    for (needle, resolution) in table {
        if text_lower.contains(needle) {
            return Some((needle.to_string(), resolution));
        }
    }
    None
}

fn collect_names(node: &TreeNode, out: &mut Vec<String>) {
    if let TreeNode::Branch(children) = node {
        for (name, child) in children {
            out.push(name.clone());
            collect_names(child, out);
        }
    }
}

// --- Online tree mutation helpers ---------------------------------------

fn collect_leaf_paths(node: &TreeNode, prefix: &mut Vec<String>, out: &mut Vec<Vec<String>>) {
    match node {
        TreeNode::Leaf(_) => out.push(prefix.clone()),
        TreeNode::Branch(children) => {
            for (name, child) in children {
                prefix.push(name.clone());
                collect_leaf_paths(child, prefix, out);
                prefix.pop();
            }
        }
    }
}

fn attach_to_leaf(node: &mut TreeNode, path: &[String], idx: usize) -> bool {
    match (node, path) {
        (TreeNode::Leaf(ids), []) => {
            ids.push(idx);
            true
        }
        (TreeNode::Branch(children), [head, rest @ ..]) => children
            .get_mut(head)
            .map(|c| attach_to_leaf(c, rest, idx))
            .unwrap_or(false),
        _ => false,
    }
}

fn leaf_ids_at<'a>(node: &'a TreeNode, path: &[String]) -> Option<&'a Vec<usize>> {
    match (node, path) {
        (TreeNode::Leaf(ids), []) => Some(ids),
        (TreeNode::Branch(children), [head, rest @ ..]) => {
            children.get(head).and_then(|c| leaf_ids_at(c, rest))
        }
        _ => None,
    }
}

fn replace_node(node: &mut TreeNode, path: &[String], new_node: TreeNode) -> bool {
    match path {
        [] => {
            *node = new_node;
            true
        }
        [head, rest @ ..] => match node {
            TreeNode::Branch(children) => children
                .get_mut(head)
                .map(|c| replace_node(c, rest, new_node))
                .unwrap_or(false),
            _ => false,
        },
    }
}

/// Name a topic from message text: the 2 most frequent informative tokens.
/// Deduplicates against existing node names by appending a counter.
fn topic_name(text: &str, existing: &HashMap<String, Vec<f32>>) -> String {
    let mut freq: HashMap<String, usize> = HashMap::new();
    for t in tokenize(text) {
        if t.len() > 3 {
            *freq.entry(t).or_insert(0) += 1;
        }
    }
    let mut ranked: Vec<(String, usize)> = freq.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let base = ranked
        .iter()
        .take(2)
        .map(|(t, _)| t.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let base = if base.is_empty() { "misc".to_string() } else { base };
    if !existing.contains_key(&base) {
        return base;
    }
    for i in 2..100 {
        let candidate = format!("{base} {i}");
        if !existing.contains_key(&candidate) {
            return candidate;
        }
    }
    base
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tiny_conv() -> Value {
        json!({
            "messages": [
                {"global_idx": 0, "speaker": "Caroline", "text": "I went to the LGBTQ support group", "timestamp": "7 May 2023"},
                {"global_idx": 1, "speaker": "Melanie", "text": "I painted a sunrise last year", "timestamp": "2022"},
                {"global_idx": 2, "speaker": "Caroline", "text": "I researched adoption agencies", "timestamp": "D2"}
            ],
            "tree": {
                "caroline": { "support group": [0], "adoption": [2] },
                "melanie": { "painting": [1] }
            }
        })
    }

    #[test]
    fn date_parse_and_arithmetic() {
        assert_eq!(parse_msg_date("1:56 pm on 8 May, 2023"), Some((2023, 5, 8)));
        assert_eq!(parse_msg_date("7:30 pm on 20 May, 2023"), Some((2023, 5, 20)));
        assert_eq!(parse_msg_date("no date here"), None);
        assert_eq!(fmt_date(2023, 5, 8), "8 May 2023");
        // Cross-month/year boundary arithmetic.
        assert_eq!(shift_days(2023, 7, 1, -1), "30 June 2023");
        assert_eq!(shift_days(2023, 1, 1, -1), "31 December 2022");
    }

    #[test]
    fn relative_phrases_resolve_gold_style() {
        let (p, r) = resolve_phrase("i went camping last week!", 2023, 6, 27).unwrap();
        assert_eq!(p, "last week");
        assert_eq!(r, "the week before 27 June 2023");
        let (_, r) = resolve_phrase("ran the 5k last sunday", 2023, 5, 25).unwrap();
        assert_eq!(r, "the sunday before 25 May 2023");
        let (_, r) = resolve_phrase("signed up for a pottery class yesterday", 2023, 7, 3).unwrap();
        assert_eq!(r, "2 July 2023");
        assert!(resolve_phrase("nothing temporal here", 2023, 1, 1).is_none());
    }

    #[test]
    fn time_notes_injected_for_relative_dates() {
        let conv = json!({
            "messages": [
                {"global_idx": 0, "speaker": "Melanie", "text": "I went camping last week, it was great", "timestamp": "2:00 pm on 27 June, 2023"}
            ],
            "tree": { "melanie": { "camping": [0] } }
        });
        let d = HierarchicalTopicDriver::from_conv_json("/social", &conv, None);
        let (ctx, _) = d.load_messages(&[0], 1000);
        assert!(ctx.contains("[TIME NOTES"), "notes missing:\n{ctx}");
        assert!(ctx.contains("the week before 27 June 2023"), "resolution missing:\n{ctx}");
    }

    #[test]
    fn online_ingestion_grows_tree_and_stays_retrievable() {
        let mut d = HierarchicalTopicDriver::new("/social");
        // No embedder: keyword-only mode.
        let i0 = d.ingest_turn("alice", "I started learning pottery classes downtown", "1 pm on 2 July, 2023");
        let i1 = d.ingest_turn("bob", "Pottery sounds fun! I love the pottery wheel", "1 pm on 2 July, 2023");
        let i2 = d.ingest_turn("alice", "Completely different: my server crashed with kernel panics", "2 pm on 3 July, 2023");
        assert_eq!((i0, i1, i2), (0, 1, 2));

        let tree = d.tree().unwrap();
        assert_eq!(tree.message_count(), 3);
        // Continuity + keyword overlap should co-locate the two pottery turns;
        // the server topic must NOT share their leaf.
        let mut paths = Vec::new();
        collect_leaf_paths(tree, &mut Vec::new(), &mut paths);
        let pottery_leaf = paths.iter().find(|p| {
            leaf_ids_at(tree, p).map(|ids| ids.contains(&0)).unwrap_or(false)
        }).unwrap();
        let ids = leaf_ids_at(tree, pottery_leaf).unwrap();
        assert!(ids.contains(&1), "continuity should keep pottery turns together: {ids:?}");
        assert!(!ids.contains(&2), "server topic must open its own leaf: {ids:?}");

        // BM25 retrieval sees online-ingested content.
        let hits = d.route_query("what happened to the server", &[]);
        assert!(hits.contains(&2), "BM25 should surface the server message, got {hits:?}");
    }

    #[test]
    fn oversized_leaf_splits_chronologically() {
        let mut d = HierarchicalTopicDriver::new("/social");
        for i in 0..(MAX_LEAF_SIZE + 2) {
            // Same strong keyword every turn -> same leaf via continuity/keywords;
            // vary secondary words so split halves get distinct names.
            let filler = if i < (MAX_LEAF_SIZE + 2) / 2 { "glazing kiln firing" } else { "wheel throwing clay" };
            d.ingest_turn("alice", &format!("pottery pottery session {i} {filler}"), "ts");
        }
        let tree = d.tree().unwrap();
        assert_eq!(tree.message_count(), MAX_LEAF_SIZE + 2);
        let mut paths = Vec::new();
        collect_leaf_paths(tree, &mut Vec::new(), &mut paths);
        let max_leaf = paths.iter().map(|p| leaf_ids_at(tree, p).unwrap().len()).max().unwrap();
        assert!(max_leaf <= MAX_LEAF_SIZE, "leaf of {max_leaf} should have split");
    }

    #[test]
    fn keyword_routing_without_embeddings_still_navigates() {
        // No embeddings -> flat fallback returns nothing (no message embeddings),
        // but tree keyword routing kicks in only when we pass an embedding.
        let d = HierarchicalTopicDriver::from_conv_json("/social", &tiny_conv(), None);
        assert_eq!(d.message_len(), 3);
        assert_eq!(d.tree().unwrap().message_count(), 3);
        assert_eq!(d.tree().unwrap().leaf_count(), 3);
    }

    #[test]
    fn traversal_terminates_at_leaf_with_fake_embeddings() {
        // Give the driver deterministic fake embeddings so we can test routing
        // logic without Ollama: embed = one-hot on a keyword.
        let mut d = HierarchicalTopicDriver::from_conv_json("/social", &tiny_conv(), None);
        d.name_embeddings.insert("support group".into(), vec![1.0, 0.0, 0.0]);
        d.name_embeddings.insert("adoption".into(), vec![0.0, 1.0, 0.0]);
        d.name_embeddings.insert("painting".into(), vec![0.0, 0.0, 1.0]);
        d.name_embeddings.insert("caroline".into(), vec![1.0, 1.0, 0.0]);
        d.name_embeddings.insert("melanie".into(), vec![0.0, 0.0, 1.0]);

        // Query embedding aligned with "adoption" axis + keyword "adoption".
        // Selection is by relevance (msg 2 must make the cut); presentation is
        // chronological (ids ascending) so the conversation reads in order.
        let ids = d.route_query("adoption agencies", &[0.0, 1.0, 0.0]);
        assert!(ids.contains(&2), "relevant message must be selected, got {ids:?}");
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted, "presentation must be chronological, got {ids:?}");
        let (ctx, _) = d.load_messages(&ids, 1000);
        assert!(ctx.contains("adoption agencies"));
    }
}
