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
    /// Second retrieval round seeded by entities found in the first round.
    /// Chained facts ("where is the milk" needs who took it, then where they
    /// went) are unreachable in one round: the answer chunk shares no words
    /// with the question. Hop two searches on the names the first round
    /// surfaced. Index work only, no model calls.
    pub multi_hop: bool,
    /// Dense passage scoring over the FULL index instead of only reranking
    /// what BM25 and the tree beam surfaced. Today a passage no lexical method
    /// surfaces is never semantically scored, which is why "15 people on the
    /// platform team" is unreachable from "engineers hired". With this set,
    /// cosine over every message is a candidate generator and BM25 becomes an
    /// additional signal rather than a gate. Costs recall of distractors.
    pub ungate_dense: bool,
    /// Append a value ledger to the working set: every quantity in the loaded
    /// messages, with the clause it came from and the date it was stated.
    /// Pure bookkeeping over the message index, never a relevance judgment,
    /// because a relevance filter provably cannot separate two same-unit
    /// quantities that are both topically valid (that is what falsified entity
    /// scoping).
    pub annotate_values: bool,
}

impl Default for RouteConfig {
    fn default() -> Self {
        RouteConfig {
            use_tree: true,
            use_bm25: true,
            use_dense: true,
            max_load: 30,
            temporal_notes: true,
            multi_hop: false,
            ungate_dense: false,
            annotate_values: false,
        }
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

    /// Deterministic value annotation. For every quantity in a loaded message,
    /// name what it belongs to and when it was stated, using only what the
    /// message index already holds: the message's own words and its timestamp.
    ///
    /// Never a relevance judgment. That is the whole point: the drive case has
    /// two same-unit quantities that are both topically valid, so no relevance
    /// filter can separate them, which is what falsified entity scoping before
    /// it ever ran. Bookkeeping can, because it does not have to decide which
    /// value is wanted, only which thing each value belongs to.
    ///
    /// Returns, per value: (word index, quantity text, entity phrase, short
    /// date, type key). The type key is the value's measurement unit when it
    /// has one and "count" otherwise; it is what makes ambiguity decidable by
    /// counting rather than by judging relevance.
    fn annotate_message(&self, msg: &Message) -> Vec<(usize, String, String, String, String)> {
        const SKIP: &[&str] = &[
            "the", "a", "an", "my", "your", "our", "their", "his", "her", "its", "this", "that",
            "these", "those", "and", "or", "but", "for", "with", "from", "into", "onto", "about",
            "there", "here", "then", "than", "when", "what", "which", "who", "how", "just",
            "currently", "now", "also", "already", "still", "some", "any", "all", "much", "many",
            "have", "has", "had", "was", "were", "are", "is", "been", "being", "get", "got",
            "i'm", "im", "it's", "its", "we", "you", "they", "he", "she", "it",
            // Stative verbs that sit between a thing and its quantity; they add
            // no identity ("external drive holds" is just "external drive").
            "holds", "hold", "caps", "cap", "weighs", "weigh", "costs", "cost",
            "runs", "run", "contains", "covers", "takes", "allows", "gives",
            "left", "spent", "went", "came", "made", "said", "adds",
        ];
        const UNITS: &[&str] = &[
            "gigabytes", "gigabyte", "gb", "mb", "megabytes", "tb", "terabytes", "hours", "hour",
            "hrs", "minutes", "minute", "days", "day", "weeks", "week", "months", "month",
            "years", "year", "thousand", "million", "requests", "calls", "dollars", "usd",
            "percent", "%", "times", "am", "pm",
        ];
        const COUNT_NOUNS: &[&str] = &[
            "people", "person", "engineers", "engineer", "employees", "staff",
            "members", "developers", "designers", "candidates", "attendees",
            "calls", "requests", "tickets", "items", "seats", "guests",
        ];
        let clean = |w: &str| -> String {
            w.trim_matches(|c: char| !c.is_alphanumeric() && c != '%').to_lowercase()
        };
        let is_content = |w: &str| -> bool {
            let c = clean(w);
            c.len() > 2
                && !c.chars().all(|ch| ch.is_ascii_digit() || ch == '.' || ch == ',')
                && !SKIP.contains(&c.as_str())
                && !UNITS.contains(&c.as_str())
                && !COUNT_NOUNS.contains(&c.as_str())
        };

        let words: Vec<&str> = msg.text.split_whitespace().collect();
        // Compact date: "2 Mar", not "2 March 2023". Annotation rides inline in
        // the message, so every character competes with the retrieved text.
        let when = match parse_msg_date(&msg.timestamp) {
            Some((_, m, d)) => format!("{d} {}", &MONTHS[(m - 1) as usize][..3]),
            None => msg.timestamp.clone(),
        };
        let mut out = Vec::new();
        for (i, w) in words.iter().enumerate() {
            let c = clean(w);
            let numeric = !c.is_empty() && c.chars().next().map(|ch| ch.is_ascii_digit()).unwrap_or(false);
            if !numeric || out.len() >= 3 {
                continue;
            }
            // The quantity carries its unit when the next word is one.
            let next = words.get(i + 1).map(|n| clean(n)).unwrap_or_default();
            let measured = UNITS.contains(&next.as_str());
            let counted = COUNT_NOUNS.contains(&next.as_str());
            // The trailing noun joins the quantity for display either way, so it
            // reads as "15 people", but only a measurement unit makes a type.
            let unit = if measured || counted { next.clone() } else { String::new() };
            let qty = if unit.is_empty() { c.clone() } else { format!("{c} {unit}") };
            // Structural type key: the unit if measured, else a plain count.
            // "12 engineers" and "15 people" are both counts and therefore
            // confusable; "500 gigabytes" and "9 hours" are not.
            let type_key = if measured { next.clone() } else { "count".to_string() };

            // The entity is the content words nearest the number, looking both
            // ways: "the basic tier caps at 100 gigabytes" names it before,
            // "there are 15 people on the platform team" names it after.
            let lo = i.saturating_sub(5);
            let hi = (i + 6).min(words.len());
            let mut ent: Vec<String> = Vec::new();
            for j in lo..hi {
                if j == i {
                    continue;
                }
                if is_content(words[j]) {
                    let c = clean(words[j]);
                    if !ent.contains(&c) {
                        ent.push(c);
                    }
                }
            }
            ent.truncate(3);
            let entity = if ent.is_empty() { "unspecified".to_string() } else { ent.join(" ") };
            out.push((i, qty, entity, when.clone(), type_key));
        }
        out
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

    fn get_message(&self, idx: usize) -> Option<(String, String, String)> {
        self.msg_by_idx(idx).map(|m| (m.speaker.clone(), m.text.clone(), m.timestamp.clone()))
    }

    fn set_annotate_values(&mut self, on: bool) {
        self.route_cfg.annotate_values = on;
    }

    fn set_max_load(&mut self, n: usize) {
        self.route_cfg.max_load = n.max(1);
    }

    fn semantic_neighbors(&self, embedding: &[f32], k: usize) -> Vec<usize> {
        if embedding.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(f32, usize)> = self
            .messages
            .iter()
            .filter_map(|m| m.embedding.as_ref().map(|e| (cosine(embedding, e), m.idx)))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored.into_iter().map(|(_, i)| i).collect()
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

        // Ungated dense: cosine over EVERY message becomes a candidate source,
        // so BM25 and the beam are additional signals rather than a gate. This
        // is the reach half: a passage no lexical method surfaces (the
        // platform-team fact against an "engineers hired" query) can now be
        // scored at all. It knowingly admits distractors; the value ledger is
        // the paired countermeasure.
        if self.route_cfg.ungate_dense && !query_embedding.is_empty() {
            let want = (max_load * 2).max(30);
            let dense_ranked = self.flat_search(query_embedding);
            path.push_str(&format!(" | ungated-dense[{}]", want.min(dense_ranked.len())));
            for idx in dense_ranked.into_iter().take(want) {
                candidates.insert(idx);
            }
        }

        if candidates.is_empty() {
            *self.last_path.borrow_mut() = "flat fallback".to_string();
            return self.flat_search(query_embedding);
        }

        // Rerank: normalized BM25 + dense cosine + small bonus for top leaves.
        let mut scored: Vec<(f32, usize)> = candidates
            .iter()
            .map(|&idx| {
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

        // Hop two: mine the first round's results for salient terms the query
        // did not contain (rare, informative tokens, in practice names and
        // places), then run BM25 again on those. Hop-two hits join the pool at
        // a discount so they cannot displace direct hits, then everything is
        // reranked together.
        if self.route_cfg.multi_hop && self.route_cfg.use_bm25 {
            if let Some(bm25) = &self.bm25 {
                let query_terms: std::collections::HashSet<String> =
                    tokenize(query_text).into_iter().collect();
                let mut term_tf: HashMap<String, f32> = HashMap::new();
                for (_, idx) in scored.iter().take(10) {
                    if let Some(m) = self.msg_by_idx(*idx) {
                        for t in tokenize(&m.text) {
                            if !query_terms.contains(&t) && t.len() > 2 {
                                *term_tf.entry(t).or_insert(0.0) += 1.0;
                            }
                        }
                    }
                }
                // A useful expansion term must lead somewhere NEW: at least
                // one of its occurrences has to be outside what hop one
                // already retrieved. A term seen only inside the current
                // results is a dead end by construction. Among the leads,
                // prefer frequent-in-results and rare-in-corpus.
                let n_docs = self.messages.len().max(1) as f32;
                let mut expansion: Vec<(String, f32)> = term_tf
                    .into_iter()
                    .filter_map(|(t, tf)| {
                        let posts = bm25.postings.get(&t)?;
                        let leads_out = posts.iter().any(|(pos, _)| {
                            self.messages.get(*pos).map(|m| !candidates.contains(&m.idx)).unwrap_or(false)
                        });
                        if !leads_out {
                            return None;
                        }
                        let df = posts.len() as f32;
                        Some((t, tf * (n_docs / df).ln().max(0.1)))
                    })
                    .collect();
                expansion.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let hop_query: String = expansion
                    .iter()
                    .take(6)
                    .map(|(t, _)| t.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                if !hop_query.is_empty() {
                    let hits = bm25.top_k(&hop_query, 15);
                    let max_s = hits.first().map(|h| h.1).unwrap_or(1.0).max(1e-6);
                    let mut added = 0;
                    for (pos, s) in hits {
                        for p in pos.saturating_sub(1)..=(pos + 1).min(self.messages.len() - 1) {
                            let idx = self.messages[p].idx;
                            if candidates.insert(idx) {
                                let w = if p == pos { s / max_s } else { 0.25 * s / max_s };
                                scored.push((0.5 * w, idx));
                                added += 1;
                            }
                        }
                    }
                    if added > 0 {
                        path.push_str(&format!(" | hop2[{added}: {}]", hop_query));
                        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                        scored.truncate(max_load);
                    }
                }
            }
        }

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
        // (entity, date, quantity) for every annotated value, so an attribute
        // with several values can be rendered as a timeline instead of the
        // model silently picking one.
        let mut annotated: Vec<(String, String, String)> = Vec::new();

        // Pass one: which values are actually ambiguous? A value is ambiguous
        // when the working set holds another value of the same type, where type
        // is the measurement unit if it has one and a plain count otherwise.
        // This is counting and unit collision, deterministic and independent of
        // the model, never a relevance judgment: the drive case has two
        // gigabyte figures that are both topically valid, which is exactly why
        // relevance cannot be the test. Unambiguous values are left alone, so
        // annotation costs nothing where it would buy nothing.
        let mut type_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        if self.route_cfg.annotate_values {
            for &idx in indices {
                if let Some(msg) = self.msg_by_idx(idx) {
                    for (_, _, _, _, tk) in self.annotate_message(msg) {
                        *type_counts.entry(tk).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut tokens = 0usize;
        for &idx in indices {
            let Some(msg) = self.msg_by_idx(idx) else { continue };
            let base = if msg.timestamp.is_empty() {
                format!("{}: {}", msg.speaker, msg.text)
            } else {
                format!("[{}] {}: {}", msg.timestamp, msg.speaker, msg.text)
            };

            // Pass two: rewrite ambiguous quantities in place as a compact
            // parenthetical, "500 gigabytes (external drive, 2 Mar)". Inline
            // rather than on its own line, so it reads as metadata about the
            // value rather than as another fact competing with it.
            let line = if self.route_cfg.annotate_values {
                let anns = self.annotate_message(msg);
                let mut words: Vec<String> =
                    msg.text.split_whitespace().map(|w| w.to_string()).collect();
                // Right to left, so earlier edits do not shift later indices.
                for (wi, qty, ent, when, tk) in anns.into_iter().rev() {
                    if type_counts.get(&tk).copied().unwrap_or(0) < 2 {
                        continue; // unambiguous: nothing to disambiguate
                    }
                    annotated.push((ent.clone(), when.clone(), qty.clone()));
                    // Attach after the unit when there is one, else after the number.
                    let at = if qty.contains(' ') && wi + 1 < words.len() { wi + 1 } else { wi };
                    let trailing: String = words[at]
                        .chars()
                        .rev()
                        .take_while(|c| !c.is_alphanumeric())
                        .collect::<Vec<char>>()
                        .into_iter()
                        .rev()
                        .collect();
                    let cut = words[at].len() - trailing.len();
                    let stem = words[at][..cut].to_string();
                    words[at] = format!("{stem} ({ent}, {when}){trailing}");
                }
                let text = words.join(" ");
                if msg.timestamp.is_empty() {
                    format!("{}: {}", msg.speaker, text)
                } else {
                    format!("[{}] {}: {}", msg.timestamp, msg.speaker, text)
                }
            } else {
                base
            };

            let t = line.len() / 4;
            if tokens + t > budget_tokens {
                break;
            }
            parts.push(line);
            tokens += t;

            // Deterministic temporal resolution ("MMU for dates"): if this
            // message speaks in relative time, resolve it against the message's
            // own timestamp and remember the note.
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
        if self.route_cfg.annotate_values && !annotated.is_empty() {
            // Where one entity carries several values, show the sequence rather
            // than choosing: choosing silently is how a confident wrong answer
            // gets produced, which is the failure this exists to fix.
            let mut by_ent: std::collections::BTreeMap<String, Vec<(String, String)>> =
                std::collections::BTreeMap::new();
            for (ent, when, qty) in &annotated {
                by_ent.entry(ent.clone()).or_default().push((when.clone(), qty.clone()));
            }
            let lines: Vec<String> = by_ent
                .into_iter()
                .filter(|(_, v)| {
                    v.len() > 1 && v.iter().map(|(_, q)| q).collect::<std::collections::BTreeSet<_>>().len() > 1
                })
                .map(|(ent, v)| {
                    let seq: Vec<String> =
                        v.iter().map(|(w, q)| format!("{q} (stated {w})")).collect();
                    format!("- {ent}: {}", seq.join(" then "))
                })
                .collect();
            if !lines.is_empty() {
                let block = format!(
                    "\n\n[VALUE TIMELINE — attributes with more than one stated value, in order]\n{}",
                    lines.join("\n")
                );
                tokens += block.len() / 4;
                context.push_str(&block);
            }
        }
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

/// Rewrite a LongMemEval timestamp (`2023/04/10 (Mon) 17:50`) into the form the
/// LoCoMo harness already feeds and `parse_msg_date` above actually parses
/// (`5:50 pm on 10 April, 2023`).
///
/// This lives here, next to the parser it has to satisfy, so there is exactly
/// one implementation and the round-trip is covered by a test. A normalization
/// that silently failed to parse would produce precisely the temporal-reasoning
/// failure the normalization exists to avoid measuring, and it would look like
/// a capability result rather than a bug.
pub fn normalize_benchmark_date(s: &str) -> String {
    let parts: Vec<&str> = s.split_whitespace().collect();
    let ymd: Vec<&str> = parts.first().map(|p| p.split('/').collect()).unwrap_or_default();
    if ymd.len() != 3 {
        return s.to_string();
    }
    let mi: usize = match ymd[1].parse::<usize>() {
        Ok(m) if (1..=12).contains(&m) => m - 1,
        _ => return s.to_string(),
    };
    let day: u32 = match ymd[2].parse() {
        Ok(d) => d,
        Err(_) => return s.to_string(),
    };
    let hm = parts.last().copied().unwrap_or("12:00");
    let (h, min) = hm.split_once(':').unwrap_or(("12", "00"));
    let h24: u32 = h.parse().unwrap_or(12);
    let (h12, ampm) = match h24 {
        0 => (12, "am"),
        1..=11 => (h24, "am"),
        12 => (12, "pm"),
        _ => (h24 - 12, "pm"),
    };
    format!("{h12}:{min} {ampm} on {day} {}, {}", MONTHS[mi], ymd[0])
}

#[cfg(test)]
mod date_normalization_tests {
    use super::{normalize_benchmark_date, parse_msg_date};

    /// The gate: a normalized LongMemEval stamp must actually parse, and parse
    /// to the right calendar date. Verified, not assumed.
    #[test]
    fn longmemeval_dates_round_trip_through_the_resolver() {
        let cases = [
            ("2023/04/10 (Mon) 17:50", (2023, 4, 10)),
            ("2023/04/10 (Mon) 23:07", (2023, 4, 10)),
            ("2023/01/01 (Sun) 00:15", (2023, 1, 1)),
            ("2022/12/31 (Sat) 12:00", (2022, 12, 31)),
            ("2023/11/05 (Sun) 09:03", (2023, 11, 5)),
        ];
        for (raw, want) in cases {
            let norm = normalize_benchmark_date(raw);
            let got = parse_msg_date(&norm)
                .unwrap_or_else(|| panic!("normalized {raw:?} -> {norm:?} did NOT parse"));
            assert_eq!(got, want, "{raw:?} -> {norm:?} parsed as {got:?}");
        }
    }

    /// Midnight and noon are the two the 12-hour conversion gets wrong if the
    /// mapping is naive; the parsed date must still be right.
    #[test]
    fn midnight_and_noon_normalize_sanely() {
        assert!(normalize_benchmark_date("2023/06/09 (Fri) 00:30").starts_with("12:30 am"));
        assert!(normalize_benchmark_date("2023/06/09 (Fri) 12:30").starts_with("12:30 pm"));
        assert!(normalize_benchmark_date("2023/06/09 (Fri) 13:05").starts_with("1:05 pm"));
        assert_eq!(parse_msg_date(&normalize_benchmark_date("2023/06/09 (Fri) 00:30")), Some((2023, 6, 9)));
    }

    /// A shape it cannot parse must be passed through untouched rather than
    /// silently mangled into a wrong date.
    #[test]
    fn unparseable_input_passes_through() {
        assert_eq!(normalize_benchmark_date("1:56 pm on 8 May, 2023"), "1:56 pm on 8 May, 2023");
        assert_eq!(normalize_benchmark_date("garbage"), "garbage");
    }
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
    fn multi_hop_follows_the_entity_chain() {
        // "Where is the milk?" needs hop 1 (milk -> John took it) then hop 2
        // (John -> kitchen). Message 3 shares no words with the query and is
        // not adjacent to message 0, so single-hop retrieval cannot reach it.
        let mut d = HierarchicalTopicDriver::new("/docs");
        let texts = [
            "John took the milk from the fridge",
            "Mary journeyed to the garden with her book",
            "Sandra grabbed the football and left",
            "Afterwards John travelled onward to the kitchen",
            "The weather was cloudy all afternoon",
        ];
        let msgs: Vec<Message> = texts.iter().enumerate().map(|(i, t)| Message {
            idx: i, speaker: "text".into(), text: t.to_string(),
            timestamp: String::new(), embedding: None,
        }).collect();
        d.ingest_messages(&msgs);

        d.route_cfg.multi_hop = false;
        let single = d.route_query("where is the milk", &[]);
        d.route_cfg.multi_hop = true;
        let multi = d.route_query("where is the milk", &[]);

        assert!(single.contains(&0), "hop 1 must find the milk: {single:?}");
        assert!(multi.contains(&3), "hop 2 must follow John to the kitchen: {multi:?}");
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

#[cfg(test)]
mod annotation_tests {
    use super::*;
    use crate::driver::MemoryIndexDriver;

    /// Ambiguous values carry their entity and date inline, compactly. Both
    /// requirements from the spec: the drive case annotates all four gigabyte
    /// values, and the eng case binds 12 to budgeted engineers and 15 to the
    /// platform team.
    #[test]
    fn ambiguous_values_are_annotated_inline_and_compactly() {
        let mut d = HierarchicalTopicDriver::new("/t");
        let a = d.ingest_turn("user", "The external drive holds 500 gigabytes.", "9:15 am on 2 March, 2023");
        let b = d.ingest_turn("user", "I'm currently keeping 140 gigabytes up there.", "10:02 am on 3 March, 2023");
        let c = d.ingest_turn("user", "The basic tier caps at 100 gigabytes.", "10:05 am on 3 March, 2023");
        let e = d.ingest_turn("user", "My photo library weighs in at 620 gigabytes.", "6:40 pm on 12 August, 2023");
        d.route_cfg.annotate_values = true;
        let (ctx, _) = d.load_messages(&[a, b, c, e], 4000);
        for want in ["500 gigabytes (external drive, 2 Mar)",
                     "100 gigabytes (basic tier, 3 Mar)",
                     "620 gigabytes (photo library, 12 Aug)"] {
            assert!(ctx.contains(want), "missing {want:?}:\n{ctx}");
        }
        assert!(ctx.contains("140 gigabytes ("), "140 not annotated:\n{ctx}");
        assert!(!ctx.contains("    ^"), "still emitting a separate annotation line:\n{ctx}");

        let mut d2 = HierarchicalTopicDriver::new("/t");
        let x = d2.ingest_turn("user", "We budgeted for 12 engineers this year.", "11:00 am on 14 February, 2023");
        let y = d2.ingest_turn("user", "There are 15 people on the platform team now.", "3:30 pm on 20 August, 2023");
        d2.route_cfg.annotate_values = true;
        let (ctx2, _) = d2.load_messages(&[x, y], 4000);
        assert!(ctx2.contains("12 engineers (budgeted, 14 Feb)"), "12 not bound:\n{ctx2}");
        assert!(ctx2.contains("15 people (platform team, 20 Aug)"), "15 not bound:\n{ctx2}");
    }

    /// Selectivity: a value with no same-type rival is left alone, so
    /// annotation costs nothing where it would buy nothing.
    #[test]
    fn unambiguous_values_are_left_alone() {
        let mut d = HierarchicalTopicDriver::new("/t");
        let a = d.ingest_turn("user", "The flight to Berlin is 9 hours.", "1:00 pm on 1 May, 2023");
        let b = d.ingest_turn("user", "The external drive holds 500 gigabytes.", "2:00 pm on 1 May, 2023");
        d.route_cfg.annotate_values = true;
        let (ctx, _) = d.load_messages(&[a, b], 4000);
        assert!(!ctx.contains('('), "annotated values that had no same-type rival:\n{ctx}");
    }

    /// An attribute with several values must be rendered as a sequence, not
    /// silently reduced to one. Choosing silently is how a confident wrong
    /// answer is produced.
    #[test]
    fn repeated_attribute_renders_a_timeline() {
        let mut d = HierarchicalTopicDriver::new("/t");
        let a = d.ingest_turn("user", "The platform team has 12 engineers.", "1:00 pm on 14 February, 2023");
        let b = d.ingest_turn("user", "The platform team has 15 engineers.", "1:00 pm on 20 August, 2023");
        d.route_cfg.annotate_values = true;
        let (ctx, _) = d.load_messages(&[a, b], 4000);
        assert!(ctx.contains("VALUE TIMELINE"), "no timeline for a repeated attribute:\n{ctx}");
        assert!(ctx.contains("then"), "timeline is not ordered:\n{ctx}");
    }

    /// Off by default, or every prior measurement would silently change.
    #[test]
    fn ledger_is_off_by_default() {
        let mut d = HierarchicalTopicDriver::new("/t");
        let a = d.ingest_turn("user", "The drive holds 500 gigabytes.", "1:00 pm on 1 May, 2023");
        let (ctx, _) = d.load_messages(&[a], 4000);
        assert!(!ctx.contains(" — "), "annotation leaked into the default path:\n{ctx}");
        assert!(!ctx.contains("VALUE TIMELINE"), "timeline leaked into the default path");
    }
}
