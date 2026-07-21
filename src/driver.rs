//! The VFS driver layer.
//!
//! `MemoryIndexDriver` is the strict contract every memory system implements
//! (spec §4.1). `TreeNode` is the type-safe fix (spec §4.2) for the Python crash:
//! a node is *definitively* a `Branch` or a `Leaf`, so mixed-type siblings, the
//! thing that crashed `isinstance`-based traversal, are just a `match` arm here.

use std::collections::HashMap;

use serde_json::Value;

/// A raw conversation message, addressed by its global index.
#[derive(Clone, Debug)]
pub struct Message {
    pub idx: usize,
    pub speaker: String,
    pub text: String,
    pub timestamp: String,
    pub embedding: Option<Vec<f32>>,
}

/// The strict contract for a memory index (spec §4.1).
///
/// A driver owns a domain (a VFS namespace like `/social` or `/workspace`),
/// ingests messages into its own index, and routes a query to the message
/// indices it deems relevant. The kernel stays domain-agnostic and only ever
/// talks to drivers through this trait.
pub trait MemoryIndexDriver {
    /// The VFS namespace this driver owns, e.g. `/social`.
    fn namespace(&self) -> &str;

    /// Build/refresh the index from a batch of messages.
    fn ingest_messages(&mut self, messages: &[Message]);

    /// Online ingestion of a single conversation turn. The driver assigns the
    /// message id, updates its indexes incrementally, and grows its routing
    /// structure. Returns the assigned id.
    fn ingest_turn(&mut self, speaker: &str, text: &str, timestamp: &str) -> usize;

    /// Route a query to a set of message indices.
    ///
    /// `query_embedding` may be empty when embeddings are unavailable; a driver
    /// must degrade gracefully (keyword-only) in that case.
    fn route_query(&self, query_text: &str, query_embedding: &[f32]) -> Vec<usize>;

    /// Render the given message indices into a context block within a token
    /// budget (~4 chars/token). Returns (text, tokens_used).
    fn load_messages(&self, indices: &[usize], budget_tokens: usize) -> (String, usize);

    /// Persist driver state to disk, if the driver supports it. Default: no-op.
    fn persist(&self, _path: &str) -> std::io::Result<()> {
        Ok(())
    }

    /// Read one indexed message back out as (speaker, text, timestamp), for
    /// surfaces that show retrieval results next to their source turns.
    /// Default: not supported.
    fn get_message(&self, _idx: usize) -> Option<(String, String, String)> {
        None
    }

    /// Cap on how many messages one query may load. Default: not tunable.
    fn set_max_load(&mut self, _n: usize) {}

    /// Base retrieval by entity graph (issue #14): resolve the query's
    /// entities to nodes and return the messages they carry. Default: none.
    fn entity_route(&self, _query: &str) -> Vec<usize> { Vec::new() }

    /// Pure dense nearest neighbours over every indexed message, bypassing
    /// the candidate gate entirely. Exists for fault re-pages: the model
    /// names a gap in its own vocabulary, and when that shares no tokens
    /// with the stored mention, the lexical candidate pool never contains
    /// the target for the dense scorer to find. Default: unsupported.
    fn semantic_neighbors(&self, _embedding: &[f32], _k: usize) -> Vec<usize> {
        Vec::new()
    }
}

/// Type-safe partitioned tree (spec §4.2).
///
/// ```text
/// pub enum TreeNode {
///     Branch(HashMap<String, Box<TreeNode>>),
///     Leaf(Vec<usize>), // Message IDs
/// }
/// ```
///
/// The Python tree stored *mixed* children at one level (some dicts, some lists
/// of ints); dynamic `isinstance` checks blew up at the leaf boundary. Here each
/// child deserializes independently into the correct variant, so a branch whose
/// siblings are a mix of sub-branches and leaves is representable and safe.
#[derive(Clone, Debug)]
pub enum TreeNode {
    Branch(HashMap<String, Box<TreeNode>>),
    Leaf(Vec<usize>),
}

impl TreeNode {
    /// Serialize back to the same JSON shape `from_json` reads, so a grown
    /// tree can be persisted to disk and reloaded.
    pub fn to_json(&self) -> Value {
        match self {
            TreeNode::Leaf(ids) => Value::Array(ids.iter().map(|&i| Value::from(i as u64)).collect()),
            TreeNode::Branch(children) => {
                let mut map = serde_json::Map::new();
                for (name, child) in children {
                    map.insert(name.clone(), child.to_json());
                }
                Value::Object(map)
            }
        }
    }

    /// Parse a `serde_json::Value` from the Claude-partitioned tree.
    /// A JSON array -> `Leaf`; a JSON object -> `Branch`; recursively.
    pub fn from_json(v: &Value) -> Option<TreeNode> {
        match v {
            Value::Array(items) => {
                let ids = items.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect();
                Some(TreeNode::Leaf(ids))
            }
            Value::Object(map) => {
                let mut children = HashMap::new();
                for (k, child) in map {
                    if let Some(node) = TreeNode::from_json(child) {
                        children.insert(k.clone(), Box::new(node));
                    }
                }
                Some(TreeNode::Branch(children))
            }
            _ => None,
        }
    }

    /// Count all message ids reachable under this node.
    pub fn message_count(&self) -> usize {
        match self {
            TreeNode::Leaf(ids) => ids.len(),
            TreeNode::Branch(children) => children.values().map(|c| c.message_count()).sum(),
        }
    }

    /// Number of leaves under this node.
    pub fn leaf_count(&self) -> usize {
        match self {
            TreeNode::Leaf(_) => 1,
            TreeNode::Branch(children) => children.values().map(|c| c.leaf_count()).sum(),
        }
    }

    /// Max depth (a lone leaf has depth 1).
    pub fn depth(&self) -> usize {
        match self {
            TreeNode::Leaf(_) => 1,
            TreeNode::Branch(children) => {
                1 + children.values().map(|c| c.depth()).max().unwrap_or(0)
            }
        }
    }
}

/// Cosine similarity between two equal-length vectors. Returns 0 on mismatch.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn treenode_parses_mixed_siblings() {
        // This is exactly the shape that crashed the Python traversal:
        // one sibling is a sub-branch (object), the other is a leaf (array).
        let v = json!({
            "melanie": {
                "adoption journey": [1, 2, 3],
                "art projects": { "painting": [4, 5], "sculpture": [6] }
            }
        });
        let node = TreeNode::from_json(&v).unwrap();
        assert_eq!(node.message_count(), 6);
        assert_eq!(node.leaf_count(), 3);
        // melanie -> {adoption journey(leaf), art projects(branch)} : mixed, no panic.
        match &node {
            TreeNode::Branch(root) => match root.get("melanie").unwrap().as_ref() {
                TreeNode::Branch(mel) => {
                    assert!(matches!(mel.get("adoption journey").unwrap().as_ref(), TreeNode::Leaf(_)));
                    assert!(matches!(mel.get("art projects").unwrap().as_ref(), TreeNode::Branch(_)));
                }
                _ => panic!("melanie should be a branch"),
            },
            _ => panic!("root should be a branch"),
        }
    }

    #[test]
    fn cosine_basic() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0); // length mismatch
    }
}
