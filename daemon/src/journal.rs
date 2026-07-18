//! The journal: an append-only turn log, the timeline's source of truth.
//!
//! Every user and assistant turn plus markers for memory events, stored as
//! JSONL under `~/.aios/journal/`. Owned by the daemon, NEVER used for
//! retrieval (the drivers do retrieval). Timeline reads the journal; memory
//! search reads the kernel.
//!
//! Incognito turns are appended with `ephemeral: true` and purged on boot
//! and on privacy-mode exit.

use std::io::Write;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::state::now_ms;

#[derive(Clone, Serialize, Deserialize)]
pub struct Entry {
    pub id: u64,
    pub ts_ms: u64,
    /// "user" | "assistant" | "memory" | "evict" | "system"
    pub kind: String,
    pub text: String,
    #[serde(default)]
    pub meta: Value,
    #[serde(default)]
    pub ephemeral: bool,
}

pub struct Journal {
    path: String,
    entries: Vec<Entry>,
    next_id: u64,
}

impl Journal {
    /// Load the journal, dropping any ephemeral entries a crashed incognito
    /// session may have left behind (rewrites the file if it does).
    pub fn open(path: &str) -> Self {
        let mut entries: Vec<Entry> = Vec::new();
        let mut had_ephemeral = false;
        if let Ok(data) = std::fs::read_to_string(path) {
            for line in data.lines() {
                match serde_json::from_str::<Entry>(line) {
                    Ok(e) if e.ephemeral => had_ephemeral = true,
                    Ok(e) => entries.push(e),
                    Err(_) => {}
                }
            }
        }
        let next_id = entries.last().map(|e| e.id + 1).unwrap_or(1);
        let mut j = Journal { path: path.to_string(), entries, next_id };
        if had_ephemeral {
            j.rewrite();
        }
        j
    }

    pub fn append(&mut self, kind: &str, text: &str, meta: Value, ephemeral: bool) -> u64 {
        let entry = Entry {
            id: self.next_id,
            ts_ms: now_ms(),
            kind: kind.to_string(),
            text: text.to_string(),
            meta,
            ephemeral,
        };
        self.next_id += 1;
        if let Ok(line) = serde_json::to_string(&entry) {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.path) {
                let _ = writeln!(f, "{line}");
            }
        }
        let id = entry.id;
        self.entries.push(entry);
        id
    }

    /// Newest-first page for the timeline. `before` is an entry id (exclusive);
    /// 0 means "from the end".
    pub fn page(&self, before: u64, limit: usize) -> Vec<&Entry> {
        self.entries
            .iter()
            .rev()
            .filter(|e| before == 0 || e.id < before)
            .take(limit)
            .collect()
    }

    pub fn recent(&self, n: usize) -> Vec<&Entry> {
        let start = self.entries.len().saturating_sub(n);
        self.entries[start..].iter().collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Drop ephemeral entries from memory and disk (incognito exit).
    pub fn purge_ephemeral(&mut self) {
        let before = self.entries.len();
        self.entries.retain(|e| !e.ephemeral);
        if self.entries.len() != before {
            self.rewrite();
        }
    }

    fn rewrite(&mut self) {
        let mut out = String::new();
        for e in &self.entries {
            if let Ok(line) = serde_json::to_string(e) {
                out.push_str(&line);
                out.push('\n');
            }
        }
        std::fs::write(&self.path, out).ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> String {
        let p = std::env::temp_dir().join(format!("aios_journal_test_{name}.jsonl"));
        std::fs::remove_file(&p).ok();
        p.to_string_lossy().into_owned()
    }

    #[test]
    fn append_page_roundtrip() {
        let path = tmp("roundtrip");
        let mut j = Journal::open(&path);
        for i in 0..5 {
            j.append("user", &format!("msg {i}"), Value::Null, false);
        }
        let page = j.page(0, 2);
        assert_eq!(page.len(), 2);
        assert_eq!(page[0].text, "msg 4");
        let older = j.page(page[1].id, 10);
        assert_eq!(older.len(), 3);
        // Survives reload.
        let j2 = Journal::open(&path);
        assert_eq!(j2.len(), 5);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn ephemeral_purged_on_open() {
        let path = tmp("ephemeral");
        {
            let mut j = Journal::open(&path);
            j.append("user", "keep", Value::Null, false);
            j.append("user", "secret", Value::Null, true);
        }
        let j = Journal::open(&path);
        assert_eq!(j.len(), 1);
        assert_eq!(j.recent(10)[0].text, "keep");
        // The file itself no longer contains the secret.
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("secret"));
        std::fs::remove_file(&path).ok();
    }
}
