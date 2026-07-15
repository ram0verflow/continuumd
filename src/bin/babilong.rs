//! BABILong runner: needles hidden in book-length haystacks.
//!
//! Each sample's haystack is chunked into pseudo-messages and ingested with
//! BM25 only (no embeddings: embedding thousands of chunks per sample would
//! dominate runtime, and bAbI facts are lexically distinctive, so sparse
//! retrieval is the honest fit). The kernel then routes the question, loads
//! the top chunks, and the model answers. Scoring is exact containment of
//! the gold word, the standard for this benchmark.
//!
//! Usage: babilong <data/babilong_64k.jsonl> [--model M] [--limit N]

use std::time::Instant;

use aios::driver::{Message, MemoryIndexDriver};
use aios::hierarchical::HierarchicalTopicDriver;
use aios::kernel::{Kernel, KernelConfig};
use aios::ollama::Ollama;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "data/babilong_64k.jsonl".into());
    let mut model = "llama3.1:8b".to_string();
    let mut limit = usize::MAX;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { model = args.get(i + 1).cloned().unwrap_or(model); i += 2; }
            "--limit" => { limit = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(limit); i += 2; }
            _ => i += 1,
        }
    }

    let ollama = Ollama::new(&model, "nomic-embed-text");
    if !ollama.healthy() {
        eprintln!("ollama not reachable");
        std::process::exit(1);
    }

    let data = std::fs::read_to_string(&path).expect("read babilong jsonl");
    let mut per_task: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    let mut n = 0usize;
    let mut ok_total = 0usize;
    let mut lat: Vec<f64> = Vec::new();

    for line in data.lines() {
        if n >= limit {
            break;
        }
        let row: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let task = row["task"].as_str().unwrap_or("?").to_string();
        let input = row["input"].as_str().unwrap_or("");
        let question = row["question"].as_str().unwrap_or("");
        let target = row["target"].as_str().unwrap_or("");
        if input.is_empty() || target.is_empty() {
            continue;
        }
        n += 1;

        // Chunk the haystack into pseudo-messages, ~400 chars on sentence-ish
        // boundaries, and ingest without embeddings.
        let mut driver = HierarchicalTopicDriver::new("/docs");
        let mut msgs = Vec::new();
        let mut chunk = String::new();
        let mut idx = 0usize;
        for piece in input.split_inclusive(['.', '\n']) {
            chunk.push_str(piece);
            if chunk.len() >= 400 {
                msgs.push(Message { idx, speaker: "text".into(), text: std::mem::take(&mut chunk), timestamp: String::new(), embedding: None });
                idx += 1;
            }
        }
        if !chunk.trim().is_empty() {
            msgs.push(Message { idx, speaker: "text".into(), text: chunk, timestamp: String::new(), embedding: None });
        }
        let n_chunks = msgs.len();
        driver.ingest_messages(&msgs);

        let mut kernel = Kernel::new(ollama.clone(), KernelConfig::default());
        kernel.mount(Box::new(driver));

        let t0 = Instant::now();
        let result = kernel.query(question, &[]);
        let secs = t0.elapsed().as_secs_f64();
        lat.push(secs);

        let hit = result.response.to_lowercase().contains(&target.to_lowercase());
        let e = per_task.entry(task.clone()).or_insert((0, 0));
        e.1 += 1;
        if hit {
            e.0 += 1;
            ok_total += 1;
        }
        eprintln!(
            "[{n:>3}] {task} {} chunks={n_chunks} loaded={} {:.1}s | want {target} | got {}",
            if hit { "OK  " } else { "MISS" },
            result.messages_loaded,
            secs,
            result.response.replace('\n', " ").chars().take(60).collect::<String>()
        );
    }

    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("\n========== BABILONG ({path}) ==========");
    println!("model {model} | samples {n} | accuracy {ok_total}/{n} ({:.1}%)",
        100.0 * ok_total as f64 / n.max(1) as f64);
    for (t, (ok, tot)) in &per_task {
        println!("  {t}: {ok}/{tot}");
    }
    if !lat.is_empty() {
        println!("latency p50 {:.1}s p95 {:.1}s", lat[lat.len() / 2], lat[lat.len() * 95 / 100]);
    }
    println!("=========================================");
}
