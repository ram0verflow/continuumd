//! AIOS, a Memory Operating System kernel for LLMs.
//!
//! `main` is a small demo/inspection CLI. The real surface is the library
//! modules (kernel, drivers, store, eviction) and the `eval` binary.

use std::env;
use std::io::{BufRead, Write};

use aios::eviction::ContextWindow;
use aios::hierarchical::HierarchicalTopicDriver;
use aios::kernel::{Kernel, KernelConfig};
use aios::ollama::{ChatMessage, Ollama};
use aios::store::MemoryStore;

fn main() {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("info");

    match cmd {
        "info" => info(),
        "tree" => describe_tree(args.get(2).map(|s| s.as_str())),
        "ask" => ask(&args[2..]),
        "chat" => chat(args.iter().any(|a| a == "--kv")),
        "serve" => {
            let port = flag_value(&args, "--port").and_then(|v| v.parse().ok()).unwrap_or(3210);
            let model = flag_value(&args, "--model").unwrap_or_else(|| "llama3.1:8b".to_string());
            aios::server::run(port, &model);
        }
        "daemon" => {
            // The daemon lives in its own crate so this library stays
            // dependency-free and embeddable.
            eprintln!("the daemon is its own binary:");
            eprintln!("  cargo run --release -p aios-daemon    # http://localhost:4310, state in ~/.aios/");
        }
        _ => {
            eprintln!("usage: aios [info | tree <conv.json> | ask <question> | chat [--kv] | serve [--port P] [--model M] | daemon]");
        }
    }
}

fn flag_value(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

const KV_SESSION_FILE: &str = "chat_session.bin";

/// The live OS loop: multi-turn REPL where session messages accumulate in a
/// ContextWindow, evict under pressure (demotion → store archive), and the
/// evicted summary rides along so old turns stay reachable. With `--kv`, the
/// session's ATTENTION STATES persist across restarts: restored from disk on
/// startup, saved on exit, infinite chat with instant memory.
fn chat(use_kv: bool) {
    let path = "data/conv_0.json";
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read conv")).expect("json");

    let ollama = Ollama::new("llama3.1:8b", "nomic-embed-text");
    eprintln!("[aios chat: building name embeddings…]");
    let mut driver = HierarchicalTopicDriver::from_conv_json("/social", &data, Some(&ollama));
    driver.set_embedder(ollama.clone());

    let mut kernel = Kernel::new(ollama, KernelConfig::default());
    kernel.mount(Box::new(driver));

    if use_kv {
        let server = aios::llamaserver::LlamaServer::new(8080);
        if !server.healthy() {
            eprintln!("[--kv] llama-server not reachable on :8080. Start it first:");
            eprintln!("  llama-server -m <gguf-or-ollama-blob> --port 8080 -c 8192 \\");
            eprintln!("    --slots --slot-save-path kv_slots/ -np 1");
            return;
        }
        kernel.set_kv_backend(server);
        match kernel.restore_kv(KV_SESSION_FILE) {
            Ok(n) => eprintln!("[--kv] restored {n} tokens of attention state from disk"),
            Err(_) => eprintln!("[--kv] no saved session, starting cold"),
        }
    }

    // Session RAM: small budget so eviction is visible in a demo session.
    let mut store = MemoryStore::new();
    let mut window = ContextWindow::new(600, None);
    let mut turn = 0u64;

    eprintln!("[type a message; 'quit' exits; ':stat' shows pressure]");
    let stdin = std::io::stdin();
    loop {
        print!("> ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }
        if input == ":stat" {
            println!(
                "pressure={} used={}/{} slots={} evictions={} summary={}ch",
                window.pressure_level(),
                window.used_tokens(),
                window.budget_tokens,
                window.slots.len(),
                window.total_evictions,
                window.evicted_summary.len()
            );
            continue;
        }
        turn += 1;

        // Build session for the kernel: evicted summary first (as system
        // context), then whatever messages are still resident in RAM.
        let mut session: Vec<ChatMessage> = Vec::new();
        if !window.evicted_summary.is_empty() {
            session.push(ChatMessage::new("system", format!("[PREVIOUS CONTEXT] {}", window.evicted_summary)));
        }
        for slot in &window.slots {
            if let Some((role, content)) = slot.content.split_once(": ") {
                session.push(ChatMessage::new(role, content));
            }
        }

        let result = kernel.query(input, &session);
        println!("{}", result.response.trim());
        if result.page_faulted {
            eprintln!(
                "  [page fault → '{}'{}]",
                result.fault_topic,
                if result.fault_retried { ", retried OK" } else { ", unresolved" }
            );
        }

        // Write path: classify memory updates, apply to store, and ingest the
        // exchange into the driver so it becomes retrievable memory.
        let wrote = kernel.write_back(
            &mut store,
            input,
            result.response.trim(),
            &aios::hierarchical::today_timestamp(),
            turn as f64,
        );
        if !wrote.is_empty() {
            eprintln!("  [write-back: {} memory update(s)]", wrote.len());
        }

        // Load the turn into RAM; evict under pressure; demotions → archive.
        window.load_message("user", input, false);
        window.load_message("assistant", result.response.trim(), false);
        if window.pressure_level() != "OK" {
            let before = window.total_evictions;
            window.evict_messages(4);
            eprintln!(
                "  [pressure {} → evicted {} msgs, summary {} chars]",
                window.pressure_level(),
                window.total_evictions - before,
                window.evicted_summary.len()
            );
        }
        for (branch, role, content) in window.drain_demotions() {
            store.add_archive(&branch, &role, &content, turn as f64);
        }
    }
    if use_kv && kernel.has_kv_backend() {
        match kernel.save_kv(KV_SESSION_FILE) {
            Ok(s) => eprintln!(
                "[--kv] paged out {} tokens of attention state ({} MB) to disk",
                s.tokens,
                s.bytes / 1_048_576
            ),
            Err(e) => eprintln!("[--kv] save failed: {e}"),
        }
    }
    let s = store.stats();
    eprintln!("[session over: {} turns, {} archived, {} evictions]", turn, s.archive_entries, window.total_evictions);
}

fn info() {
    println!("AIOS: Memory Operating System for LLMs (Rust kernel)");
    println!("  kernel      : domain-agnostic, page-fault loop, VFS namespaces");
    println!("  drivers     : HierarchicalTopicDriver (/social)");
    println!("  store       : 4-level hierarchy, versioned, demotion-not-deletion");
    println!("  eviction    : token-pressure scoring, demote to archive");
    let ol = Ollama::new("llama3.1:8b", "nomic-embed-text");
    println!("  ollama      : {}", if ol.healthy() { "UP" } else { "DOWN" });
}

fn describe_tree(path: Option<&str>) {
    let path = path.unwrap_or("data/conv_0.json");
    let data: serde_json::Value = match std::fs::read_to_string(path) {
        Ok(s) => serde_json::from_str(&s).expect("valid json"),
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return;
        }
    };
    let driver = HierarchicalTopicDriver::from_conv_json("/social", &data, None);
    println!("messages: {}", driver.message_len());
    if let Some(t) = driver.tree() {
        println!("tree depth : {}", t.depth());
        println!("leaves     : {}", t.leaf_count());
        println!("in-tree msg: {}", t.message_count());
    }
}

fn ask(args: &[String]) {
    let question = args.join(" ");
    if question.is_empty() {
        eprintln!("usage: aios ask <question>");
        return;
    }
    let path = "data/conv_0.json";
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read conv")).expect("json");

    let ollama = Ollama::new("llama3.1:8b", "nomic-embed-text");
    eprintln!("[building name embeddings…]");
    let driver = HierarchicalTopicDriver::from_conv_json("/social", &data, Some(&ollama));

    let mut kernel = Kernel::new(ollama, KernelConfig::default());
    kernel.mount(Box::new(driver));

    let result = kernel.query(&question, &[]);
    println!("Q: {question}");
    println!("A: {}", result.response.trim());
    println!(
        "   [ns={} loaded={} budget={} fault={} retried={}]",
        result.namespace,
        result.messages_loaded,
        result.memory_budget_tokens,
        result.page_faulted,
        result.fault_retried
    );
}
