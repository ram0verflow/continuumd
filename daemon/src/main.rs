//! `aios-daemon`: the long-lived localhost service that owns the kernel, the
//! store, the journal, and all provider connections. The desktop app, a CLI,
//! a VS Code extension, and a browser extension are all thin clients of the
//! same memory.
//!
//! There are no sessions anywhere in this binary: one user, one timeline,
//! one memory that outlives every process and every model.

mod api;
mod bedrock;
mod calc;
mod journal;
mod mcp;
mod providers;
mod state;
mod websearch;
mod worker;

use std::net::TcpListener;
use std::sync::atomic::AtomicU64;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use aios::store::MemoryStore;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let port: u16 = flag(&args, "--port").and_then(|v| v.parse().ok()).unwrap_or(4310);
    let ui_dir = flag(&args, "--ui").unwrap_or_else(|| "app/dist".to_string());

    let dirs = state::AiosDirs::create();
    state::migrate_from_companion(&dirs);

    let settings = state::Settings::load(&dirs);
    settings.save(&dirs); // materialize defaults on first run
    let journal = journal::Journal::open(&dirs.journal_path());
    let store = MemoryStore::load(&dirs.store_path()).unwrap_or_default();

    let (tx, rx) = mpsc::channel();
    let shared = Arc::new(state::Shared {
        tx,
        settings: Mutex::new(settings),
        journal: Mutex::new(journal),
        store: Mutex::new(store),
        dirs,
        ui_dir,
        cancels: Mutex::new(Default::default()),
        turn_counter: AtomicU64::new(0),
        status: Mutex::new(serde_json::Value::Null),
    });

    {
        let shared = shared.clone();
        thread::spawn(move || worker::run(rx, shared));
    }

    let listener = TcpListener::bind(("127.0.0.1", port)).unwrap_or_else(|e| {
        eprintln!("cannot bind 127.0.0.1:{port}: {e}");
        std::process::exit(1);
    });
    println!("aios daemon up: http://localhost:{port}  (state in ~/.aios/)");

    for stream in listener.incoming().flatten() {
        let shared = shared.clone();
        thread::spawn(move || api::handle(stream, shared));
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}
