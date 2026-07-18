//! Shared daemon state: paths, settings, keys, and the cross-thread handles.
//!
//! One worker thread owns the kernel (see `worker.rs`); everything HTTP
//! threads may touch directly lives here behind mutexes: the journal, the
//! store (browse/correct/delete are store-only), settings, and the status
//! snapshot the worker refreshes after every turn.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use aios::store::MemoryStore;

use crate::journal::Journal;

/// Wall-clock seconds as the store's Timestamp. The store's versioning only
/// needs monotonic-ish ordering; real time makes "last updated" meaningful.
pub fn now_ts() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// --- Paths -----------------------------------------------------------------

#[derive(Clone)]
pub struct AiosDirs {
    pub root: PathBuf,
}

impl AiosDirs {
    /// `~/.aios/`, created on first run. Key file permissions are the
    /// caller's job (we only ever read keys).
    pub fn create() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let root = PathBuf::from(home).join(".aios");
        std::fs::create_dir_all(root.join("journal")).ok();
        AiosDirs { root }
    }

    pub fn store_path(&self) -> String {
        self.root.join("store.json").to_string_lossy().into_owned()
    }
    pub fn driver_path(&self) -> String {
        self.root.join("driver.json").to_string_lossy().into_owned()
    }
    pub fn journal_path(&self) -> String {
        self.root.join("journal").join("journal.jsonl").to_string_lossy().into_owned()
    }
    pub fn settings_path(&self) -> String {
        self.root.join("settings.json").to_string_lossy().into_owned()
    }
    pub fn keys_path(&self) -> String {
        self.root.join("keys").to_string_lossy().into_owned()
    }
    pub fn mcp_path(&self) -> String {
        self.root.join("mcp.json").to_string_lossy().into_owned()
    }
    pub fn media_dir(&self) -> std::path::PathBuf {
        let d = self.root.join("media");
        std::fs::create_dir_all(&d).ok();
        d
    }
}

/// `aios serve` users' companion/ state loads unchanged: if the daemon has
/// no store yet and a companion/ directory is present in the cwd, adopt it.
pub fn migrate_from_companion(dirs: &AiosDirs) {
    let store_dst = dirs.store_path();
    if std::path::Path::new(&store_dst).exists() {
        return;
    }
    if std::path::Path::new("companion/store.json").exists() {
        if std::fs::copy("companion/store.json", &store_dst).is_ok() {
            eprintln!("[migrate] adopted companion/store.json -> {store_dst}");
        }
    }
    let driver_dst = dirs.driver_path();
    if !std::path::Path::new(&driver_dst).exists()
        && std::path::Path::new("companion/driver.json").exists()
    {
        if std::fs::copy("companion/driver.json", &driver_dst).is_ok() {
            eprintln!("[migrate] adopted companion/driver.json -> {driver_dst}");
        }
    }
}

// --- Settings ----------------------------------------------------------------

pub const MODE_PERSISTENT: &str = "persistent";
pub const MODE_INCOGNITO: &str = "incognito";
pub const MODE_PAUSED: &str = "paused";

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// "ollama" | "claude" | "openai_compat" | "llamaserver"
    pub provider: String,
    /// The answer model for the active provider.
    pub model: String,
    /// OpenAI-compatible base URL (LM Studio, vLLM, OpenRouter, ...).
    pub base_url: String,
    /// Local model used for kernel-internal work (write-back classification).
    /// Always Ollama; memory formation stays local regardless of provider.
    pub local_model: String,
    /// Which brain forms memories: "local" (private, the local_model) or
    /// "answer" (the active provider — sharper classification, but hosted
    /// providers then see the exchange twice).
    pub memory_model: String,
    pub embed_model: String,
    pub num_ctx: usize,
    pub max_response_tokens: usize,
    pub temperature: f32,
    /// Cap on messages paged in per query. The ablation table says removing
    /// this cap is the single worst thing you can do; keep it modest.
    pub max_retrieved: usize,
    /// Session RAM budget (tokens) for the eviction window.
    pub window_budget: usize,
    /// "persistent" | "incognito" | "paused"
    pub privacy_mode: String,
    /// The /social conversation driver on/off.
    pub social_enabled: bool,
    /// Allow the model to raise WEB_NEEDED faults (searched by the daemon).
    pub web_enabled: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            provider: "ollama".into(),
            model: "llama3.1:8b".into(),
            base_url: "https://api.openai.com/v1".into(),
            local_model: "llama3.1:8b".into(),
            memory_model: "local".into(),
            embed_model: "nomic-embed-text".into(),
            num_ctx: 4096,
            max_response_tokens: 512,
            temperature: 0.0,
            max_retrieved: 30,
            window_budget: 1200,
            privacy_mode: MODE_PERSISTENT.into(),
            social_enabled: true,
            web_enabled: true,
        }
    }
}

impl Settings {
    pub fn load(dirs: &AiosDirs) -> Self {
        std::fs::read_to_string(dirs.settings_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, dirs: &AiosDirs) {
        if let Ok(s) = serde_json::to_string_pretty(self) {
            std::fs::write(dirs.settings_path(), s).ok();
        }
    }
}

// --- Keys ---------------------------------------------------------------------

/// Provider API keys. Read from `~/.aios/keys` (a JSON object like
/// {"anthropic": "sk-...", "openai": "..."}) with env-var fallback.
/// Never serialized back out, never logged, never returned over the API.
pub struct Keys {
    map: HashMap<String, String>,
}

impl Keys {
    pub fn load(dirs: &AiosDirs) -> Self {
        let mut map: HashMap<String, String> = std::fs::read_to_string(dirs.keys_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        for (name, env) in [("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")] {
            if !map.contains_key(name) {
                if let Ok(v) = std::env::var(env) {
                    if !v.is_empty() {
                        map.insert(name.into(), v);
                    }
                }
            }
        }
        Keys { map }
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.map.get(name).map(|s| s.as_str()).filter(|s| !s.is_empty())
    }

    pub fn present(&self) -> Vec<&str> {
        self.map.iter().filter(|(_, v)| !v.is_empty()).map(|(k, _)| k.as_str()).collect()
    }
}

// --- Worker requests ------------------------------------------------------------

/// SSE events out of a turn, as loose JSON values ({"t": "tok", ...}).
pub type EventTx = Sender<Value>;

pub enum Req {
    Turn {
        id: u64,
        text: String,
        /// data: URLs for the providers.
        images: Vec<String>,
        /// Saved media filenames for the journal.
        image_files: Vec<String>,
        cancel: Arc<AtomicBool>,
        events: EventTx,
    },
    Search { q: String, resp: Sender<Value> },
    KvSave { resp: Sender<Value> },
    KvRestore { resp: Sender<Value> },
    /// Settings were updated by an HTTP thread; re-read them.
    SettingsChanged,
}

// --- Shared -----------------------------------------------------------------------

pub struct Shared {
    pub tx: Sender<Req>,
    pub settings: Mutex<Settings>,
    pub journal: Mutex<Journal>,
    pub store: Mutex<MemoryStore>,
    pub dirs: AiosDirs,
    pub ui_dir: String,
    pub cancels: Mutex<HashMap<u64, Arc<AtomicBool>>>,
    pub turn_counter: AtomicU64,
    /// Snapshot the worker refreshes after boot and every turn, so a slow
    /// generation never blocks /v1/status.
    pub status: Mutex<Value>,
}

impl Shared {
    pub fn keys(&self) -> Keys {
        Keys::load(&self.dirs)
    }
}
