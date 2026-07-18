# AIOS Companion, Product Design Document

This document is written for a fresh Claude session that will build the
application. It assumes zero context. Read the whole thing before writing
code. The repository you are working in already contains the memory kernel;
your job is the application around it, and the framing matters:

**You are not building a chat app. You are building an operating system for
a continuous AI relationship.** The difference shows up in dozens of small
decisions: there are no sessions, no chat list, no "new chat" button, no
per-conversation state in the frontend. There is one user, one timeline,
one memory that outlives every process and every model.

## Mission

Build the first AI application where the user never starts a new
conversation. The assistant is one persistent intelligence that remembers
the user's life, projects, preferences, and history. The underlying model
(Claude, GPT, Gemini, local Llama) is replaceable at any moment without
losing continuity.

AIOS provides persistence. Models provide intelligence.

```
Traditional:  chat -> prompt -> LLM -> forget everything

AIOS:         persistent memory -> kernel -> working set -> any model
```

The LLM is stateless. The kernel owns state.

## Ground truth: what already exists in this repo

Do not rebuild any of this. The Rust crate `aios` contains a working kernel
and two memory drivers, benchmarked and documented in README.md.

The kernel (`src/kernel.rs`):
- `Kernel::new(ollama, config)`, `kernel.mount(Box<dyn MemoryIndexDriver>)`
- `kernel.query(user_message, session) -> QueryResult` runs the full loop:
  route memory, assemble `[MEMORY_BLOCK: /ns/...]` context, generate,
  intercept `CONTEXT_NEEDED:` page faults, re-page and retry once.
  QueryResult carries: response, page_faulted, fault_topic, fault_retried,
  messages_loaded, namespace, retrieval_ms, generation_ms.
- `kernel.prepare(...)` / `kernel.prepare_fault(...)` return the assembled
  message list without generating, for callers that stream generation
  themselves. `kernel.complete_messages(...)` is the non-streaming call.
- `kernel.write_back(store, user_msg, reply, timestamp, now) -> Vec<WriteBack>`
  classifies the exchange (IDENTITY_UPDATE / NEW_BRANCH / BRANCH_UPDATE /
  DECISION / PREFERENCE_CHANGE / EPHEMERAL), applies it to the store, and
  ingests both turns into the driver index. This is memory formation.
- `kernel.set_kv_backend(LlamaServer)` plus `save_kv` / `restore_kv` page
  attention states to and from disk when llama-server is running.

The conversation driver (`src/hierarchical.rs`, namespace `/social`):
- hybrid retrieval: topic tree beam + BM25 + dense embeddings, reranked,
  capped at 30 messages, presented chronologically. The cap matters: the
  ablation table in README.md shows removing it is the single worst thing
  you can do (0.449 -> 0.118).
- `ingest_turn(speaker, text, timestamp)` grows the index online, about
  30 ms per message, no LLM calls.
- deterministic date resolution: "last week" said on a timestamped message
  becomes a resolved [TIME NOTES] line. Do not reimplement date logic in
  the frontend.
- `save(path)` / `load(path)` persist the whole driver (messages,
  embeddings, tree).

The code driver (`src/codegraph.rs`, namespace `/workspace`): symbol
extraction plus BM25 with a relevance gate, `ingest_file(path, source)`.
Dense embeddings deliberately unused there.

A working prototype of the service already exists: `aios serve`
(`src/server.rs`) is a single-file web companion with SSE streaming,
write-back, eviction, persistence to `companion/`, and a memory panel.
Treat it as the reference implementation of the conversation flow, then
supersede it.

Also on disk: a fine-tuned local model (`aios-ft-r2-full` in Ollama) that
emits the page-fault token reliably; the base models `llama3.1:8b`,
`mistral:7b`, `phi3:mini`; and `nomic-embed-text` for embeddings.

Known behaviors to respect, all measured (see README.md):
- Small local models decorate recalled facts (added a year to a date once).
- The tuned model is better on conversation, worse on book-style text.
  Different volumes may want different answer models.
- Chained multi-hop questions are a known retrieval weakness. A second
  blind retrieval hop exists behind a flag and is off because it measured
  net negative. Do not turn it on by default.
- KV state files are model-locked and about 125 KB per token. Text is the
  source of truth; KV is a per-model cache tier, never the store.

## Architecture

```
Frontend (React/TS)
      |
      v
AIOS daemon  (localhost, one process, owns everything below)
      |
      +---------------------------+
      v                           v
  Kernel + drivers           Provider adapters
      |                           |
      v                           v
  ~/.aios/ storage       Claude API / OpenAI-compat / Ollama / llama-server
```

**Treat AIOS as a daemon, not a library.** One long-lived localhost service
owns the kernel, the store, the journal, and all provider connections. The
desktop app, a CLI, a VS Code extension, and a browser extension all become
thin clients of the same memory. This is the single most important
architectural decision in this document.

Concretely: evolve `aios serve` into `aios daemon`. The current server is
single-threaded and blocking; the daemon needs one worker thread owning the
kernel (an actor: requests in over a channel, events out), so a slow
generation never blocks status endpoints, and cancellation is possible.

### Daemon API (localhost only, JSON over HTTP, SSE for streams)

```
POST /v1/turn            {text}            -> SSE: route, tok, fault, mem, evict, done
POST /v1/turn/cancel     {turn_id}
GET  /v1/timeline?before=<ts>&limit=50     -> the journal, newest first
GET  /v1/memory/search?q=...               -> ranked memories with sources
GET  /v1/memory/browse                     -> identity, branches, versions
GET  /v1/status                            -> model, drivers, pressure, kv, counters
PUT  /v1/settings        {model, budget, privacy_mode, drivers, ...}
POST /v1/kv/save         POST /v1/kv/restore
```

### The journal (new, small, required)

The kernel's memory is organized by topic, not by time, and the context
window is ephemeral. The timeline UI needs an append-only turn log: every
user and assistant turn with timestamp, plus markers for memory events
(remembered X, evicted N). Store it as JSONL under `~/.aios/journal/`,
owned by the daemon, never used for retrieval (the drivers do retrieval).
Timeline reads the journal; memory search reads the kernel.

### Provider adapters

One trait, provider-independent message schema (role, content), streaming
via callback, capability flags (supports_system, max_context):

- ClaudeAdapter: Anthropic Messages API, SSE streaming. Key from
  `~/.aios/keys` or env; never stored in the repo, never logged.
- OpenAICompatAdapter: one adapter covers OpenAI, LM Studio, vLLM,
  OpenRouter, Gemini's OpenAI endpoint. Base URL + key are settings.
- OllamaAdapter and LlamaServerAdapter: already exist in the crate
  (`src/ollama.rs`, `src/llamaserver.rs`); wrap them in the trait.

Frontier models are not fine-tuned for the fault protocol. Give them the
protocol in the system prompt and rely on the kernel's soft-refusal
detector (`detect_page_fault`) to catch "I don't have that information"
phrasings. Local tuned model gets the exact SYSTEM_TEMPLATE already in
`kernel.rs`; its bytes are load-bearing (the model was trained on them).

Model switching preserves continuity by construction, since memory never
lives in the model. The KV cache tier does not survive a model switch;
drop it silently and rebuild.

## UX

### No new chat

The app opens into the one timeline, already warm. First render after boot
shows a short daemon-composed digest from the journal:

```
Good morning.
Yesterday we continued the compiler project.
Your dentist appointment is tomorrow.
What would you like to work on?
```

No chat picker. No titles. No session IDs anywhere in the product, the
API, or the code.

### Timeline

Grouped by time, like Messages: Today, Yesterday, Last week, March.
Infinite scroll backwards through the journal. Jumping to a date is
navigation, not "opening an old chat."

### Search searches memory, not chats

"When did I decide to switch to Postgres?" is answered by the kernel
(`/v1/memory/search`), with the memory shown alongside its source turns
and version history. Results link back into the timeline.

### Memory browser

A separate view. Identity at top, then branches (topics) with summary,
recent facts, archive count, and per-value version history (the store is
copy-on-write; surface it). Every memory shows: current value, history,
source turn, last used. Users can correct or delete a memory; correction
writes a new version, deletion is the one true delete and requires
confirmation.

### Memory inspector (power users)

A per-response disclosure: memories retrieved, namespace, tokens loaded,
retrieval/generation latency, faults and retries, memory writes. All of
this already exists in QueryResult and the SSE events; render it.

### Settings

Model (per volume: /social and /workspace may differ), provider keys,
memory budget, max retrieved memories, temperature, driver on/off,
privacy mode.

### Privacy modes

- Persistent (default): full write-back.
- Incognito: kernel.query runs, write_back and ingest are skipped, journal
  entries are marked ephemeral and purged on exit.
- Paused: reads allowed, no writes, banner visible.

Implemented in the daemon, not the frontend, so every client honors them.

### Onboarding

The first conversation IS onboarding. The assistant asks who you are, what
you're working on, and whether to remember; write-back does the rest. The
memory browser fills up in front of the user during the first minute.
That's the product demo.

## What to market (and not)

Never say RAG, vectors, BM25, context window. Say:

```
Never explain yourself twice.
One AI. One relationship.
Kill the app, switch the model, come back in a month. It remembers.
```

## Success metrics

Product: return-after-30-days, average continuity (days between "who are
you" style re-explanations, higher is better), memory search usage, user
corrections per hundred memories (lower is better).
System (already instrumented): retrieval latency, fault rate, working set
size, memory growth on disk.

## MVP (build this, nothing else)

1. `aios daemon` with the API above, actor-threaded, journal included.
2. React + TypeScript frontend: timeline, composer with streaming, memory
   browser, memory search, settings, inspector disclosure. Local only.
3. ClaudeAdapter + OllamaAdapter behind the provider trait.
4. Privacy modes.
5. Migration: `aios serve` users' `companion/` state loads unchanged.

Explicitly out of MVP: plugins, browser/calendar/files drivers, sync,
multi-device, collaboration, Tauri packaging (wrap later; the web app
against localhost is enough to validate).

## Phase 2

Workspace integration (open folder -> CodeGraphDriver indexes it -> the
assistant knows the project), browser/files/calendar drivers, Tauri
desktop packaging, VS Code extension speaking to the same daemon.

## Phase 3

Multi-device sync of the store (text tier only; KV never syncs), shared
memories, collaboration.

## Technical requirements

- React + TypeScript frontend, no state that the daemon should own.
- SSE streaming with cancellation; the daemon already streams tokens.
- Adapter pattern for providers; provider-independent message schema.
- All persistence through the kernel and journal; the frontend never
  touches storage.
- Background indexing for drivers (workspace indexing must not block the
  timeline).
- Keys in `~/.aios/`, mode 600, never committed, never logged.
- Keep the kernel crate free of app dependencies: the daemon may live in
  this workspace, but `aios` the library must stay embeddable.

## Notes for the builder

Read README.md next; it contains the measured numbers behind every claim
here and the honest list of what does not work yet. The three commands
that prove your environment works before you write anything:

```
cargo test                                # 28 tests, no network
cargo run --release -- serve              # the prototype you are replacing
curl localhost:3210/api/status            # the API you are generalizing
```
