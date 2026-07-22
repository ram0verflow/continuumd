# continuum

An experiment in giving a local LLM long term memory by treating the context window like RAM. A small Rust kernel decides what gets paged into the window for each query, the model is trained to reply `CONTEXT_NEEDED: <topic>` when the loaded memory does not answer the question, and everything lives on disk in a versioned store so old facts get demoted rather than deleted.

It grew into an application: a daemon that owns the memory, and a web app in front of it with one timeline and no sessions. Kill the process, switch the model, come back a month later. It remembers.

![the timeline: a web fault, a memory forming, a cited answer](shots/app-timeline.png)

The whole thing runs on one MacBook (M5, 24GB). Everything from embeddings to the fine tune runs locally through Ollama and MLX. I spent about five dollars total on cloud, all of it on Claude API calls to grade benchmark answers.

I make no claims beyond the numbers below, which come from one benchmark on one machine. Read the caveats before quoting anything.

## Numbers

LoCoMo benchmark, 10 conversations, 1542 answerable and 444 unanswerable questions. Same answer model everywhere (llama 3.1 8B), same judge everywhere (claude haiku 4.5). Mem0 is the open source version, run locally with its BM25 extra installed, same protocol.

| system | answerable | refused unanswerable |
|---|---|---|
| no memory attached | 1.3% | |
| mem0 (OSS, local) | 31.0% | 80.5% |
| continuum | 54.3% | 41.7% |
| continuum + fine tune (conv 0 only) | 55.9% | 55.3% |

The fine tuned row is conv 0 only because the tune trained on the other nine. On that same held out conversation the untuned model gets 48.0% and refuses 25.5%, so the tune helped both numbers.

Caveats that matter. Mem0 reports 62 to 67% in its own publications using stronger answer models, so a good part of any system's headline number is the answer model, not the memory layer. Mem0's high refusal rate here comes partly from retrieving less: a system that finds little says "I don't know" a lot, which looks disciplined on unanswerable questions. Mem0 ingestion also cost 25 minutes to 4 hours per conversation on this hardware since it extracts facts with an LLM, versus about a minute here. The judge is nondeterministic by roughly one question per 150. The KV and coding session results are single digit sample sizes. And all of this is one benchmark.

## What holds

Four claims that are measured and have survived re-running. The numbers behind them, and the ones that did not survive, are in [FINDINGS.md](FINDINGS.md).

**Recall under total eviction.** A stress harness forces the session window down to 500 tokens, plants ten facts, and buries them under thirty-plus distractor turns so every planted fact is demoted out of the window into the archive, then asks for them all back. Across two runs, 19 of 20 came back exact, at 32 to 59 ms of retrieval, with the window peaking at 499 of 500 and never going over. A separate endurance run went 130 turns over 104 minutes and returned 9 of 10.

![recall after total eviction, exact answers at ~40ms](shots/stress-recall.png)

**Versioned corrections.** The store is copy on write: a correction writes a new version and keeps the history, and deletion is the one true delete. When the details learn a new dentist date, the topic summary versions from the 14th to the 21st with the old value retained, verifiable in the memory browser, and a contradicted fact usually answers with the current value.

**Honest refusals, with a caveat that stays attached.** Asked for a pool locker combination when only a gym one was ever mentioned, or a wedding date that was never given, the model page-faults and answers an honest "I don't have that." The caveat is real and not averaged away: in adversarial trials the model sometimes confabulates instead, returning the gym combination with confidence for a locker never mentioned. A per-model leak gauntlet found the pool-locker frame is the one that tempts, and the daemon defaults to the model that never leaked it. Honest refusal is the default; confabulation is the failure the fault fine-tune exists to reduce, not a solved problem.

**The LoCoMo numbers above**, from one benchmark on one machine, graded by an external judge because grading with the 8B answer model itself scored 19 points too high.

## How it was built

Continuum began as an older Python prototype and a short spec, ported to Rust into a four level versioned store (identity, topic summaries, details, raw archive) with a domain agnostic kernel over pluggable retrieval drivers. Retrieval took the longest to get right. Beam search over a topic tree, plus BM25 alongside embeddings, got recall up; the change that actually mattered was capping the load at 30 reranked messages presented chronologically. Loading around a hundred messages made the 8B model miss facts sitting in plain sight in its own context, and the cap fixed accuracy and halved latency at once. Dates were the worst question category, so relative phrases like "last week" are resolved in plain Rust against the timestamp of the message that said them, which moved that category from 62% to 81%. The fine tune (QLoRA on llama 3.1 8B, overnight on the laptop) trained the `CONTEXT_NEEDED` refusal behaviour from the benchmark's own conversations 1 through 9, with conversation 0 held out and never trained on. Every dead end behind those sentences, and there were many, is in [FINDINGS.md](FINDINGS.md).

## Running it

You need Rust and Ollama with two models pulled.

```
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve

cargo test                                # unit tests, no network needed
cargo build --release -p continuumd
(cd app && npm install && npm run build)  # once, for the UI
./target/release/continuumd              # http://localhost:4310
```

Keys, all optional, live in `~/.continuum/keys` as one JSON object, chmod 600, never logged, never returned by the API: `anthropic` for Claude, `openai` for OpenAI compatible endpoints, `brave` for real web search. MCP servers go in `~/.continuum/mcp.json` and their tools show up on the next daemon start.

The API, localhost JSON with SSE for the turn stream: `POST /v1/turn`, `POST /v1/turn/cancel`, `GET /v1/timeline`, `GET /v1/memory/search`, `GET /v1/memory/browse`, `POST /v1/memory/correct`, `POST /v1/memory/delete`, `GET/PUT /v1/settings`, `GET /v1/status`, `GET /v1/models`, `GET /v1/digest`, `GET /v1/media/<file>`, `POST /v1/kv/*`.

The older single file companion still works if you want the minimal version:

```
./target/release/continuum serve --model llama3.1:8b     # http://localhost:3210
```

For the benchmark CLI, get locomo10.json from the snap-research/locomo repo on GitHub, put it at data/locomo10.json, then:

```
./target/release/continuum ask "When did Caroline go to the LGBTQ support group?"
./target/release/continuum chat
```

Chat with KV persistence (attention states saved to disk on exit, restored on start). Needs llama-server, which reads GGUF straight out of the Ollama blob store:

```
brew install llama.cpp
BLOB=~/.ollama/models/blobs/$(ollama show llama3.1:8b --modelfile | grep -o 'sha256[^ ]*' | head -1 | tr : -)
mkdir -p kv_slots
llama-server -m $BLOB --port 8080 -c 8192 --slots --slot-save-path $PWD/kv_slots -np 1 &
./target/release/continuum chat --kv
```

## More

- **[FINDINGS.md](FINDINGS.md)** is the full research log, in order, including everything that failed or was retracted (the graph experiments, the off-the-shelf graph-RAG comparison, the conflation problem that is still open).
- **[DOCS.md](DOCS.md)** is the reference: how a turn flows, the daemon and app architecture, the API, and how to run the benchmarks, the fine tune, and the KV experiments.
- **[DESIGN.md](DESIGN.md)** is the product spec the daemon and app were built from.
