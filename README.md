# aios

An experiment in giving a local LLM long term memory by treating the context window like RAM. A small Rust kernel decides what gets paged into the window for each query, the model is trained to reply `CONTEXT_NEEDED: <topic>` when the loaded memory does not answer the question, and everything lives on disk in a versioned store so old facts get demoted rather than deleted.

The whole thing runs on one MacBook (M5, 24GB). Everything from embeddings to the fine tune runs locally through Ollama and MLX. I spent about five dollars total on cloud, all of it on Claude API calls to grade benchmark answers.

I make no claims beyond the numbers below, which come from one benchmark on one machine. Read the caveats before quoting anything.

## How this was built

I started from an older Python prototype and a short spec. The port to Rust gave me a four level store (identity, topic summaries, details, raw archive) where every value keeps its version history, and a kernel/driver split where the kernel is domain agnostic and each driver owns retrieval for one kind of memory. There is a conversation driver and a code driver.

Retrieval took the longest to get right. The first version walked a topic tree down a single path and missed most things. Beam search over the tree helped. Adding BM25 alongside embeddings helped recall but made answers worse, and it took me a while to see why: I was loading around 100 messages per query and the 8B model would miss facts that were sitting in plain sight in its own context. Capping the load at 30 messages, reranked and then presented in chronological order, fixed accuracy and halved latency at the same time.

Dates were the worst question category. LoCoMo gold answers look like "the week before 27 June 2023" and a small model is bad at calendar math. So the code resolves phrases like "last week" or "yesterday" against the timestamp of the message that said them, in plain Rust, and injects the resolved date as a note. That category went from 62% to 81% on my local judge.

The fine tune came from the benchmark itself. Conversations 1 through 9 supplied synthetic training examples of three kinds: evidence loaded so answer it, evidence withheld so say CONTEXT_NEEDED, and a trap loaded (a question about the wrong person) so still say CONTEXT_NEEDED. Conversation 0 was held out and never used for training. QLoRA on llama 3.1 8B through MLX, overnight on the laptop. The first round refused too much. The second round changed the mix and paired every trap with a question the same context could answer. The share of refusal examples in the training data acts like a dial for how conservative the model is.

Two things about evaluation that I got wrong at first and had to fix. I was grading answers with the same 8B model that produced them, and when I re graded with Claude Haiku the score dropped 19 points. Every number below is from the external judge. And I worried the base model might know this public dataset from pretraining, so I asked it the conv 0 questions with no memory attached. It scored 1.3%, which is guessing.

One result I did not expect: the nine conversations I never tuned retrieval on scored a bit higher than the one I iterated against. So the retrieval stack was not overfit to my dev set, the dev conversation just happens to be harder.

The KV cache part came last. Prefill is 97 to 99% of query latency here, and llama.cpp can save per sequence KV state to disk and shift RoPE positions of cached tokens. That means a memory block can be encoded once at position zero, saved, and later restored at any offset and stitched next to other blocks without re reading the text. It works: the model answered correctly from a block that had been shifted 30 positions. There is also a harness that runs a fake coding session where the codebase is five times the context window, and the planted facts survive while questions about absent code get refused. I wrote that codebase and those questions myself, so treat it as a demo rather than an evaluation.

## Numbers

LoCoMo benchmark, 10 conversations, 1542 answerable and 444 unanswerable questions. Same answer model everywhere (llama 3.1 8B), same judge everywhere (claude haiku 4.5). Mem0 is the open source version, run locally with its BM25 extra installed, same protocol.

| system | answerable | refused unanswerable |
|---|---|---|
| no memory attached | 1.3% | |
| mem0 (OSS, local) | 31.0% | 80.5% |
| aios | 54.3% | 41.7% |
| aios + fine tune (conv 0 only) | 55.9% | 55.3% |

The fine tuned row is conv 0 only because the tune trained on the other nine. On that same held out conversation the untuned model gets 48.0% and refuses 25.5%, so the tune helped both numbers.

Caveats that matter. Mem0 reports 62 to 67% in its own publications using stronger answer models, so a good part of any system's headline number is the answer model, not the memory layer. Mem0's high refusal rate here comes partly from retrieving less: a system that finds little says "I don't know" a lot, which looks disciplined on unanswerable questions. Mem0 ingestion also cost 25 minutes to 4 hours per conversation on this hardware since it extracts facts with an LLM, versus about a minute here. The judge is nondeterministic by roughly one question per 150. The KV and coding session results are single digit sample sizes. And all of this is one benchmark.

## Running it

You need Rust and Ollama with two models pulled. The dataset comes from GitHub.

```
# models
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve

# dataset: get locomo10.json from the snap-research/locomo repo on GitHub
# and put it at data/locomo10.json

cargo test            # 27 unit tests, no network or models needed
cargo build --release
```

Basic use:

```
./target/release/aios info
./target/release/aios ask "When did Caroline go to the LGBTQ support group?"
./target/release/aios chat
```

## The companion

`aios serve` runs the whole thing as a local web app on http://localhost:3210.
One binary, no other dependencies, nothing leaves your machine.

```
./target/release/aios serve --model llama3.1:8b
```

The left side is a chat with streamed replies. The right side has two tabs:
a kernel log showing what happened on every turn (memories paged in, page
faults, what got written back, evictions) and a memory tab that lists what
it currently believes about you, topic by topic. Its memory lives in
`companion/` on disk, so you can kill the process, start it again, and it
still knows what you told it. Tell it your name in one session and ask who
you are in the next.

There is an endurance script that hammers this loop: it plants ten facts,
buries them under a hundred turns of unrelated chatter on the small fixed
window, then asks for them back. `python3 endurance.py <port>` against a
running server.

Notes from using it: write back runs one extra model call per turn, so
replies take a few seconds longer than plain chat. The 8B model sometimes
decorates recalled facts, in one test it added a year to a date I never
gave it. And if llama-server is already running on port 8080 the companion
picks it up and uses KV state save and restore automatically.

Chat with KV persistence (attention states saved to disk on exit, restored on start). Needs llama-server, which reads GGUF straight out of the Ollama blob store:

```
brew install llama.cpp
BLOB=~/.ollama/models/blobs/$(ollama show llama3.1:8b --modelfile | grep -o 'sha256[^ ]*' | head -1 | tr : -)
mkdir -p kv_slots
llama-server -m $BLOB --port 8080 -c 8192 --slots --slot-save-path $PWD/kv_slots -np 1 &
./target/release/aios chat --kv
```

## Ablations, other models, a second benchmark

Removing one retrieval component at a time (conversation 0, base llama 3.1,
ROUGE-L, same 154 questions):

| configuration | ROUGE-L | page faults |
|---|---|---|
| full pipeline | 0.449 | 14 |
| without tree routing | 0.455 | 12 |
| without dense embeddings | 0.442 | 16 |
| without the date resolver | 0.424 | 13 |
| without BM25 | 0.265 | 66 |
| without the 30 message cap | 0.118 | 1 |

The cap and BM25 carry the system. Removing the cap reproduces the failure
that shaped the design: the model reads a hundred loosely relevant messages,
answers wrongly with confidence, and generation time doubles. The tree adds
nothing on this benchmark; its case is browsing and scale, not QA, and I keep
it because the online ingestion path builds it for free.

Same stack, different answer models, same questions: mistral 7b scores 0.432,
nearly identical to llama's 0.449, so the architecture is not tuned to one
model family. phi3 mini (3.8B) drops to 0.215; below some capability floor
the model cannot use what gets paged in.

Retrieval cost, measured: 33 to 57 ms at the median against 7 to 14 seconds
of generation. The memory side is about half a percent of a query.

BABILong (facts hidden in 64k tokens of book text, answered through a 4k
window, exact match): qa1 13/20, qa2 0/20, qa3 2/20, qa4 12/20, qa5 20/20.
Single fact tasks work well through sparse retrieval. Chained fact tasks
fail, the same multi hop weakness LoCoMo showed, and the known failure mode
of retrieval systems generally. Fetch the data with `fetch_babilong.py`,
run with `cargo run --release --bin babilong`.

## Running the benchmarks

Generate predictions for one conversation, or all ten:

```
./target/release/eval --conv 0 --limit 999 --no-judge --jsonl fullbench/aios_conv0.jsonl
./target/release/eval --conv 0 --adv-only --jsonl fullbench/aios_adv0.jsonl

for i in 0 1 2 3 4 5 6 7 8 9; do
  ./target/release/eval --conv $i --limit 999 --no-judge --jsonl fullbench/aios_conv$i.jsonl
  ./target/release/eval --conv $i --adv-only --jsonl fullbench/aios_adv$i.jsonl
done
```

Grade them. Put an API key in `.env` (either `ANTHROPIC_API_KEY=...` or `OPENAI_API_KEY=...`, the script picks whichever exists). Grading all ten conversations cost me under two dollars.

```
python3 judge_frontier.py "fullbench/aios_conv*.jsonl" "fullbench/aios_adv*.jsonl"
```

The no memory baseline:

```
python3 contamination_gen.py
python3 judge_frontier.py fullbench/contamination_conv0.jsonl
```

The mem0 comparison. Ingestion is slow, hours on my machine:

```
python3 -m venv .venv && .venv/bin/pip install mem0ai ollama fastembed
.venv/bin/python mem0_bench.py
python3 judge_frontier.py "fullbench/mem0_conv?.jsonl"
```

## The fine tune

```
.venv/bin/pip install mlx-lm "transformers==4.56.2"
python3 gen_training_data.py     # writes ft_data/ from conversations 1-9

.venv/bin/python -m mlx_lm lora --train \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --data ft_data --batch-size 2 --iters 800 --num-layers 16 \
  --max-seq-length 3400 --grad-checkpoint --learning-rate 1e-5 \
  --adapter-path adapters --save-every 200

.venv/bin/python -m mlx_lm fuse \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --adapter-path adapters --save-path fused --dequantize

# convert fused/ to GGUF with llama.cpp's convert_hf_to_gguf.py, then:
ollama create aios-ft -f Modelfile    # FROM ./your.gguf
./target/release/eval --conv 0 --limit 999 --model aios-ft --judge-model llama3.1:8b
```

Training took my machine about 12 hours per round because it throttles. A rented GPU would do it in under an hour.

## The KV experiments

These use llama.cpp through FFI, so the first build takes a couple of minutes.

```
BLOB=<path to a llama 3.1 gguf, the ollama blob works>
cargo run --release -p kvpoc --bin kvpoc -- $BLOB          # encode two blocks, shift one, stitch, query
cargo run --release -p kvpoc --bin cacheblend -- $BLOB data/conv_0.json   # stitched vs monolithic answers
cargo run --release -p kvpoc --bin codesession -- $BLOB    # fake coding session, codebase 5x the window
```

A known cosmetic issue: the kvpoc binaries hit a Metal assert inside llama.cpp during process exit, after results are printed. Upstream PR 17869.

## Layout

```
src/kernel.rs        page fault loop, context assembly, write back
src/driver.rs        the driver trait and the tree node type
src/hierarchical.rs  conversation driver: tree + BM25 + embeddings, online ingestion, date resolver
src/codegraph.rs     code driver: symbol extraction + BM25, no embeddings
src/store.rs         four level versioned store
src/eviction.rs      context window eviction and demotion
src/llamaserver.rs   llama-server client for KV save/restore
src/server.rs        the companion web app (aios serve), UI embedded from src/ui.html
src/bin/eval.rs      LoCoMo runner
src/bin/stress.rs    all ten conversations merged into one store
src/bin/transfer.rs  fine tuned model on code questions it never trained on
kvpoc/               KV cache proofs of concept
```

Not done: a latency comparison against warm prefix caching, and a test on a real repository instead of a synthetic one.
