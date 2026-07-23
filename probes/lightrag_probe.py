#!/usr/bin/env python
"""Deterministic-composition probe for LightRAG, mirroring examples/retrieval_probe.rs.

Indexes the same disjoint fact set with LightRAG (LLM entity/relation extraction
via local Ollama qwen2.5:14b + nomic-embed), then for each crux query pulls the
RETRIEVED CONTEXT ONLY (no LLM answer, no judge) and reports which source facts
came back and whether both operands the synthesis question needs are present.

Also dumps the extracted knowledge graph so we can see how ingestion-time entity
linking handled "I'm currently keeping 140 gigabytes up there" — whether it
attached the 140 to the storage tier (coref-aware) or to a literal token.
"""
import asyncio, os, glob, json, re, sys

WORK = os.path.join(os.path.dirname(__file__), "lightrag_work")
LLM_MODEL = "qwen2.5:14b"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768

FACTS = [
    ("api",     "My API plan allows 50 thousand requests per month.", "I've burned through about 62 thousand calls already."),
    ("flat",    "The new flat is 2200 a month.",                      "My take home pay works out to 2600."),
    ("drive",   "The external drive holds 500 gigabytes.",            "My photo library weighs in at 620 gigabytes."),
    ("grant",   "The grant deadline is March 10th.",                  "I get back from Tokyo on March 8th."),
    ("eng",     "We budgeted for 12 engineers this year.",            "There are 15 people on the platform team now."),
    ("laptop",  "The flight to Berlin is 9 hours.",                   "This laptop runs about 6 hours on a charge."),
    ("gift",    "I set aside 300 for presents this year.",            "The wedding one cost 180 and the birthday one 140."),
    ("storage", "The basic tier caps at 100 gigabytes.",              "I'm currently keeping 140 gigabytes up there."),
]
DISTRACTORS = [
    "What's a good warmup before a run?", "Tell me something about the moon.",
    "How do noise cancelling headphones work?", "I saw a heron by the river today.",
    "Explain eventual consistency without jargon.", "What makes sourdough different from regular bread?",
    "The gym was packed today.", "Recommend a novel for a long flight.",
]
# (query, label whose two facts we WANT both back, label that must NOT bleed in)
QUERIES = [
    ("will the backup fit on the drive?",                 "drive",   "storage"),
    ("are we over the engineering hiring budget?",        "eng",     ""),
    ("do I need to move off the basic tier?",             "storage", ""),
    ("will my battery cover the entire Berlin flight?",   "laptop",  ""),
]

# Map an exact fact string -> label like "drive.a"
LABEL_OF = {}
for lab, a, b in FACTS:
    LABEL_OF[a] = f"{lab}.a"
    LABEL_OF[b] = f"{lab}.b"

def label_for_chunk(text):
    """Best-effort: which planted fact does a retrieved chunk correspond to."""
    t = text.strip()
    for fact, lab in LABEL_OF.items():
        if fact in t or t in fact:
            return lab
    return None

async def my_embed(texts):
    """Direct Ollama nomic embed (768-dim), bypassing LightRAG's ollama_embed
    which is decorated with a default embedding_dim=1024 validator."""
    import numpy as np, ollama
    client = ollama.AsyncClient(host="http://localhost:11434")
    out = []
    for t in texts:
        r = await client.embeddings(model=EMBED_MODEL, prompt=t)
        out.append(r["embedding"])
    return np.array(out, dtype=np.float32)

async def main():
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.ollama import ollama_model_complete
    from lightrag.utils import EmbeddingFunc
    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status
    except Exception:
        initialize_pipeline_status = None

    os.makedirs(WORK, exist_ok=True)
    rag = LightRAG(
        working_dir=WORK,
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 8192, "temperature": 0}},
        embedding_func=EmbeddingFunc(embedding_dim=EMBED_DIM, max_token_size=8192, func=my_embed),
    )
    await rag.initialize_storages()
    if initialize_pipeline_status:
        await initialize_pipeline_status()

    # Insert each fact/distractor as its own document so chunks stay atomic.
    docs = [a for _, a, _ in FACTS] + [b for _, _, b in FACTS] + DISTRACTORS
    print(f"[index] inserting {len(docs)} atomic docs via {LLM_MODEL} extraction ...", flush=True)
    for d in docs:
        await rag.ainsert(d)
    print("[index] done", flush=True)

    # --- Dump the extracted KG: how did ingestion link the storage-140 fact? ---
    dump_graph()

    for q, want, avoid in QUERIES:
        for mode in ("local", "global", "hybrid"):
            ctx = await rag.aquery(q, param=QueryParam(mode=mode, only_need_context=True))
            labels = extract_labels(ctx)
            reached = [l for l in labels if l and l.startswith(want)]
            both = f"{want}.a" in labels and f"{want}.b" in labels
            bled = bool(avoid) and any(l and l.startswith(avoid) for l in labels)
            print(f"\nQ[{mode:6}] {q}")
            print(f"   want both {want}.a+{want}.b | avoid {avoid or '-'}")
            print(f"   -> both={'Y' if both else 'N'} reached={reached} bleed={'BLEED' if bled else 'ok'}")
            print(f"      all facts in context (ranked): {labels}")

def extract_labels(ctx):
    """Parse LightRAG only_need_context output for retrieved source-chunk facts,
    in the order they appear (proxy for rank)."""
    if isinstance(ctx, dict):
        ctx = json.dumps(ctx)
    labels = []
    for m in re.finditer(r'[A-Z][^\n"]{5,120}', ctx):
        lab = label_for_chunk(m.group(0))
        if lab and lab not in labels:
            labels.append(lab)
    return labels

def dump_graph():
    print("\n===== EXTRACTED KNOWLEDGE GRAPH (entities) =====")
    gml = glob.glob(os.path.join(WORK, "*graph*"))
    ents_files = glob.glob(os.path.join(WORK, "vdb_entities.json"))
    for f in ents_files:
        try:
            d = json.load(open(f))
            data = d.get("data", d)
            names = [e.get("entity_name") or e.get("__id__") or e.get("content", "")[:40] for e in (data if isinstance(data, list) else [])]
            print(f"  entities ({len(names)}): {sorted(set(str(n) for n in names))}")
        except Exception as e:
            print(f"  (couldn't parse {f}: {e})")
    # relationships around the storage / 140 fact
    rels = glob.glob(os.path.join(WORK, "vdb_relationships.json"))
    for f in rels:
        try:
            d = json.load(open(f)); data = d.get("data", d)
            print(f"  relationships: {len(data) if isinstance(data,list) else '?'}")
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())
