#!/usr/bin/env python
"""Query-only pass over the already-indexed LightRAG store (no re-indexing).
Robust fact labeling by exact-substring, ordered by position in the context."""
import asyncio, os, json, glob

WORK = os.path.join(os.path.dirname(__file__), "lightrag_work")
LLM_MODEL, EMBED_MODEL, EMBED_DIM = "qwen2.5:14b", "nomic-embed-text", 768

FACTS = [
    ("api","My API plan allows 50 thousand requests per month.","I've burned through about 62 thousand calls already."),
    ("flat","The new flat is 2200 a month.","My take home pay works out to 2600."),
    ("drive","The external drive holds 500 gigabytes.","My photo library weighs in at 620 gigabytes."),
    ("grant","The grant deadline is March 10th.","I get back from Tokyo on March 8th."),
    ("eng","We budgeted for 12 engineers this year.","There are 15 people on the platform team now."),
    ("laptop","The flight to Berlin is 9 hours.","This laptop runs about 6 hours on a charge."),
    ("gift","I set aside 300 for presents this year.","The wedding one cost 180 and the birthday one 140."),
    ("storage","The basic tier caps at 100 gigabytes.","I'm currently keeping 140 gigabytes up there."),
]
QUERIES = [
    ("will the backup fit on the drive?","drive","storage"),
    ("are we over the engineering hiring budget?","eng",""),
    ("do I need to move off the basic tier?","storage",""),
    ("will my battery cover the entire Berlin flight?","laptop",""),
]
LABELS = {}
for lab,a,b in FACTS: LABELS[a]=f"{lab}.a"; LABELS[b]=f"{lab}.b"

def labels_in_order(ctx):
    if not isinstance(ctx,str): ctx=json.dumps(ctx)
    hits=[]
    for fact,lab in LABELS.items():
        # strip a leading pronoun/quote artifact; match the distinctive tail
        key = fact.rstrip(".")
        pos = ctx.find(key[:40])
        if pos>=0: hits.append((pos,lab))
    return [lab for _,lab in sorted(hits)]

async def my_embed(texts):
    import numpy as np, ollama
    client = ollama.AsyncClient(host="http://localhost:11434")
    out=[]
    for t in texts:
        r = await client.embeddings(model=EMBED_MODEL, prompt=t); out.append(r["embedding"])
    return np.array(out, dtype=np.float32)

async def main():
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.ollama import ollama_model_complete
    from lightrag.utils import EmbeddingFunc
    try: from lightrag.kg.shared_storage import initialize_pipeline_status
    except Exception: initialize_pipeline_status=None
    rag = LightRAG(working_dir=WORK, llm_model_func=ollama_model_complete, llm_model_name=LLM_MODEL,
        llm_model_kwargs={"host":"http://localhost:11434","options":{"num_ctx":8192,"temperature":0}},
        embedding_func=EmbeddingFunc(embedding_dim=EMBED_DIM,max_token_size=8192,func=my_embed))
    await rag.initialize_storages()
    if initialize_pipeline_status: await initialize_pipeline_status()

    # KG entity dump: how did ingestion link the storage-140 fact?
    for f in glob.glob(os.path.join(WORK,"vdb_entities.json")):
        d=json.load(open(f)); data=d.get("data",d)
        names=sorted({str(e.get("entity_name") or e.get("__id__") or "") for e in data})
        print(f"[KG] {len(names)} entities: {names}\n")

    first=True
    for q,want,avoid in QUERIES:
        for mode in ("local","global","hybrid"):
            ctx = await rag.aquery(q, param=QueryParam(mode=mode, only_need_context=True))
            if first:
                print("===== RAW context sample (drive/local) =====")
                print((ctx if isinstance(ctx,str) else json.dumps(ctx))[:1500])
                print("===== end sample =====\n"); first=False
            labs = labels_in_order(ctx)
            both = f"{want}.a" in labs and f"{want}.b" in labs
            bled = bool(avoid) and any(l.startswith(avoid) for l in labs)
            print(f"[{mode:6}] {q[:38]:38} both={'Y' if both else 'N'} bleed={'BLEED' if bled else 'ok':5} facts={labs}")

if __name__=="__main__": asyncio.run(main())
