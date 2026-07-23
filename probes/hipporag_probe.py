#!/usr/bin/env python
"""Deterministic-composition probe for HippoRAG, mirroring examples/retrieval_probe.rs
and lightrag_probe.py. Real published HippoRAG 2.0.0a3: OpenIE via local Ollama
(qwen2.5:14b through the OpenAI-compatible endpoint), embeddings via HippoRAG's
native facebook/contriever, and its Personalized-PageRank retrieval.

For each crux query we take HippoRAG's ranked passages (no QA/LLM answer) and
report whether both operands the synthesis question needs came back and whether
the distractor bled in — same composition check as the other methods.
"""
import os, sys, json

os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-ollama")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SAVE = os.path.join(os.path.dirname(__file__), "hippo_work")
LLM = "qwen2.5:14b"
LLM_BASE = "http://localhost:11434/v1"
EMBED = "facebook/contriever"

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
DISTRACTORS = [
    "What's a good warmup before a run?","Tell me something about the moon.",
    "How do noise cancelling headphones work?","I saw a heron by the river today.",
    "Explain eventual consistency without jargon.","What makes sourdough different from regular bread?",
    "The gym was packed today.","Recommend a novel for a long flight.",
]
QUERIES = [
    ("will the backup fit on the drive?","drive","storage"),
    ("are we over the engineering hiring budget?","eng",""),
    ("do I need to move off the basic tier?","storage",""),
    ("will my battery cover the entire Berlin flight?","laptop",""),
]
LABEL = {}
for lab,a,b in FACTS: LABEL[a]=f"{lab}.a"; LABEL[b]=f"{lab}.b"

def label_of(passage):
    p = passage.strip()
    for fact,l in LABEL.items():
        if fact[:38] in p or p[:38] in fact:
            return l
    return None

def main():
    from hipporag import HippoRAG
    docs = [a for _,a,_ in FACTS] + [b for _,_,b in FACTS] + DISTRACTORS
    hr = HippoRAG(save_dir=SAVE, llm_model_name=LLM, llm_base_url=LLM_BASE,
                  embedding_model_name=EMBED)
    print(f"[index] {len(docs)} docs via OpenIE({LLM}) + {EMBED} ...", flush=True)
    hr.index(docs=docs)
    print("[index] done", flush=True)

    for q, want, avoid in QUERIES:
        sols = hr.retrieve(queries=[q], num_to_retrieve=len(docs))
        sol = sols[0]
        passages = getattr(sol, "docs", None) or getattr(sol, "documents", None) or []
        ranked = []
        for p in passages:
            l = label_of(p if isinstance(p,str) else str(p))
            if l and l not in ranked:
                ranked.append(l)
        both = f"{want}.a" in ranked and f"{want}.b" in ranked
        bled = bool(avoid) and any(l.startswith(avoid) for l in ranked)
        # rank of first want-fact and of the distractor
        print(f"\nQ: {q}")
        print(f"   want {want}.a+{want}.b | avoid {avoid or '-'}")
        print(f"   -> both={'Y' if both else 'N'}  bleed={'BLEED' if bled else 'ok'}")
        print(f"      ranked facts: {ranked}")

if __name__ == "__main__":
    main()
