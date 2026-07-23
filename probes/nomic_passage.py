#!/usr/bin/env python
"""Final attribution control: nomic-embed-text at the PASSAGE level (not entity
strings). Our entity graph embedded entity tokens; contriever/HippoRAG embed full
passages. This isolates embedding-MODEL (nomic vs contriever) from our entity-graph
DESIGN. If nomic-passage also ranks drive.b near the top, the entity-level design was
the culprit; if nomic-passage ranks it low like our entity graph, it's the model."""
import ollama, numpy as np
from probe_labels import keep_fact

FACTS = [
    ("api.a","My API plan allows 50 thousand requests per month."),("api.b","I've burned through about 62 thousand calls already."),
    ("flat.a","The new flat is 2200 a month."),("flat.b","My take home pay works out to 2600."),
    ("drive.a","The external drive holds 500 gigabytes."),("drive.b","My photo library weighs in at 620 gigabytes."),
    ("grant.a","The grant deadline is March 10th."),("grant.b","I get back from Tokyo on March 8th."),
    ("eng.a","We budgeted for 12 engineers this year."),("eng.b","There are 15 people on the platform team now."),
    ("laptop.a","The flight to Berlin is 9 hours."),("laptop.b","This laptop runs about 6 hours on a charge."),
    ("gift.a","I set aside 300 for presents this year."),("gift.b","The wedding one cost 180 and the birthday one 140."),
    ("storage.a","The basic tier caps at 100 gigabytes."),("storage.b","I'm currently keeping 140 gigabytes up there."),
]
DISTR = ["What's a good warmup before a run?","Tell me something about the moon.","How do noise cancelling headphones work?",
    "I saw a heron by the river today.","Explain eventual consistency without jargon.","What makes sourdough different from regular bread?",
    "The gym was packed today.","Recommend a novel for a long flight."]
QUERIES = [("will the backup fit on the drive?","drive","storage"),("are we over the engineering hiring budget?","eng",""),
    ("do I need to move off the basic tier?","storage",""),("will my battery cover the entire Berlin flight?","laptop","")]

def emb(t):
    return np.array(ollama.Client(host="http://localhost:11434").embeddings(model="nomic-embed-text", prompt=t)["embedding"], dtype=np.float32)

def main():
    labels = [l for l,_ in FACTS]+[f"d{i}" for i in range(len(DISTR))]
    texts  = [t for _,t in FACTS]+DISTR
    E = np.stack([emb(t) for t in texts]); E /= np.linalg.norm(E,axis=1,keepdims=True)+1e-9
    for q,want,avoid in QUERIES:
        qv = emb(q); qv/=np.linalg.norm(qv)+1e-9
        sims = E@qv
        order = np.argsort(-sims)
        fact_ranked = [labels[i] for i in order if keep_fact(labels[i])]
        pos = {l:i for i,l in enumerate(fact_ranked)}
        print(f"\nQ: {q}")
        print(f"   {want}.a rank={pos.get(want+'.a')}  {want}.b rank={pos.get(want+'.b')}"
              + (f"  |  {avoid}.b(distractor) rank={pos.get(avoid+'.b')}" if avoid else ""))
        print(f"   top-5: {fact_ranked[:5]}")

if __name__ == "__main__":
    main()
