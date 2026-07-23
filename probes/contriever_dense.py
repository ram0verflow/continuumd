#!/usr/bin/env python
"""Attribution control: pure facebook/contriever dense retrieval, NO graph, NO PPR.
If this alone ranks drive.b (the 620) near the top, then HippoRAG's apparent win
over our nomic embedding graph is the embedding model (contriever > nomic), not the
graph algorithm. If contriever alone ranks drive.b low like nomic did, the graph/PPR
is doing the work. This is the experiment that attributes the result."""
import torch, torch.nn.functional as F
from probe_labels import keep_fact
from transformers import AutoTokenizer, AutoModel

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

def mean_pool(out, mask):
    tok = out[0]; mask = mask.unsqueeze(-1).expand(tok.size()).float()
    return (tok*mask).sum(1)/mask.sum(1).clamp(min=1e-9)

def main():
    tok = AutoTokenizer.from_pretrained("facebook/contriever")
    mdl = AutoModel.from_pretrained("facebook/contriever"); mdl.eval()
    labels = [l for l,_ in FACTS] + [f"d{i}" for i in range(len(DISTR))]
    texts  = [t for _,t in FACTS] + DISTR
    with torch.no_grad():
        enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
        E = mean_pool(mdl(**enc), enc["attention_mask"])
    for q, want, avoid in QUERIES:
        with torch.no_grad():
            qenc = tok([q], padding=True, truncation=True, return_tensors="pt")
            qv = mean_pool(mdl(**qenc), qenc["attention_mask"])
        sims = (E @ qv.T).squeeze(1)
        order = sims.argsort(descending=True).tolist()
        ranked = [(labels[i], float(sims[i])) for i in order]
        fact_ranked = [(l,s) for l,s in ranked if keep_fact(l)]  # real facts contain a dot; distractors are d0..d7
        pos = {l:i for i,(l,_) in enumerate(fact_ranked)}
        print(f"\nQ: {q}")
        print(f"   {want}.a rank={pos.get(want+'.a')}  {want}.b rank={pos.get(want+'.b')}"
              + (f"  |  {avoid}.a rank={pos.get(avoid+'.a')} {avoid}.b rank={pos.get(avoid+'.b')}" if avoid else ""))
        print(f"   top-5: {[l for l,_ in fact_ranked[:5]]}")

if __name__ == "__main__":
    main()
