#!/usr/bin/env python3
"""Keyword disjoint composition: the case the driver index cannot reach.

Every case needs two facts. The question shares vocabulary with at most one
of them, and the two facts share almost nothing with each other, so lexical
retrieval (BM25 plus embeddings over message text) can surface one side and
has no route to the other. This is the exact shape of the allowance case,
which is the only composition case that has ever passed, and it passed only
in the run where store_context carried the second fact into the prompt.

That single pass is n of 1. This harness exists to find out whether it
replicates across eight cases of the same shape, because that decides
whether the store's runtime path is worth building out (issue #14) or
whether the one pass was luck.

Protocol matches stress_discriminate.py: same interleaving, same gaps, same
grader, per answer attribution logged.

  AIOS_HOME=/tmp/aios-disj ./target/release/aios-daemon --port 4317 &
  python3 stress_disjoint.py 4317 off
  python3 stress_disjoint.py 4317 on     # fresh state first
"""

import json
import sys
import urllib.request

from grading import verdict

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 4310
MODE = sys.argv[2] if len(sys.argv) > 2 else "off"
GAPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
BASE = f"http://127.0.0.1:{PORT}"

# fact_a is the side the question can reach lexically.
# fact_b is deliberately disjoint from both the question and fact_a.
CASES = [
    {
        "name": "api allowance",
        "a": "My API plan allows 50 thousand requests per month.",
        "b": "I've burned through about 62 thousand calls already.",
        "q": "am I over my monthly API allowance?",
        "needles": ["12 thousand", "12,000", "12000", "exceeded"],
    },
    {
        "name": "flat affordability",
        "a": "The new flat is 2200 a month.",
        "b": "My take home pay works out to 2600.",
        "q": "can I afford the new flat?",
        "needles": ["400"],
    },
    {
        "name": "drive capacity",
        "a": "The external drive holds 500 gigabytes.",
        "b": "My photo library weighs in at 620 gigabytes.",
        "q": "will the backup fit on the drive?",
        "needles": ["120", "will not fit", "won't fit", "too large", "not enough"],
    },
    {
        "name": "grant deadline",
        "a": "The grant deadline is March 10th.",
        "b": "I get back from Tokyo on March 8th.",
        "q": "how long after I land is the grant due?",
        "needles": ["2 days", "two days"],
    },
    {
        "name": "eng headcount",
        "a": "We budgeted for 12 engineers this year.",
        "b": "There are 15 people on the platform team now.",
        "q": "are we over the engineering hiring budget?",
        # "over the engineering hiring budget" as an adjacent phrase is safe:
        # a hedge would restate the question but verdict() also rejects punts.
        # This case produced the grader's first false NEGATIVE (a correct
        # "you are over the engineering hiring budget" graded FAIL).
        "needles": ["3", "exceeded", "over the engineering hiring budget", "over budget"],
    },
    {
        "name": "laptop battery",
        "a": "The flight to Berlin is 9 hours.",
        "b": "This laptop runs about 6 hours on a charge.",
        "q": "can I work the whole way to Berlin without a power outlet?",
        "needles": ["3", "will not", "won't", "not enough", "no"],
    },
    {
        "name": "gift budget",
        "a": "I set aside 300 for presents this year.",
        "b": "The wedding one cost 180 and the birthday one 140.",
        "q": "am I still inside what I set aside for presents?",
        "needles": ["20", "320", "exceeded"],
    },
    {
        "name": "storage tier",
        "a": "The basic tier caps at 100 gigabytes.",
        "b": "I'm currently keeping 140 gigabytes up there.",
        "q": "do I need to move off the basic tier?",
        "needles": ["40", "yes", "exceeded"],
    },
]

GAP = [
    "What's a good warmup before a run?",
    "Tell me something about the moon.",
    "How do noise cancelling headphones work?",
    "I saw a heron by the river today.",
    "Explain eventual consistency without jargon.",
    "What makes sourdough different from regular bread?",
    "The gym was packed today.",
    "Recommend a novel for a long flight.",
]


def turn(text, timeout=240):
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/turn", data=body, headers={"Content-Type": "application/json"}
    )
    reply, done = "", None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                if ev.get("t") == "done":
                    done = ev
                    reply = ev.get("reply", "")
    return reply, done


def put_settings(patch):
    body = json.dumps(patch).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/settings", data=body, method="PUT",
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=30).read()


def overlap(x, y):
    """Informative token overlap, to prove the cases really are disjoint."""
    def toks(s):
        return {t for t in "".join(c.lower() if c.isalnum() else " " for c in s).split()
                if len(t) > 3 and t not in {"this", "that", "with", "have", "about", "already"}}
    a, b = toks(x), toks(y)
    return sorted(a & b)


def main():
    on = MODE == "on"
    put_settings({"window_budget": 500, "store_context": on})
    print(f"condition: store_context {'ON' if on else 'OFF'}\n")

    print("— disjointness check (shared informative tokens between the two facts)")
    for c in CASES:
        print(f"  {c['name']:18} a/b: {overlap(c['a'], c['b']) or 'none'}"
              f"   q/b: {overlap(c['q'], c['b']) or 'none'}")

    rounds, gi = 2, 0
    for r in range(rounds):
        for c in CASES:
            turn(c["a"] if r == 0 else c["b"])
        for _ in range(GAPS):
            turn(GAP[gi % len(GAP)])
            gi += 1

    print("\n— questions")
    results = []
    for c in CASES:
        reply, done = turn(c["q"])
        insp = (done or {}).get("inspector", {})
        ok = verdict(reply, c["needles"])
        results.append({"case": c["name"], "reply": reply, "ok": ok,
                        "store_topics": insp.get("store_topics"),
                        "loaded": insp.get("loaded"),
                        "faulted": insp.get("faulted"),
                        "actions": insp.get("actions"),
                        "loop_trace": insp.get("loop_trace")})
        print(f"  [{'PASS' if ok else 'FAIL'}] st={insp.get('store_topics')} "
              f"{c['name']:18} -> {reply[:66]!r}")

    errored = sum(1 for r in results if r["reply"].startswith("[ERROR"))
    if errored >= 3:
        print(f"!! infra failure: {errored}/{len(results)} turns errored; run INVALID")
    passed = sum(r["ok"] for r in results)
    with_store = sum(1 for r in results if (r["store_topics"] or 0) > 0)
    print(f"\n=== store_context {'ON' if on else 'OFF'}: {passed}/{len(CASES)} "
          f"| answers that paged a store topic: {with_store} ===")
    with open(f"/tmp/disjoint_{MODE}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
