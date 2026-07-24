#!/usr/bin/env python3
"""Cases where finding the message is not enough.

The compaction harness tests facts that live in one message, which raw
message retrieval already wins. These cases don't live in one message:

  synthesis      the answer must be computed from two mentions that share
                 no keywords with each other or with the question
  current-vs-ever  a fact contradicted repeatedly; the answer is the last
                 value, and every mention keyword-matches the question
                 equally, so recency has to come from somewhere real
  cross-branch   two separately stated facts that must be combined to
                 answer a question nobody stated

Run once with store_context off and once with on (the script sets it),
fresh CONTINUUM_HOME each time, and compare:

  CONTINUUM_HOME=/tmp/continuum-disc ./target/release/continuumd --port 4316 &
  python3 stress_discriminate.py 4316 off
  python3 stress_discriminate.py 4316 on     # fresh state again first
"""

import json
import os
import sys
import urllib.request

from grading import verdict, contains_any
from judge import verdict as judge_verdict

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 4310
MODE = sys.argv[2] if len(sys.argv) > 2 else "off"
GAPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2  # distractors per round; raise for long transcripts
BASE = f"http://127.0.0.1:{PORT}"

# Each case: ordered mentions (planted with distractor gaps between them),
# then a question, an expected needle, and forbidden stale needles.
CASES = [
    {
        "name": "synthesis: date shift",
        "mentions": [
            "My dentist appointment is on October 14th.",
            "Heads up: I'm pushing everything in my calendar back by exactly one week.",
        ],
        "question": "when is my dentist appointment?",
        "truth": "October 21st (originally the 14th, pushed back one week)",
        "forbidden": ["october 14"],
    },
    {
        "name": "synthesis: arithmetic",
        "mentions": [
            "My rent is 1800 a month.",
            "My landlord told me everything goes up by 200 starting next month.",
        ],
        "question": "what will my rent be next month?",
        "truth": "2000 a month (1800 plus the 200 increase)",
        "forbidden": ["1800"],
    },
    {
        "name": "current-vs-ever: editor",
        "mentions": [
            "My main editor is vim.",
            "I switched my main editor to helix this week.",
            "Helix didn't stick, I'm back on neovim as my main editor.",
            "Final answer: I've settled on zed as my main editor for good.",
        ],
        "question": "what is my main editor these days?",
        "truth": "zed, and nothing else; naming vim, helix or neovim as the current editor is wrong",
        "forbidden": ["vim", "helix", "neovim"],
    },
    {
        "name": "current-vs-ever: standup time",
        "mentions": [
            "Our team standup is at 9:00.",
            "Standup moved to 9:30 from this sprint.",
            "Standup is now 8:45, the earlier slot won the vote.",
        ],
        "question": "what time is standup?",
        "truth": "8:45",
        "forbidden": ["9:00", "9:30"],
    },
    {
        "name": "cross-branch: over allowance",
        "mentions": [
            "My API plan allows 50 thousand requests per month.",
            "I've burned through about 62 thousand calls so far this month.",
        ],
        "question": "am I over my monthly API allowance?",
        "truth": "over the allowance by 12,000 (62k used vs 50k plan)",
        "forbidden": [],
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


def main():
    on = MODE == "on"
    put_settings({"window_budget": 500, "store_context": on})
    print(f"condition: store_context {'ON' if on else 'OFF'}\n")

    # Interleave: all cases' mentions in rounds, distractors between rounds,
    # so mentions of one case are far apart in the transcript.
    rounds = max(len(c["mentions"]) for c in CASES)
    gi = 0
    for r in range(rounds):
        for c in CASES:
            if r < len(c["mentions"]):
                turn(c["mentions"][r])
        for _ in range(GAPS):
            turn(GAP[gi % len(GAP)])
            gi += 1

    print("— questions")
    results = []
    for c in CASES:
        reply, done = turn(c["question"])
        insp = (done or {}).get("inspector", {})
        low = reply.lower()
        # Verdict-only grading (#23): the judge decides correctness against the
        # gold. Substring needles produced six corrupted results in this project,
        # in both directions, most recently passing "Neovim (affectionately
        # referred to as zed)" because the string "zed" appeared in it.
        ok = judge_verdict(c["question"], c["truth"], reply)
        stale = contains_any(reply, c["forbidden"]) and not ok
        results.append({"case": c["name"], "reply": reply, "ok": ok, "stale": stale,
                        "store_topics": insp.get("store_topics"),
                        "loaded": insp.get("loaded"),
                        "actions": insp.get("actions"),
                        "loop_trace": insp.get("loop_trace")})
        mark = "PASS" if ok else ("STALE" if stale else "FAIL")
        print(f"  [{mark:5}] st={insp.get('store_topics')} {c['name']:28} -> {reply[:70]!r}")

    errored = sum(1 for r in results if r["reply"].startswith("[ERROR"))
    if errored >= 3:
        print("!! infra failure: %d/%d turns errored (bedrock transport); run INVALID, not a model result" % (errored, len(results)))
    passed = sum(r["ok"] for r in results)
    print(f"\n=== store_context {'ON' if on else 'OFF'}: {passed}/{len(CASES)} ===")
    # Artifact naming carries the whole arm configuration, not just the mode
    # this script happens to set. Deriving the filename from store_context alone
    # meant two arms that varied a different flag both wrote discriminate_off
    # and the second silently destroyed the first, losing the raw text needed to
    # audit the comparison. The config is read back from the daemon so the name
    # reflects what actually ran, not what was requested.
    try:
        cfg = json.loads(urllib.request.urlopen(f"{BASE}/v1/settings", timeout=15).read())
    except Exception:
        cfg = {}
    parts = [f"store-{'on' if on else 'off'}",
             f"annotate-{'on' if cfg.get('annotate_values') else 'off'}",
             f"ungate-{'on' if cfg.get('entity_routing') or cfg.get('ungate_dense') else 'off'}",
             f"model-{str(cfg.get('model', 'unknown')).replace(':', '_').replace('/', '_')}"]
    tag = os.environ.get("ARM_TAG", "")
    if tag:
        parts.append(f"run-{tag}")
    name = "_".join(parts)
    out_path = f"/tmp/discriminate_{name}.json"
    with open(out_path, "w") as f:
        json.dump({"arm": parts, "settings": cfg, "results": results}, f, indent=2)
    print(f"[artifact: {out_path}]")


if __name__ == "__main__":
    main()
