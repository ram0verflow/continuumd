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
fresh AIOS_HOME each time, and compare:

  AIOS_HOME=/tmp/aios-disc ./target/release/aios-daemon --port 4316 &
  python3 stress_discriminate.py 4316 off
  python3 stress_discriminate.py 4316 on     # fresh state again first
"""

import json
import sys
import urllib.request

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
        "needles": ["october 21"],
        "forbidden": ["october 14"],
    },
    {
        "name": "synthesis: arithmetic",
        "mentions": [
            "My rent is 1800 a month.",
            "My landlord told me everything goes up by 200 starting next month.",
        ],
        "question": "what will my rent be next month?",
        "needles": ["2000", "2,000"],
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
        "needles": ["zed"],
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
        "needles": ["8:45"],
        "forbidden": ["9:00", "9:30"],
    },
    {
        "name": "cross-branch: over allowance",
        "mentions": [
            "My API plan allows 50 thousand requests per month.",
            "I've burned through about 62 thousand calls so far this month.",
        ],
        "question": "am I over my monthly API allowance?",
        # A verdict, not an echo of the question: "over" alone false-passed
        # a reply that merely restated the question and asked for the data.
        "needles": ["over by", "you're over", "you are over", "exceed", "12 thousand", "12,000", "12000"],
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
        ok = any(n in low for n in c["needles"])
        stale = any(f in low for f in c["forbidden"]) and not ok
        results.append({"case": c["name"], "reply": reply, "ok": ok, "stale": stale,
                        "store_topics": insp.get("store_topics"),
                        "loaded": insp.get("loaded")})
        mark = "PASS" if ok else ("STALE" if stale else "FAIL")
        print(f"  [{mark:5}] st={insp.get('store_topics')} {c['name']:28} -> {reply[:70]!r}")

    passed = sum(r["ok"] for r in results)
    print(f"\n=== store_context {'ON' if on else 'OFF'}: {passed}/{len(CASES)} ===")
    with open(f"/tmp/discriminate_{MODE}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
