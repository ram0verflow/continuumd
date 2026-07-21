#!/usr/bin/env python3
"""Endurance run against a live `continuum serve` instance.

The claim under test: the context window never dies. Facts planted in the
first twenty turns must survive a hundred turns of unrelated chatter on a
1200 token window, recalled at the end through retrieval alone. The window
must stay at or under budget the whole way.

Usage: python3 endurance.py [port] [filler_turns]
Writes endurance_log.jsonl and prints a summary.
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3211
FILLER = int(sys.argv[2]) if len(sys.argv) > 2 else 100

FACTS = [
    ("my gym locker code is K47-KESTREL", "kestrel"),
    ("my dentist is called Dr. Marrow", "marrow"),
    ("my cat is named Bagel", "bagel"),
    ("I drive a green 2009 Corolla", "corolla"),
    ("my sister lives in Tromso", "tromso"),
    ("my favorite dish is mushroom shakshuka", "shakshuka"),
    ("my wifi password is PLUM-ORBIT-88", "plum-orbit-88"),
    ("my piano teacher is Miss Ilona", "ilona"),
    ("I am allergic to cashews", "cashew"),
    ("my marathon is on October 19th", "october 19"),
]

PROBES = [
    "what is my gym locker code?",
    "what is my dentist's name?",
    "what is my cat's name?",
    "what car do I drive?",
    "where does my sister live?",
    "what is my favorite dish?",
    "what is my wifi password?",
    "who is my piano teacher?",
    "what am I allergic to?",
    "when is my marathon?",
]

TOPICS = [
    "work was busy today, lots of meetings about the quarterly report",
    "I watched a documentary about deep sea creatures last night",
    "thinking about repainting the kitchen, maybe a light grey",
    "the weather has been strange lately, rain then sun then rain",
    "I tried a new running route through the park this morning",
    "my neighbor is learning the trumpet and it is going badly",
    "been reading a novel about a lighthouse keeper, quite slow but good",
    "the coffee machine at the office broke again",
    "planning to visit the farmers market this weekend",
    "my phone battery barely lasts half a day now",
    "there was a power cut for an hour this evening",
    "I keep meaning to fix the squeaky door hinge",
    "saw a great blue heron by the river today",
    "the gym was packed, could barely get on a treadmill",
    "trying to drink more water, bought a huge bottle for my desk",
    "the bus was twenty minutes late again",
    "made soup from scratch, turned out too salty",
    "my houseplant is finally growing a new leaf",
    "watched the game last night, terrible refereeing",
    "thinking about learning some basic woodworking",
]


def chat(msg):
    body = json.dumps({"message": msg}).encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}/api/chat", body, {"Content-Type": "application/json"}
    )
    t0 = time.time()
    raw = urllib.request.urlopen(req, timeout=600).read().decode()
    latency = time.time() - t0
    done = {}
    for frame in raw.split("\n\n"):
        if frame.startswith("data: "):
            ev = json.loads(frame[6:])
            if ev.get("t") == "done":
                done = ev
    return done, latency


def main():
    log = open("endurance_log.jsonl", "w")
    turn = 0
    max_used = 0
    over_budget = 0
    t_start = time.time()

    def run(msg, phase, expect=None):
        nonlocal turn, max_used, over_budget
        turn += 1
        done, latency = chat(msg)
        p = done.get("pressure", {})
        used, budget = p.get("used", 0), p.get("budget", 1)
        max_used = max(max_used, used)
        if used > budget:
            over_budget += 1
        hit = None
        if expect is not None:
            hit = expect.lower() in done.get("reply", "").lower()
        rec = {"turn": turn, "phase": phase, "latency_s": round(latency, 1),
               "used": used, "budget": budget, "level": p.get("level", ""),
               "evictions": p.get("evictions", 0), "loaded": done.get("loaded", 0),
               "expect": expect, "hit": hit, "reply": done.get("reply", "")[:110]}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        mark = "" if hit is None else (" RECALL OK" if hit else " RECALL MISS")
        print(f"[{turn:>3}] {phase:<6} {latency:5.1f}s used {used:>4}/{budget} "
              f"ev {p.get('evictions', 0):>3}{mark}", flush=True)
        return rec

    # Plant facts across the first stretch, chatter in between.
    fi, ti = 0, 0
    for i in range(20):
        if i % 2 == 0 and fi < len(FACTS):
            run(FACTS[fi][0], "fact")
            fi += 1
        else:
            run(TOPICS[ti % len(TOPICS)] + ".", "filler")
            ti += 1

    # The grind: unrelated chatter to force eviction over and over.
    for i in range(FILLER):
        run(TOPICS[ti % len(TOPICS)] + f" (day {i + 1})", "filler")
        ti += 1

    # Recall probes.
    hits = 0
    results = []
    for probe, (_, expect) in zip(PROBES, FACTS):
        r = run(probe, "probe", expect)
        hits += bool(r["hit"])
        results.append((expect, r["hit"], r["reply"]))

    mins = (time.time() - t_start) / 60
    print("\n========== ENDURANCE RESULT ==========")
    print(f"turns: {turn} over {mins:.0f} minutes")
    print(f"window: max used {max_used} of budget, {over_budget} turns over budget")
    print(f"recall of facts planted {FILLER}+ turns ago: {hits}/{len(FACTS)}")
    for e, h, r in results:
        print(f"  [{'ok' if h else 'MISS'}] {e}: {r[:70]}")
    print("=======================================")


if __name__ == "__main__":
    main()
