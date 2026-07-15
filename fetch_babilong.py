#!/usr/bin/env python3
"""Fetch a BABILong slice for the aios runner.

BABILong hides bAbI facts inside long book text. It is a second benchmark
family for aios: the haystack is far bigger than the model window, so the
system has to find the needles by retrieval, which is exactly the claim.

Writes data/babilong_<len>.jsonl with one {task, input, question, target}
per line. Usage: fetch_babilong.py [length] [per_task]
"""
import json
import sys

from datasets import load_dataset

LENGTH = sys.argv[1] if len(sys.argv) > 1 else "64k"
PER_TASK = int(sys.argv[2]) if len(sys.argv) > 2 else 20
TASKS = ["qa1", "qa2", "qa3", "qa4", "qa5"]

out = open(f"data/babilong_{LENGTH}.jsonl", "w")
total = 0
for task in TASKS:
    ds = load_dataset("RMT-team/babilong", LENGTH, split=task)
    for i, row in enumerate(ds):
        if i >= PER_TASK:
            break
        out.write(json.dumps({
            "task": task,
            "input": row["input"],
            "question": row["question"],
            "target": row["target"],
        }) + "\n")
        total += 1
    print(f"{task}: {min(PER_TASK, len(ds))} samples", flush=True)
out.close()
print(f"wrote {total} samples to data/babilong_{LENGTH}.jsonl")
