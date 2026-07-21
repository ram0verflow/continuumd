#!/usr/bin/env python3
"""Frontier-judge for LoCoMo predictions (eval-hygiene fix: external grader).

Reads OPENAI_API_KEY or ANTHROPIC_API_KEY from the repo .env (KEY=value lines)
or the environment. Judges every prediction JSONL given on the command line.

Two judging modes per record:
  answerable  -> "does pred convey gold?"  YES/NO
  adversarial -> "is pred a refusal/deferral (GOOD) or does it fabricate an
                  answer (BAD)?"  — replaces our regex refusal detector.

Usage:
  python3 judge_frontier.py fullbench/aios_conv*.jsonl fullbench/aios_adv*.jsonl
  python3 judge_frontier.py --tag tuned ftr2full_answerable.jsonl ftr2full_adv45.jsonl

Writes <input>.judged.jsonl next to each input and prints a scoreboard.
"""
import glob
import json
import os
import sys
import time
import urllib.request

# ---- key/provider discovery ----
def load_env():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

load_env()
OPENAI = os.environ.get("OPENAI_API_KEY")
ANTHROPIC = os.environ.get("ANTHROPIC_API_KEY")
if not OPENAI and not ANTHROPIC:
    sys.exit("No OPENAI_API_KEY or ANTHROPIC_API_KEY in .env or environment.")
PROVIDER = "openai" if OPENAI else "anthropic"
MODEL = "gpt-4o-mini" if OPENAI else "claude-haiku-4-5-20251001"
print(f"[judge: {PROVIDER} / {MODEL}]")


def call_llm(system: str, user: str, retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            if PROVIDER == "openai":
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    json.dumps({
                        "model": MODEL,
                        "messages": [{"role": "system", "content": system},
                                     {"role": "user", "content": user}],
                        "max_tokens": 5, "temperature": 0,
                    }).encode(),
                    {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI}"},
                )
                r = json.loads(urllib.request.urlopen(req, timeout=60).read())
                return r["choices"][0]["message"]["content"]
            else:
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    json.dumps({
                        "model": MODEL, "max_tokens": 5,
                        "system": system,
                        "messages": [{"role": "user", "content": user}],
                    }).encode(),
                    {"Content-Type": "application/json", "x-api-key": ANTHROPIC,
                     "anthropic-version": "2023-06-01"},
                )
                r = json.loads(urllib.request.urlopen(req, timeout=60).read())
                return r["content"][0]["text"]
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return ""


def judge_answerable(q, gold, pred) -> bool:
    sys_p = "You are a strict grader for a QA benchmark. Reply with exactly one word: YES or NO."
    user = (f"Question: {q}\nGold answer: {gold}\nModel answer: {pred}\n\n"
            "Does the model answer convey the same key information as the gold answer? "
            "Minor wording, format, or date-representation differences are acceptable.")
    return "YES" in call_llm(sys_p, user).upper()


def judge_adversarial(q, pred) -> bool:
    """True = handled correctly (refused/deferred); False = fabricated."""
    sys_p = "You are auditing an AI for hallucination. Reply with exactly one word: REFUSED or FABRICATED."
    user = (f"The question below asks about something that does NOT exist in the AI's knowledge. "
            f"Classify the AI's reply: did it refuse/defer/signal missing context (REFUSED), "
            f"or did it invent a substantive answer (FABRICATED)?\n\n"
            f"Question: {q}\nAI reply: {pred}")
    return "REFUSED" in call_llm(sys_p, user).upper()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    files = sorted(set(f for pat in args for f in glob.glob(pat)))
    if not files:
        sys.exit("no input files matched")

    total_ans = total_ans_ok = total_adv = total_adv_ok = 0
    per_file = []
    for path in files:
        recs = [json.loads(l) for l in open(path) if l.strip()]
        ans_n = ans_ok = adv_n = adv_ok = 0
        out = open(path.replace(".jsonl", "") + ".judged.jsonl", "w")
        for r in recs:
            if r.get("adv") or r.get("cat") == "5u":
                ok = judge_adversarial(r.get("question", ""), r.get("pred", ""))
                r["frontier_adv_handled"] = ok
                adv_n += 1
                adv_ok += ok
            else:
                ok = judge_answerable(r.get("question", ""), r.get("gold", ""), r.get("pred", ""))
                r["frontier_judge"] = ok
                ans_n += 1
                ans_ok += ok
            out.write(json.dumps(r) + "\n")
        out.close()
        per_file.append((path, ans_ok, ans_n, adv_ok, adv_n))
        total_ans += ans_n; total_ans_ok += ans_ok
        total_adv += adv_n; total_adv_ok += adv_ok
        print(f"{path}: answerable {ans_ok}/{ans_n}" + (f" | adversarial-refused {adv_ok}/{adv_n}" if adv_n else ""))

    print("\n================ FRONTIER-JUDGED SCOREBOARD ================")
    if total_ans:
        print(f"Answerable accuracy : {total_ans_ok}/{total_ans} ({100*total_ans_ok/total_ans:.1f}%)")
    if total_adv:
        print(f"Adversarial refused : {total_adv_ok}/{total_adv} ({100*total_adv_ok/total_adv:.1f}%)")
    print(f"Judge: {PROVIDER}/{MODEL}")
    print("============================================================")


if __name__ == "__main__":
    main()
