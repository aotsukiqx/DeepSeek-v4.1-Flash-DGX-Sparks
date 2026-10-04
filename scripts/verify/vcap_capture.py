#!/usr/bin/env python3
"""Drive the DSV41_VERIFY_CAP_LOG trace capture against a serving fleet.

Runs one request at a time (bs=1) through the OpenAI API and records the byte
offset of /state/vcap-log.bin inside dsv41-head before and after each phase, so
scripts/verify/vcap_fit.py can segment the 7-float32 records (conf1..5, live,
accepted) by workload and temperature. The boot must run DSV41_VERIFY_CAP=conf:0.0
(uncut) so acceptance is uncensored; decode is ~8x slower with the sync-per-step
logger, which is fine - this run measures distributions, not speed.

Usage (from the Mac, fleet idle, capture boot already healthy):
  python3 scripts/verify/vcap_capture.py --head colin@192.168.2.23 \
      --out docs/results/vcap-<date>/manifest.json [--reps 3]
"""
import argparse
import json
import subprocess
import time
import urllib.request

PROSE = [
    "Write a short essay about the keepers of a remote lighthouse on a rocky "
    "coast, their daily routines, the storms they weather, and the ships they "
    "guide safely home. Aim for vivid, concrete prose.",
    "Describe a small fishing village at dawn: the boats leaving the harbour, "
    "the market on the pier, the children on their way to school, and one old "
    "fisherman who stays behind. Tell it as a story.",
]
CODE = [
    "Implement an LRU cache in Python with get and put methods, O(1) operations, "
    "a capacity argument, and a small test suite using assert. Output code only.",
    "Write a Python function that parses a CSV string with quoted fields into a "
    "list of rows, handles escaped quotes, and include tests with assert. "
    "Output code only.",
]
PHASES = [("greedy", 0.0), ("t07", 0.7), ("t10", 1.0)]


def log_bytes(head):
    """Size of the trace inside dsv41-head; 0 before the first logged step."""
    out = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", head,
         "docker exec dsv41-head stat -c %s /state/vcap-log.bin"],
        capture_output=True, text=True)
    return int(out.stdout.strip() or 0) if out.returncode == 0 else 0


def complete(url, model, prompt, temperature, max_tokens):
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, "top_p": 0.95, "thinking": False,
        "max_tokens": max_tokens, "stream": False,
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        res = json.load(r)
    usage = res.get("usage", {})
    return {"wall_s": round(time.time() - t0, 1),
            "completion_tokens": usage.get("completion_tokens"),
            "text_sha": str(hash(res["choices"][0]["message"].get("content", "")))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="user@head for offset probing")
    ap.add_argument("--url", default="http://192.168.2.23:8000")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--out", required=True, help="manifest.json path")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=384)
    args = ap.parse_args()

    manifest = {"url": args.url, "model": args.model, "reps": args.reps,
                "max_tokens": args.max_tokens, "row_bytes": 28, "phases": []}
    for label, temp in PHASES:
        for cls, prompts in (("prose", PROSE), ("code", CODE)):
            start = log_bytes(args.head)
            phase = {"label": f"{label}_{cls}", "temperature": temp, "top_p": 0.95,
                     "start_byte": start, "requests": []}
            for i in range(args.reps):
                prompt = prompts[i % len(prompts)]
                phase["requests"].append(
                    complete(args.url, args.model, prompt, temp, args.max_tokens))
                print(f"[{phase['label']}] rep {i + 1}/{args.reps} "
                      f"{phase['requests'][-1]}", flush=True)
            phase["end_byte"] = log_bytes(args.head)
            phase["rows"] = (phase["end_byte"] - phase["start_byte"]) // 28
            manifest["phases"].append(phase)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=1)
    total = sum(p["rows"] for p in manifest["phases"])
    print(f"manifest -> {args.out}; total rows {total}")


if __name__ == "__main__":
    main()
