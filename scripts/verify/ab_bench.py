#!/usr/bin/env python3
"""A/B decode bench for DSV41 verify-cap arms (same-caliber probes, idle fleet).

Five sequential greedy completions per prompt class (384 tokens, thinking off),
discarding the first, reporting median tok/s from completion wall time, plus the
engine's own accept length over the same window (joined by the caller from the
head log). Prompts are the vcap_capture pair, so greedy texts are comparable
across arms and against the trace manifest.

Usage: python3 scripts/verify/ab_bench.py --out <json> [--reps 5]
"""
import argparse
import json
import statistics
import time

from vcap_capture import PROSE, CODE, complete


def bench(label, prompts, reps, url, model):
    runs = []
    for i in range(reps):
        r = complete(url, model, prompts[i % len(prompts)], 0.0, 384)
        r["tok_s"] = round(r["completion_tokens"] / r["wall_s"], 2)
        r["rep"] = i
        runs.append(r)
        print(f"[{label}] rep {i + 1}/{reps} {r['tok_s']} tok/s sha {r['text_sha']}", flush=True)
    kept = runs[1:]  # discard the first (cold)
    return {"label": label, "runs": runs,
            "median_tok_s": round(statistics.median(r["tok_s"] for r in kept), 2),
            "sha": kept[-1]["text_sha"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.2.23:8000")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    t0 = time.strftime("%H:%M:%S")
    out = {"started": t0,
           "prose": bench("prose", PROSE, args.reps, args.url, args.model),
           "code": bench("code", CODE, args.reps, args.url, args.model)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"prose {out['prose']['median_tok_s']} | code {out['code']['median_tok_s']} -> {args.out}")


if __name__ == "__main__":
    main()
