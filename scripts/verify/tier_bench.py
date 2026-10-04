#!/usr/bin/env python3
"""Aggregate decode bench at concurrency tiers (c1/c4/c16/c32...), idle fleet.

N concurrent greedy completions of the vcap_capture prose/code prompts, 384
tokens each; reports aggregate tok/s (total completion tokens / wall) and
per-stream median. Same-caliber probes for arm-vs-arm comparison only.

Usage: python3 scripts/verify/tier_bench.py --out x.json [--tiers 16,32]
"""
import argparse
import concurrent.futures as cf
import json
import statistics
import time

from vcap_capture import PROSE, CODE, complete


def tier(label, n, prompt, url, model):
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        runs = list(ex.map(lambda i: complete(url, model, prompt, 0.0, 384), range(n)))
    wall = time.time() - t0
    toks = sum(r["completion_tokens"] or 0 for r in runs)
    per = [round((r["completion_tokens"] or 0) / r["wall_s"], 1) for r in runs]
    out = {"label": label, "n": n, "aggregate_tok_s": round(toks / wall, 1),
           "per_stream_median": round(statistics.median(per), 1), "wall_s": round(wall, 1)}
    print(out, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.2.23:8000")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers", default="16,32")
    args = ap.parse_args()
    results = []
    for t in [int(x) for x in args.tiers.split(",")]:
        results.append(tier(f"prose_c{t}", t, PROSE[0], args.url, args.model))
        results.append(tier(f"code_c{t}", t, CODE[0], args.url, args.model))
    with open(args.out, "w") as f:
        json.dump({"results": results}, f, indent=1)


if __name__ == "__main__":
    main()
