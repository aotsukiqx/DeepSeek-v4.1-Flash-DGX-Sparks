#!/usr/bin/env python3
"""Fit a verify-cap live-length rule on an uncensored DSV41_VERIFY_CAP_LOG trace.

Input: vcap-log.bin (7 float32 a request-step: conf1..5, live, accepted) plus the
manifest scripts/verify/vcap_capture.py wrote. The trace boot must have run
DSV41_VERIFY_CAP=conf:0.0 (uncut, live always maximal) so each step's accepted
prefix length A is fully observed.

Rule family (matches adapter/verify_cap.py): live L' = 1 + the number of drafts
j = 2..5 in a row whose conf_j >= thr; draft 1 is always verified. A threshold is
fitted per phase by counterfactual replay: a step capped at L' commits
min(A, L') + 1 tokens and the step costs a + b * L' ms (a, b from production
step measurements; sweep b to check sensitivity).

Objective: 1000 * E[min(A, L') + 1] / (a + b * E[L']). Caveat, inherent to offline
replay: when L' cuts a chain that would have continued, the next step's anchor
differs from the observed one; the per-step objective ignores that drift.

Usage:
  python3 scripts/verify/vcap_fit.py --log <vcap-log.bin> --manifest <manifest.json> \
      [--b-ms 3.0] [--a-ms 25.0]
"""
import argparse
import json

import numpy as np


def load(log_path, manifest_path):
    raw = np.fromfile(log_path, dtype=np.float32).reshape(-1, 7)
    # accepted column: the engine's correct_len includes the anchor when its max
    # is gamma + 1; normalise to drafts accepted (0..gamma)
    if raw.size and raw[:, 6].max() > 5:
        raw[:, 6] -= 1
    phases = []
    for p in json.load(open(manifest_path))["phases"]:
        rows = raw[p["start_byte"] // 28:p["end_byte"] // 28]
        ok = (rows[:, :5] >= 0).all(1) & (rows[:, :5] <= 1.0001).all(1) \
            & (rows[:, 5] >= 1) & (rows[:, 6] >= 0)
        phases.append((p, rows[ok]))
    return phases


def rule_live(conf, thr):
    """conf (n, 5) -> live length per the adapter's prefix rule (draft 1 free)."""
    live = np.ones(len(conf), dtype=np.int64)
    for j in range(1, 5):
        live = np.where((live == j) & (conf[:, j] >= thr[j]), j + 1, live)
    return live


def replay(conf, accepted, thr):
    """Mean committed tokens and mean live length for a threshold set."""
    live = rule_live(conf, thr)
    return (np.minimum(accepted, live) + 1).mean(), live.mean()


def fit_phase(rows, a_ms, b_ms):
    conf, accepted = rows[:, :5], rows[:, 6].astype(np.int64)
    prod_tok, prod_live = replay(conf, accepted, [0.0, 0.1, 0.1, 0.1, 0.1])
    prod_rate = 1000.0 * prod_tok / (a_ms + b_ms * prod_live)
    best = None
    for thr in np.arange(0.02, 0.91, 0.02):
        tok, live = replay(conf, accepted, [0.0, thr, thr, thr, thr])
        rate = 1000.0 * tok / (a_ms + b_ms * live)
        if best is None or rate > best[1]:
            best = (float(thr), rate, tok, live)
    return best, prod_rate


def calibration(rows):
    """P(the draft at position j is accepted | its conf), over reached steps."""
    conf, accepted = rows[:, :5], rows[:, 6].astype(np.int64)
    out = {}
    for j in range(5):
        reached = accepted >= j
        if reached.sum() < 20:
            continue
        hit = (accepted > j)[reached]
        c = conf[reached, j]
        bins = np.clip((c * 10).astype(int), 0, 9)
        out[f"draft{j + 1}"] = {
            f"{b / 10:.1f}": {"n": int((bins == b).sum()),
                              "p": round(float(hit[bins == b].mean()), 3)}
            for b in range(10) if (bins == b).sum() >= 5}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--a-ms", type=float, default=25.0)
    ap.add_argument("--b-ms", type=float, default=3.0)
    args = ap.parse_args()

    for p, rows in load(args.log, args.manifest):
        if not len(rows):
            print(f"{p['label']}: no rows")
            continue
        acc = rows[:, 6]
        (thr, rate, tok, live), prod_rate = fit_phase(rows, args.a_ms, args.b_ms)
        print(f"\n== {p['label']}  (n={len(rows)}, T={p['temperature']}, "
              f"A mean {acc.mean():.2f} p50 {np.median(acc):.0f} "
              f"p90 {np.percentile(acc, 90):.0f})")
        print(f"   best thr {thr:.2f}: E[tok] {tok:.2f} live {live:.2f} -> "
              f"{rate:.1f} tok/s | conf:0.1 replay {prod_rate:.1f} tok/s "
              f"({100 * (rate / prod_rate - 1):+.1f}%)")
        print(f"   calibration {json.dumps(calibration(rows))}")


if __name__ == "__main__":
    main()
