# gamma=8 probe (2026-10-04): widening is real for code; the confidence head is the blocker

Three-boot decisive experiment on the P1 "widen the verify window" route.

## Boot 1 - gamma=8, full production graph set: OOM

CUDA-graph capture at verify rows 9xbs up to bs 16 exceeds the memory budget
(the gamma=5 line leaves ~15 GB free at boot). Fits at bs<=8 graphs.

## Boot 2 - gamma=8 + verify_cap: architectural crash

`Capture cuda graph failed: mat1 and mat2 shapes cannot be multiplied (40x8448
and 5376x1)` - the DSpark confidence head's projection is trained for exactly
the checkpoint's native block of 5 positions: its input is
`gamma*1024 + 256` wide (5 -> 5376, 8 -> 8448). gamma != 5 cannot run the
confidence head as shipped. The engine itself only warns on the gamma/block_size
mismatch; the head is what breaks, and only in the static mode verify_cap
forces (the stock engine path skips it).

## Boot 3 - gamma=8, verify_cap OFF, graphs bs<=8: boots clean, decisive numbers

| metric | gamma=5 prod | gamma=8 probe |
|---|---:|---:|
| code tok/s | 120.0 | **134.8 (+12.3 %)** |
| prose tok/s | 57.3 | 39.2 (-31 %, no dead-row masking: 9 live rows/step) |
| code accept (p90 of window) | ~5.4 of 6 | **7.3-7.6 of 9** |

The untrained positions 6-8 of the parallel block extrapolate strongly on code
(+2 accepted tokens/step); prose never passes position ~2 and pays 7 dead live
rows. KV pool 6.66M at the reduced graph set.

## Plan that followed (kept for the record)

Adapter shim, not an engine fork: hook the model's compute_confidence, extend the
verify_cap rule with a position-5 threshold, trim the graph ladder for memory, gate on
qeval (greedy texts differ across gamma arms - 9-row verify changes batched rounding).

## Shim implemented and measured (same day, commits e1fb298..90d7ad4)

Three lessons to a working widened line: (1) sitecustomize's EngramFinder whitelist
must name the hooked module; (2) the wide argument is x_post_hc (the gamma*1024 markov
context, measured (64, 5120) = [bs*gamma, D] at runtime), not the 256-wide token
embedding; (3) the engine's own compute_confidence views everything through the draft
checkpoint's native gamma=5, so the adapter replaces the method: first five block
positions + anchor-and-four prev sequence through the trained head, zero-padded to
[bs, 8] (columns past five ignored by the live rule; rows past five ride
DSV41_VERIFY_CAP_EXT, default 0.5 - on the gamma=5 trace cum5 separates code p50 0.90
from prose p90 0.015).

| arm | prose | code | notes |
|---|---:|---:|---|
| gamma=8, no cap (probe) | 39.2 | 134.8 | 9 live rows/step, no dead-row masking |
| gamma=8 + EXT shim | **49.9** | **134.8** | cap costs nothing on code; prose keeps the fixed 9-row graph cost |
| gamma=5 production | 57.3 | 120.0 | |

qeval paired: BROKE 1 (math_m9: 121 vs 38 - a numeric task flipping under the changed
verify-row rounding), median -27.8 % on the prose-heavy suite. Verdict: **not adopted**;
the line stays env-gated (DSPARK_BLOCK_SIZE=8 + the shim, default off). Making it
universal needs the runner to hold two verify graph families per bs and pick per batch
by the live length (prose then replays 6-row graphs and pays nothing) - engine runner
work, the natural follow-up. Code-only deployments can take +12 % today by setting the
env.

Artifacts: `bench-gamma8-probe.json`; accept-len distribution from the boot-3
engine log (mean 2.96 mixed, bimodal p50 1.93 / p90 7.30).
