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

## Verdict: build the widening as an adapter shim, not an engine fork

1. Hook the model's `compute_confidence` to truncate `markov_embed_stack` to
   5 positions at gamma=8 (the head keeps its trained view).
2. verify_cap rule: positions 1-5 exactly as today; positions 6-8 live only
   when the position-5 cumulative confidence clears a higher threshold - code
   extends to 8, prose stops at <=5 and keeps the dead-row masking.
3. Memory: bs-16 graphs at 9 rows need a trimmed graph ladder or a lower KV pin
   (one fitting boot; the bs<=8 set fits with room).
4. Gates: greedy texts differ across gamma arms (9-row verify changes batched
   rounding), so qeval paired is the correctness gate; within-boot greedy
   repeatability still applies.

Artifacts: `bench-gamma8-probe.json`; accept-len distribution from the boot-3
engine log (mean 2.96 mixed, bimodal p50 1.93 / p90 7.30).
