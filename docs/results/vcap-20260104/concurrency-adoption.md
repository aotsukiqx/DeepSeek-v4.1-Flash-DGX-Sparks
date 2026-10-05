# Concurrency tier campaign (2026-10-05): MRR 16 -> 32 ADOPTED

Question: can a higher MAX_RUNNING_REQUESTS lift aggregate throughput without hurting
code aggregate or single-stream? The earlier c32 rejection (code agg -8 %) turned out
to be an artifact: the 20-32-row graph tiers ran UNRACED heuristic b12x plans. With
DSV41_MOE_B12X_NEXT_TUNE_ROWS=6:5+6 (the multi-family knob, adapter commit 7a35a83f)
racing every tier the engine captures, the curve is monotonic:

| tier | prose aggregate | code aggregate | per-stream (code) |
|---|---:|---:|---:|
| c16 | 197-211 | 716-734 | 45.7 |
| c20 | 219 | 801 | 40.8 |
| c24 | 239-243 | 869-888 | 37-38 |
| c28 | 256 | 945 | 34.3 |
| **c32** | **307 (+56 % vs c16)** | **1056 (+47 % vs c16)** | 33.7 |

Gates: c1 unchanged (prose 48.6 / code 118.2, the day's band); qeval paired BROKE=0
(median -1.9 %, in band); c16 aggregate not regressed. Cost: KV pool 6.44M -> 5.30M
tokens (-18 %; still ~11 concurrent 470k contexts). Boot graph memory grows with the
tier ladder - that is the whole KV cost.

Adopted production changes (.env.tp4 / .env.tp4.example):
MAX_RUNNING_REQUESTS=32, CUDA_GRAPH_MAX_BS_DECODE=32,
EXTRA_CONTAINER_ENV += DSV41_MOE_B12X_NEXT_GRAPH_BS=1,...,16,20,24,28,32 and
DSV41_MOE_B12X_NEXT_TUNE_ROWS=6:5+6.

Lessons: (1) the earlier c32-and-clocks.md rejection is OVERTURNED - never bench a
concurrency tier whose graph shapes have unraced plans; racing must cover every tier
you measure. (2) The saturation "knee" was plan quality, not bandwidth: at c32 the
raced line is +58 % over the unraced one for code. The curve had not flattened at 32;
tiers beyond need traffic that justifies them (observed peak today: 16 concurrent).

Artifacts: bench-c24-{c1,tiers}.json, bench-c32r-{c1,tiers}.json, qeval-c24/-c32r.
