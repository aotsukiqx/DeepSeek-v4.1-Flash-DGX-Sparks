# b12x_next upgrade a7d7d29b -> 1.5.0 (e4084d2e), 2026-10-04: adopted

Upstream survey before tuning (avoid reinventing): between our 2026-09-24 pin
and the 1.5.0 release (2026-09-30) b12x gained 20 commits, among them the
block-quantized launch-heuristic rework derived from reuse and occupancy
(0d6600e6), the autotune GC-exclusion fix (#417), portable tuning winner
artifacts (a489f972), FP32 dense-MLA split partials (#441), Trellis route-pack
retention (#442), and the CUTLASS DSL 4.6.2 -> 4.7.1 bump. The
compact-n64-m64 patch applied to 1.5.0 unmodified; every adapter-facing symbol
(api/plan/prepare/bind, `_FusedMoeState`, `_compact_n64_tiles`, `PreparedCall`)
survived; `scripts/fetch_runtime.sh` verified end to end locally before build.

## Fleet A/B (same probes as the verify-cap campaign, greedy, median of 4)

| image | pin | prose tok/s | code tok/s | accept-len | KV pool |
|---|---|---:|---:|---:|---:|
| old (2 boots) | a7d7d29b | 56.09 / 56.89 | 120.0 | 2.815 | 6.02M |
| **new (adopted)** | e4084d2e | **57.31** | 120.0 | 2.809 | **6.35M** |

Prose +1-2 % (old-arm boot noise ±0.8), code flat, acceptance unchanged (the
gain is step time, not acceptance). qeval paired vs the pre-upgrade baseline:
55 primary tasks, 0 broke / 0 fixed (p=1.000), median single-stream
82.8 -> 83.7 tok/s (+1.1 %). Boot gate green on all lines (hc_fused first-call
bit-identical, RoCEnante health counters zero-error through the benches).

Adopted for currency, not the headline number: the pin now tracks the upstream
heuristic/autotune line, which is the prerequisite for the next step (per-shape
tuning through b12x's portable winner artifacts instead of a homegrown sweep).
Rollback: images tagged `dsv41-4x-spark:canary-roce-pre-b12x150` on all four
nodes; repo tag `pre-b12x150`.

## Baseline profile (old pin, kept for the next step's comparison)

`baseline-trace.json.gz` (rank 0, 45 steps + a 384-token prose request,
torch profiler window): b12x W4A8 MoE phase1+phase2 dominate GPU busy
(~35 % combined), dense MXFP8 GEMMs second, attention/NCCL/RoCE behind. The
next graph-shape lever targets exactly those two groups. Superseded below: plan selection is pinned, see the pins-off arm.

## Pins-off arm (same day): the two plan pins stay mandatory

`DSV41_MOE_B12X_NEXT_PLAN_TABLE=1-8=heur` + `M64_MIN_CAP=0` (pure 1.5.0
heuristic, everything else production): prose 30.5 (-47 %), code 79.2 (-34 %).
The 1.5.0 reuse/occupancy heuristic rework (0d6600e6) still selects the `micro`
plan at <= 8 rows on the compact-N64 geometry - the measured micro/dynamic gap
the pins were installed for (523 -> 349 us per call at M=6) did not close.
Restored and verified (code 120.0 back). Consequence for the tuning step: the
known heuristic misses are already pinned, the verify-row family
({6,12,...,96}) already races at load, and the remaining unraced surface
(draft-row caps 5xbs >= 10, prefill ladder) sits on small slices of the step -
per-shape tuning is closed as below the effort line. The next decode lever is
the graph-shape one (gamma widening), not plan selection.
