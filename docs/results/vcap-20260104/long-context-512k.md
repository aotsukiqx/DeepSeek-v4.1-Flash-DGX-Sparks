# 512k long-context prose measurements (2026-10-05)

Probes: seeded varied prose (no long n-gram repetition, char-capped 2.0M ~ 492k
tokens; the 820k probe used 3.2M chars). Fleet = production MRR=32.

## Acceptance at depth is CONTENT-dependent, not length-dependent

| task at ~492k | accept len | tok/s | step |
|---|---:|---:|---:|
| novel output ("summarize the mood") | 1.95-2.38 | 53.4 | ~39 ms |
| patterned continuation ("continue this chronicle") | **5.80-5.95** | **121-126** | ~45-47 ms |

The earlier 470k foreign-traffic high acceptance was content, not depth. Patterned
continuation SATURATES gamma=5's ceiling at 492k - these requests are exactly the
gamma=8 candidates (positions 6-8 extrapolate at ~0.8+ survival per the gamma=8
trace): expected accept ~7.5, tp ~150+ at the same step cost. Novel-output prose
stays at 2.0 - gamma widening buys nothing there.

## Step-cost growth with depth is modest and diffuse

Decode step 33 ms (short) -> ~39 ms (492k novel) / ~45-47 ms (492k patterned, with
6 live rows). No single category explodes: sparse MLA attention stays ~3.3 ms
(bounded top-512 selection works), MoE ~24 ms, dense ~11.5 ms - the growth spreads
across the indexer/page-table machinery ("other"). Prefill at depth: 3.6k tok/s mean
over 204 chunks (vs 5.8k short), MoE 26% + SP collectives 19% + sparse prefill 11%,
indexer scoring only 2.6% (chunked indexing holds).

## 512k-target recommendations

1. Patterned/continuation-heavy 512k traffic: run the gamma=8 EXT line (env-gated,
   measured +12% on code; at 492k patterned accept is ceiling-saturated, so the
   upside is larger). The dual-graph W2 makes this mixed-safe later.
2. Novel-output 512k prose: no acceptance lever (drafter-bound at 2.0); step growth
   is diffuse - no single kernel target. CONTEXT_LENGTH 1M -> ~532k may release
   per-request reservations (one boot to measure the pool delta).
3. MRR for a 512k-dedicated deployment: pool 5.30M / 512k ~ 10 concurrent
   full-context requests; if concurrency stays <= 10, MRR=16 recovers 1.14M tokens
   (2 more slots). Keep 32 for mixed short-burst traffic.
4. Prefix caching is a first-class lever at 512k: the cached-prefix continuation
   skipped the ~105 s prefill entirely (30 s wall for 3000 tokens). Multi-turn
   long-context sessions should reuse prefixes by design.

Artifacts: /tmp/longprose.py pattern, deep512-trace (prefill at depth),
decode512-trace (decode at depth, 42 steps).
