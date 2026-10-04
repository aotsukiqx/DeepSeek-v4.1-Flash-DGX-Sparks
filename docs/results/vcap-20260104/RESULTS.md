# verify_cap live-length refit (2026-10-04): negative result, production conf:0.1 kept

Question: does the production verify-cap threshold (`DSV41_VERIFY_CAP=conf:0.1`)
leave speed on the table on the TP4 line? Method: uncensored trace capture at
`conf:0.0` (logging boot), offline counterfactual refit, then on-fleet A/B of the
fitted candidates against production.

## Trace

`vcap-log.bin` + `manifest.json`: 2,360 verify steps, 6 phases
({greedy, T=0.7, T=1.0} × {prose, code}), 3 × 384-token completions each,
bs=1, idle fleet, 2026-10-04. Boot arm `conf:0.0` + `DSV41_VERIFY_CAP_LOG`
(diagnostic-verified: 7 × float32 per request-step: conf1..5, live, accepted;
`correct_len` includes the anchor, the fitter normalises).

Measured per-position acceptance is temperature-insensitive (prose A mean
1.04/1.04/0.98; code 4.46/4.47/4.63 at T=0/0.7/1.0): DSpark's confidence is
already calibrated under sampling (block verification keeps sampled rows exact),
so the GLM-style noise-aware discount has nothing to buy here.

## Offline fit vs fleet A/B (greedy, 384 tokens, median of 4 after discard, bs=1)

| arm | SPEC | prose tok/s | code tok/s | accept-len (mixed window) |
|---|---|---:|---:|---:|
| C | conf:0.0 (no cap) | 50.2 | 120.0 | 2.960 |
| **B (production)** | **conf:0.1** | **56.1** | **120.0** | **2.815** |
| A | conf:0.42 (fitted) | 52.3 | 120.0 | 2.595 |

The offline fit (cumprod-prefix rule, `E[min(A,L')+1] / (a + b·E[L'])`, b swept
2-4 ms) predicted +4.9 % prose at thr 0.38-0.44. The fleet measured **-7 %**.
Both directions from 0.1 lose: 0.0 → -10.5 %, 0.42 → -6.8 %.

## Why the counterfactual was wrong: the marginal live row is convex, not linear

Inside the fixed `[bs, 6]` verify graph the dense layers always compute 6 rows;
`router_live` makes dead rows reuse the anchor's experts. Live rows read their
own (doomed, near-distinct) experts, so the step cost as a function of live
length is **convex**: cutting prose from live 5 → ~2.2 removes many distinct
expert reads (C vs B: +5.1 % tokens bought -11.7 % wall), while 2.2 → 1.5 buys
~nothing (A vs B: -7.8 % tokens for ~-1 % step). A linear b fits neither end.
The k=3↔k=5 boot comparisons (E3/E10) that motivated b ≈ 3 ms change the graph
shape, not the in-graph mask — the two costs are unrelated.

**Conclusion: production `conf:0.1` sits at the measured optimum for this line
(prose peak, code flat). Nothing adopted; fleet restored to the pre-campaign
`.env.tp4` (backup `.env.tp4.pre-vcaplog`).** The b12x/router_live dead-row
design, not the threshold rule, is what makes 0.1 near-optimal — capping harder
only discards tokens.

## Campaign notes

- **Never `rm` a trace file a long-lived process holds open.** The logger opens
  once and keeps the handle; `rm` unlinks and all further writes go to the
  deleted inode (invisible to `stat`). Baseline byte offsets instead
  (`vcap_capture.py` does this now). This cost two capture rounds and one
  diagnostic hot-patch cycle before being understood.
- The image's `__pycache__` was checked and is valid (pyc header matches the
  source mtime/size); the logger works from a fresh serve once the rm mistake
  is absent.
- Boot hygiene that worked every round: rustup reaper
  (`scripts/verify/boot_hygiene.py reap`) before each serve; orphan check after
  each stop; `EXTRA_CONTAINER_ENV`-only arms keep autotune caches valid (SPEC is
  volatile).
- qeval `before` baseline for the production arm: 75 tasks, 0 truncated,
  median 82.8 tok/s (`qeval-before.json`, run from spark-a89f). No `after` run:
  nothing was promoted.
- Artifacts: `manifest.json`, `vcap-log.bin` (75,656 bytes), `bench-{A,B,C}.json`,
  `capture-run.log`.

## Tooling kept

`scripts/verify/vcap_capture.py` (offset-baselined trace driver),
`scripts/verify/vcap_fit.py` (reader + counterfactual fitter; its linear-b
objective is superseded by this finding - treat b as convex or fit b from a
C-vs-B pair), `scripts/verify/ab_bench.py` (same-caliber A/B probes),
`scripts/verify/boot_hygiene.py` (reaper + orphan check).
