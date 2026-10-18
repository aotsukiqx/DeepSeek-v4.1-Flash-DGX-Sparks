# Serving-layer robustness merge — 2026-10-18 (commit a50dbc0)

Merge of the MiaAI-Lab 2026-10-17 robustness wave + sglang#44751 backport onto our
620c76f2 production line, verified end-to-end on the 600k-optimal config
(MRR=32, chunk 4096, MAX_TOTAL_TOKENS=6700000 pin — unchanged).

## What landed

| Change | Source | Notes |
|---|---|---|
| `adapter/history_repair.py` | MiaAI-Lab bdf657c (PR #55, half) | broken assistant tool-call arguments in history → `{"_unparseable_arguments": raw}` instead of HTTP 400; agent death-loop closed |
| `encoding_compat.install_anthropic_protocol` | 45dde44 (PR #47) | `/v1/messages` accepts `thinking: {"type":"enabled"}` without `budget_tokens` (stock 620c76f2 400s) |
| `runtime/patch_encoding_dsv41.py` | 76f43c9+bc2d3ce (#15), anchor 1 re-derived | placeholder `[image]` token in text sanitizes instead of raising (was a self-sustaining /v1/messages 500 loop; 111/145 requests failed upstream) |
| `runtime/patch_encoding_dsv41_44751.py` | sglang#44751 (a038a0dc) backport | tool-result image order + current-turn thinking preserved; git-apply clean on 620c76f2, re-expressed as count==1 anchors |
| `boot.py with_fp8_gemm_backend` | 7cc2cc7 (#31) | EXTRA_SGLANG_ARGS can no longer silently drop `--fp8-gemm-backend` (-25..-45% decode if lost) |
| `--enable-metrics` in `.env.tp4.example` | 786859e | example aligned with the deployed `.env.tp4` (already had it) |

**Deliberately NOT taken**: upstream `adapter/loop_abort.py` (bdf657c). Their merged
version sets `finished_len` unconditionally post-hoc after `original()` — the exact
pattern that hard-crashed our scheduler on 10-15 (eff1094, invariant checker
available+evictable>total, 49 min down). Their dsv4.1-branch base (f80c91a4) likely
lacks that checker; ours does not. Our 21d7682 no-truncate loop_abort stays. Their
`detect_repetition_detail` richer logging (art/period) is a possible future port that
must keep `finished_len` untouched.

**Deferred** (ops hardening, no serving impact, next ff-sync window): remote.py
`--env-file` pre-scan (6df9f4b), pack re-share (1be7cb4 lineage), per-host SSH users
(dd02792/5685a5d), shared_pad_k TP_SIZE generalization (e3b4997 — TP4 output identical).

## Verification (2026-10-18, spark-eb1a fleet, image 979fcd336dbb on head)

- Static: 5 new tests PASS locally and in-image (`test_history_repair`,
  `test_anthropic_thinking`, `test_patch_encoding_dsv41`, `test_patch_encoding_44751`,
  `test_fp8_backend_flag`); launcher_build unchanged vs HEAD (2 pre-existing failures,
  present before this merge).
- Anchors proven against the image's built (post-rocenante) `encoding_dsv41.py`;
  patch scripts are the build's drift probes (exit non-zero on anchor mismatch).
- Boot gates: all lines green (shared-pad 576→640 ARMED, indexer native no-op,
  gamma=5/verify 6, max_total_num_tokens=5613824 chunk 4096, RoCEnante world=4,
  b12x_next armed EP1, hc_fused bit-identical, sm120_candidates + mm_tokenizer_compat
  ARMED, history-repair + anthropic-thinking install lines present).
- Wire smokes 4/4: thinking-no-budget 200 (blocks thinking+text), placeholder-in-text
  200, broken-args history 200 **with the model retrying corrected arguments**,
  vision-in-tool-result 200 with correct red-left/blue-right order.
- qeval paired (worker, c1, live traffic): PRIMARY 53/55 → 53/55, BROKE=0, p=1.000;
  jsons kept at spark-a89f:~/qeval-{before,after}-serving-20261018.json.
- 600k probes (45s-gap gated): cold prefill 3584.9 tok/s (10-17 same-stack baseline
  3645, single-run band), decode net-of-TTFT 41.4/42.5 tok/s (baseline 40.3) — parity,
  as expected for serving-layer-only changes. `/metrics` 200 with 428 series.

## Rollback

- Code: cluster `git reset --hard pre-serving-robust` (+ Mac reset to 21d7682).
- Image: 4× `docker tag dsv41-4x-spark:canary-roce-pre-serving dsv41-4x-spark:canary-roce` + serve.
- Env: `.env.tp4.pre-serving` (no env change was needed this round).

## Window bundle #2 (2026-10-18 下午, commits 11d2461..c374b80)

Diagnostics first (live log, 744 prefills/1h): NO giant fresh prefills in production
traffic; the only 600k revisits (bench probes) hit cached ratio 1.00; pool pressure
max full 0.11 / swa 0.09. Conclusions: **--swa-full-tokens-ratio REJECTED** (solves a
nonexistent residency problem, would only shrink the 600k line's headroom);
**cache_control breakpoint drift: no symptom** (watch only).

Merged: sglang#43006 backport (tool_choice=none honored; git-apply clean, script
byte-identical to upstream patch), boot.py sps_table_args SPS-Engram guard
(DSPARK_SPS_ALLOW_ENGRAM=1 escape hatch; exercised live: "not found; staying on
verify-all"), loop_abort detect_repetition_detail art/period logging (finished_len
semantics untouched, FINISH_ABORT still not ported), 10 upstream ops cherry-picks
ALL clean (remote.py --env-file, pack re-share set, per-host SSH users, shared_pad_k
TP_SIZE generalization + tests), .env.tp4.example reconciled (6700000 pin value, SPS
warning, swa note with rejection evidence), fast-load fragmentation runbook section.

Verification: full static suite green; 4/4 nodes content-verified; boot gates green
(shared-pad ARMED via TP_SIZE path); SMOKE 4/4 + new tool_choice=none wire probe PASS
(plain answer, no tool_calls); qeval paired vs morning json: **BROKE=0, 2 fixed**
(json_escape/json_count, the two chronic both-fail primaries); 600k: cold 3852.8
tok/s, decode net-of-TTFT 47.2/42.2 — at/above band, parity reported (no mechanism,
no claim). Rollback: tag `pre-bundle2` (=d351896) + 4× `canary-roce-pre-bundle2` +
`.env.tp4.pre-bundle2`.

Remaining open: #42159 Engram eager-extend A/B (needs prefill_sp hash re-gate — the
only 600k-prefill-axis backport left); sglang pin migration unchanged DEFERRED.

## #42159 opening CLOSED (2026-10-18, py-spy evidence)

Zero-cost check before any A/B: traffic-gated cold-600k with py-spy (100 Hz, speedscope)
on the head scheduler_TP0 thread; capture covers the full prefill (the profiler itself
slowed it ~5x — shares are unaffected). Result over 142.3 s / 14,234 samples:

- ANY engram frame in stack: 0.9 s = **0.62%** of thread wall; leaf self-time 0.25%.
- sglang#42159's direct target (`_launch_hash_kernel`, engram_hash.py): **0.02 s = 0.014%**.
- Remaining engram time is mostly OUR adapters (engram_prefetch 0.09 s,
  _engram_apply_gate 0.01 s), which the upstream patch does not touch.

Perfect elimination of all Engram eager-extend prep buys <1% wall — an order of magnitude
below the >=2% adoption bar; the 10-12 suspicion that prefill_sp/engram_prefetch already
restructured this path is confirmed. **Decision: do not backport #42159.**

Recorded side observation (no action): host-side prefill is dominated by CUTLASS-DSL
argument marshaling (~10%: _adapt_and_get_c_pointers 8.1 s + get_rectified_args 5.9 s),
torch empty_cache 4%, deep_gemm fp8_fp4_mqa_logits 2.9%, triton driver launch 2.7% —
all pin-level levers, to fold into the eventual pin-migration evaluation.

Ops note: py-spy record -d 185 did NOT self-terminate (ran ~17 min, ~5x prefill slowdown,
production shared the window) — always `kill -INT` the in-container py-spy pid after the
window and verify cadence recovery; the SIGINT flushes the report cleanly.
