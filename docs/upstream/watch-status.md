# Upstream watch status

Dated entries are appended by the Orca automation "DSV41 upstream watch" (id `646fe624-5933-4130-b8d7-4d49f604d181`, cron `17 */4 * * *` Asia/Shanghai, runs in this worktree, agent **omp**, enabled). Script: `scripts/verify/upstream_watch.sh` (local execution only).

Provider note: the same automation run under the `claude` agent blocks on Claude Code's interactive bypass-permissions consent dialog (run 1, 2026-11-09), so `omp` is an intentional choice; switching back to claude needs that consent accepted once in the Orca terminal.

Keyword list widened 2026-11-09 after run 1 self-critique (adds `deepseek-v4`, `dsv-v4`, `dgx`, `spark`, `sm121`). Alert counts are triggers, not censuses — triage stays manual.

## 2026-11-09T05:58Z — watch run (first recorded entry)

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0; raised `/tmp/dsv41-upstream-alert`
(`issue #49173` has 2 comments; `3 DSV4-relevant commits`). Log entries 05:51:37Z–05:51:40Z.
`gh auth status` → **token for `aotsukiqx` invalid** (logged out), so the tranche below was
read via unauthenticated GitHub REST (read-only, public API). `gh issue view` / `gh pr view`
were not usable; equivalent REST payloads were inspected.

### Issues

- **#49173** — open. 2 comments, last 2026-11-08T15:48:22Z (no newer activity). Kylinny took the
  report ("I'd like to take this on", 15:21:48Z), then posted that the fix is up as **#49182**.
  Both comments were already reflected in `docs/PROJECT_STATE.md`; nothing new to absorb beyond
  the PR status.
- **#49174** — open, **0 comments**, untouched since 2026-11-08T14:34:50Z. No upstream engagement on
  the post-hoc `finished_len` pool desync.

### Fix PR #49182 (referenced from #49173)

- 2026-11-09T05:58Z: **open, not merged** (`mergedAt`/`merged_at` = null; `mergeable=true`,
  `mergeable_state=unstable`).
- headRefOid **`a49ff4d7d3f7add55279e0af3c5208d630843413`** (`fix-49173-dsv41-sm120`).
- **New since the 01:20Z record**: BBuf **APPROVED** at 04:44:27Z ("Thanks for fixing this, and
  sorry for the regression. In #48399 I narrowed the dense gates together with the sparse one,
  but only the sparse table path is SM100-only.") and posted `/tag-and-rerun-ci` at 04:44:29Z
  (PR issue comment 04:44:29Z). Six reviewers requested (HaiShaw, hebiao064, merrymercy, ispobock,
  Fridge003, Qiaolin-Yu).
- Scope confirmed in diff: `is_sm100_or_newer()` = `current_arch().is_sm_at_least(10)` restores the
  **dense** DeepGEMM prefill/decode gates in `srt/layers/attention/dsv4/v41_indexer/__init__.py`
  (plus `full_topk.py` one-liner); candidate/sparse-table gate stays strict `is_sm(10)`; adds a
  CPU-runnable gate unit test. Same shape as our bp6 analysis.
- **Status: merge pending CI re-run** (approval present). Once merged → record merge commit as the
  next pin-migration candidate; no window is implied by this automation.

### sglang main commits since 2026-11-08T14:00Z

44 commits at 05:58Z (41 at the watcher run 05:51Z; new arrivals: `e5d78e65`, `3e8d7c9e`,
`d18e36cb` + control/CI). DSV4-adjacent ones:

| Commit | PR | Subject | Class |
|---|---|---|---|
| `480dcf4ec852` | #46062 | `[Perf] [NVIDIA] deepseek-v4.1: Keep one zeroed MegaAttention packed-query buffer instead of zeroing it per layer` | **bp-class candidate** |
| `e4bf2a2084d1` | #46003 | `[Perf] dsv-v4.1: Score only candidate positions in indexer consumers` | **pin-migration-class** |
| `656ac7b1d422` | #47406 | `[Fix] deepseek-v4.1: Fix bounded replay with CP and DSpark` | **NO-OP** (CP-gated) |
| `e24833d3e8d6` | #49171 | `[Fix] rope/env: Remove the leftover Dual Chunk Attention RoPE, make env alias warnings visible, restore the DGX Spark b12x env` | **NO-OP** |
| `e5d78e65a913` | #49247 | `[Fix] [NVIDIA] dsv4: Import get_eager_max_batch_size from dp_attention in the mega backend` | **NO-OP** (main-only repair) |
| `3e8d7c9e093b` | #46214 | `[Fix] flashinfer: Apply the sliding window in ragged prefill for hybrid-SWA models` | **NO-OP now**, recheck on pin |
| `4668f37139dd` | #49088 | `[Perf] [NVIDIA] minimax-sparse: Run the decode indexer CP on Hopper at TP8 with attention DP2` | not DSV4 (non-DSV4 `indexer` keyword hit) |

Notes / evidence:

- **#46062** (`deepseek_v4_mega_backend.py`, `deepseek_v4.py`, per-layer packed-query zeroing
  removed; decode-path perf). Isolated, no new deps, file exists at pin `8817f63c`, and the pin's
  `srt/utils/common.py` still exports `get_eager_max_batch_size` (`:4225`) — i.e. the import in
  #46062 resolves on the pinned tree; do **not** take #49247's import move when backporting.
  bp-class, parked; needs decode A/B in an authorised window.
- **#49247** is the follow-up repair to #46062 (+1/-1 import line) made necessary only by #48781's
  `common.py` split. Main-only; irrelevant on the pin. Merged 05:53:44Z.
- **#46003** adds `kernels/ops/attention/dsv4/candidate_scores.py` and rewrites `scoring.py`,
  `dense_blocks.py`, `candidate_blocks.py`. Collides with local bp4 #47684 in `scoring.py`; needs
  re-derive plus numerical validation, so pin-migration-class (perf-only, not window-worthy).
- **#47406** wires `postprocess_cp_output` only inside `eager_runner._execute_extend_cp`, and its new
  `deepseek_v4_hook` validation rejects prefill CUDA graphs under prefill CP; we run TP4 with no
  attention CP. It also changes `_postprocess_output` tail selection on post-pin anchors — absorb at
  the next pin, no bp action.
- **#49171** is DCA RoPE removal (`dual_chunk_attention_config` unused on our config), a
  `DeprecationWarning`→`FutureWarning` visibility change, and cookbook recipe entries for
  `SGLANG_SM120_FLASHMLA_BACKEND=b12x` / `B12X_MLA_SM120_DSV4_H16_NATIVE=1`. Our shim
  (`runtime/flash_mla_sm120.py`) still defines its own backend selection (`triton|torch` override);
  no env semantics changed for this line.
- **#46214** is a real ragged-prefill window correctness fix, but for `--attention-backend
  flashinfer`; our SM120 attention runs through b12x/FlashMLA. Recheck only if a pin migration
  flips the attention backend.

**Watcher heuristic undercounts** (observed, worth an operator decision): the 3-commit alert matched
`indexer` in a minimax PR (#49088, false positive) and missed `#46062` / `#49171`, whose subjects say
"deepseek-v4.1" / "DGX Spark" rather than "dsv4"/"dspark". The count is a trigger, not a census;
triage stays manual. Suggest adding `deepseek-v4`, `dsv-v4`, `dgx`, `sm12` to the keyword list at
the next script touch (docs-only automation keeps commits to this file).

### Action items

1. **No maintenance window triggered.** Nothing to do on the fleet; do not schedule anything from
   this automation.
2. When **#49182** merges, record its merge commit as a next-pin candidate (SM120 gate fix; the
   decode/verify-sync question it does not address stays open).
3. **#46062** is the first decode-path bp-class candidate since bp5 — pair only inside a
   user-authorised window (with the planned verify-sync work); no schedule assigned here.
4. **#49174** still unacknowledged upstream (0 comments) — continue watching.
5. `gh` login remains invalid for the automation identity; either re-auth or keep using
   unauthenticated REST (60 req/h limit, enough for this poll cadence).


## 2026-11-09T12:20Z — watch run #3 (PR #49182 MERGED → bp6 gate chain collapses)

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0 (log 12:17:43Z–12:17:47Z), alert raised
(`#49173` closed, `8` keyword-hit commits). `gh auth status` → **still invalid token for
`aotsukiqx`** (logged out); all reads below via unauthenticated GitHub REST (41/60 req left at run
end). `gh issue view` / `gh pr view` unusable again — REST equivalents used.

### Issues

- **#49173 — CLOSED** (2026-11-09T07:26:52Z, two seconds after the fix PR squash-merged). Comment
  count still 2, i.e. **no comment arrived between the 05:58Z run and closure**; the close is the
  merge event itself. Our filed SM120 DeepGEMM indexer gate bug is therefore resolved upstream.
- **#49174** — open, **0 comments**, untouched since 2026-11-08T14:34:50Z. Still no upstream
  engagement on the post-hoc `finished_len` / loop-abort pool desync.

### PR #49182 — MERGED (the delta that matters)

- **Merged (squash) 2026-11-09T07:26:50Z** as **`5e6995dd0953189b538bc35152954414335f4970`**
  (single parent `c394281b4f66`, base `main`).
- **Head merged verbatim**: `headRefOid` still `a49ff4d7d3f7add55279e0af3c5208d630843413` — the
  approved head, no re-push after approval.
- Files (verified): `srt/layers/attention/dsv4/v41_indexer/__init__.py` +19/−4,
  `.../v41_indexer/full_topk.py` +1/−1, new CPU test
  `test/registered/unit/layers/test_dsv41_indexer_arch_gates.py` +184.
- Content (verified from the patch): adds `is_sm100_or_newer()` =
  `current_arch().is_sm_at_least(10)`; **both** `_use_deep_gemm_prefill()` and
  `use_deep_gemm_decode` switch to it; `make_candidate_indexer()` deliberately stays strict
  `is_sm100()` with an explanatory comment (sparse-table kernels assert `arch == 10`).
  `_torch_prefill`'s per-layer `Tensor.tolist()` is **untouched** — Kylinny's second half of the
  plan (proposed in #49173) did not land, so any prefill cost we still measure there is ours to
  patch.

**Conflict with our current bp6v3 — read before the next pin migrate:** our
`runtime/patch_indexer_gate_48399.py` widens the **prefill gate only** and intentionally leaves
decode strict, because bp6 v2 measured SM120 DeepGEMM decode at **36.7 tok/s, below the ≥42 gate**
(~17% regression vs bp5's 0.2.3/bf16 route; 0.2.6's arch-12 contract forces fp32 weights). Upstream
restores **decode too**. First pin after `5e6995dd` therefore re-lands the regression we withheld on
purpose:

- prefill gate hunks → drop from bp6 (now upstream), no behavior change;
- decode → must stay on the torch path until an authorised window measures SM120 DeepGEMM decode
  under 0.2.6 (paired qeval, dual gates cold ≥2500 / decode ≥42). The cheap shape is to shrink the
  bp chain to a single one-line `use_deep_gemm_decode` override.

### sglang main since 2026-11-08T14:00Z

71 commits total (+27 vs the 44 recorded at 05:58Z); **27 new since the previous run**, 8
keyword-hits on the widened census (re-checked against all 27 subjects by hand, so no commit is
missed by keyword drift this run):

| Commit | PR | Subject | Class |
|---|---|---|---|
| `5e6995dd0953` | #49182 | `[Fix] [NVIDIA] dsv4: Restore SM120/SM121 to the DeepGEMM indexer gates` | **pin-migration-class** (absorbs bp6 prefill gate; see conflict above) |
| `4fc912142ffc` | #47658 | `[Perf] SM120: honour SGLANG_OPT_DEEPGEMM_HC_PRENORM for the DeepSeek-V4.1 compensated mhc projection` | **pin-migration-class** (not bp) |
| `a1ae63265866` | #47493 | `[Fix] dp-attention: Trim MLP-sync pad rows from extend outputs before the draft extend` | **NO-OP** (pure-TP line, `dp_size=1`, no attention DP) |
| `48da6bd19044` | #27267 | `[Feat] mem_cache: hugepage preflight + allocation modes for host pools` | **pin-migration-class** (mem_cache axis) |
| `6ea644d6cec1` | #48011 | `[Refactor] mem_cache: layout planner + request demand for the unified chain` | **pin-migration-class** (major) |
| `2cbb66f563db` | #48012 | `[Refactor] mem_cache: plan/admit/reclaim unified memory through one layout search` | **pin-migration-class** (major) |
| `c89cedec2671` | #48013 | `[Refactor] mem_cache: per-window-sub-pool host pool, PD entry and metrics` | **pin-migration-class** (major) |
| `388d03b7759a` | #48014 | `[Feat] attention: serve several window sub-pools per runner in the Triton backend` | **pin-migration-class** (major) |
| `7455745a5a8d` | #48010 | `[Refactor] mem_cache: one lifecycle state + one reset for unified request rows` | **pin-migration-class** |
| `72edb269c2d5` | #48009 | `[Refactor] disaggregation: per-row unified PD transfers, page reuse once quiescent` | **pin-migration-class** |

NO-OP for this line (read, no DSV4 surface): `c69fb82a` #48871 NVFP4-MoE/Marlin, `e60f7e90`
#43675 Kimi-K3 graph reload, `a4b88011` #48656 comfyui, `5325cfabd7` #45727 hisparse MLA swap,
`ff762d62` #48942 jit FP32 activations, `2f42565f` #49260 docs/cookbook, `f01950a8` #49053 AMD CI,
`397254e9` #46577 inkling graph padding, `4b2137e5` #46663 gemma4, `c81230bf` #46586 abort on client
disconnect, `6ec54519` #48405 AMD fp8 KV, `50dcb7fc` #46050 diffusion docs, `c394281b` #48662 CLI
cookbook export, `7b917c0e` #48812 weight_checker device, `e8017bc6` #48364 CPU Intern-S2,
`d910efa0` #48583 XPU kernel, `fb359066` #49218 disaggregation staging unregister.

Notes:

- **#47658** is arch-gate broadening of the same family as #49182: adds `get_platform().is_sm120` to
  the compensated mhc route in `models/deepseek_v4.py` and to `models/deepseek_v4_mhc.py`'s
  `_mix_stats_impl`, routing SM120 to the pure-Triton `split_bf16_hc_weight` (the tf32 branch is
  less accurate and rebuilds parts per row count). Gated by `SGLANG_OPT_DEEPGEMM_HC_PRENORM`, which
  this line does not set → **no bp action**; it only matters if an hc-prenorm trial is wanted.
  Anchor caveat: needs `get_platform().is_sm120` on the target tree (we cannot confirm it at pin
  `8817f63c` without blobs) — check at apply time, don't assume.
- **mem_cache "window sub-pool" train (#48009–#48014, #27267)** is a coherent multi-PR rewrite of
  host-pool planning/admission plus Triton window sub-pools, landing within one hour. Our line is a
  single-window DSV41 TP4 layout with hand-tuned pool (~6.47M) and `window_guard` floor; expect
  real re-derive work and conflict churn at the next pin. Not bp-able chunk-by-chunk.

### Action items

1. **No maintenance window triggered; nothing to do for the fleet.** This automation does not
   schedule or perform deploys — a window requires explicit human authorisation.
2. **Next-pin candidate recorded: `5e6995dd`** (#49182 merge). At that pin: drop bp6v3's prefill-gate
   hunks, keep decode on torch (override) or measure DeepGEMM decode in-window against the ≥42
   gate. bp6 patch chain shrinks to that single decision.
3. The pin's **dspark verify host-sync decode gap (~44 ms/step)** stays **open**; #49182 does not
   address it (verify path untouched).
4. **#49174** remains unacknowledged — keep watching; the underlying loop-abort `finished_len`
   desync only matters for actions aborting/resubmitting long gens, i.e. any bd headstriking on
   this line without pool reasoning.
5. Doc debt (not fixed here, docs-only automation): this watch entry supersedes the "Status: ready
   to file" header in `docs/upstream/indexer-arch-gate-48399-bug.md` (filed as #49173, fix merged).
6. `gh` auth still invalid for the automation identity (3rd consecutive run). Unauthenticated REST
   works at the 4-hour poll cadence (~14 req/run of 60/h).
