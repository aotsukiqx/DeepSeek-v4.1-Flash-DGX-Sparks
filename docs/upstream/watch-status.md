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

## 2026-11-09T16:22Z (2026-11-10 00:22 +08) — watch run #4

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0, no stdout; raised `/tmp/dsv41-upstream-alert`
with 3 lines (`#49173` 2 comments + closed; `10 DSV4-relevant commits`). Log entries
`16:17:34Z–16:17:45Z`. `gh auth status` → **token for `aotsukiqx` still invalid** (4th consecutive
run): `gh issue view` / `gh pr view` return HTTP 401, so issues/PRs/commits were read with
unauthenticated REST — payloads are equivalent for the fields used here.

Fetch note: the `commits?per_page=100&since=…` body over plain HTTP/2 was truncated mid-JSON
(`Unterminated string… char 336736`) through this proxy path. Re-fetched with `--http1.1
--compressed` (and small `per_page` when paging the delta). No data lost; that is why the commit
census below was regenerated from the full list rather than from the watcher log alone.

### Issues / PR — no new activity

- **#49173** — **closed** 2026-11-09T07:26:52Z. Resolution is the merge of **#49182**; comment count
  still 2, last maintainer comment 2026-11-08T15:48:22Z — i.e. **no maintainer comment newer than
  what the 05:58Z entry already absorbed**, and nothing new to read this run.
- **#49174** — open, **0 comments**, untouched since 2026-11-08T14:34:50Z (now >24 h, still
  unacknowledged). No upstream engagement on the post-hoc `finished_len` pool desync.
- **#49182** — unchanged since the previous entry: `state=closed`, `merged=true`,
  `merged_at=2026-11-09T07:26:50Z`, `merged_by=kpham-sgl`, merge commit
  **`5e6995dd0953189b538bc35152954414335f4970`**, `headRefOid`
  `a49ff4d7d3f7add55279e0af3c5208d630843413` (the approved head). **Next-pin candidate stands as
  recorded**; this run adds no candidate of this class.

### sglang main since 2026-11-08T14:00Z

**76 commits** at 16:20Z (+5 vs the 71 recorded at 12:20Z). Two of the five are DSV4-axis; they are
also the reason the watcher's census moved 8 → 10 keyword hits:

| Commit | PR | Subject | Class |
|---|---|---|---|
| `704681229f1c` | #47627 | `[Perf] [NVIDIA] dspark: Stage DSv4.1 long-context target verify metadata beside the draft` | **NO-OP at bp5** (conditional pin-migration; see notes) |
| `86ed6d9c63` | #47581 | `[Perf] [NPU] dspark: Integrate argmax and verify scheduler kernels` | **NO-OP** (NPU dispatch) |
| `60d8d1f8e425` | #48786 | `[Feat] HiCache: add tma copy kernel for unified layout` | **NO-OP** (HiCache not enabled) |
| `ab84df61dd40` | #46168 | `[Perf] mooncake: Cut the host time of direct linker layer-wise loads` | **NO-OP** (no mooncake/linker tier) |
| `5c5fe25edffd` | #49318 | `[Test] npu/glm5.3: Add 2P2D 32P PD-separation perf case` | **NO-OP** (NPU CI + Ascend yaml) |

Notes / evidence:

- **#47627** is on the dspark verify area but **not applicable to this line and not a candidate for
  the decode gap**. It adds `candidate_filtered` to `DSV41_EARLY_VERIFY_VARIANTS` (both DSpark verify
  variants then stage target metadata on a side stream beside the draft graph instead of rebuilding
  it in-graph for contexts >16K tokens). Dependencies are hard: `SGLANG_OPT_DSV41_TARGET_METADATA_EARLY`
  (#45450) **and** the NVIDIA megakernel plugin
  (`SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1`). This checkout has **zero references** to either
  (`grep -r 'NV_MEGA\|mega_runtime\|SGLANG_OPT_DSV41_TARGET_METADATA_EARLY'` → no hits; no vendored
  sglang tree at reset, so no apply check was attempted — per policy that is optional and not needed
  given the config gate). Touched files: `environ.py`, `layers/attention/graph_variants.py`,
  `runner/decode_cuda_graph_runner.py`, +17-line unit test.
  Scale check against our open item: upstream measures **−20.8 µs/step p50 at bs=1 on GB300**
  (verify head kernels 56 → 19.5 µs), versus our attributed **~44 ms/step** host-sync gap — three
  orders of magnitude apart, different mechanism. It therefore neither explains nor closes the
  verify-sync gap; if anything it bounds the generic in-graph metadata build at microseconds and so
  **reinforces** that our millisecond-scale gap is the host-sync, not the metadata build.
- **#47581** landed 2026-11-09T12:19:04Z — **one minute before the previous entry's snapshot**
  (12:20Z), which is why it was missed rather than misclassified. +19 lines in
  `kernels/ops/sampling/__init__.py` and `kernels/ops/speculative/dspark/dspark_schedule.py`:
  NPU kernel dispatch for dspark argmax + verify scheduling. Ascend-only; nothing on the GB10 CUDA
  path. Worth noting only as signal: upstream is actively fusing dspark verify scheduling on two
  backends in parallel, so future edits to `dspark_schedule.py` should be expected at the next pin.
- **#48786 / #46168**: HiCache TMA page-copy kernel (`kernels/ops/kvcache/`) and mooncake
  direct-linker host-time cuts (`mem_cache/storage/mooncake_store/`, `hybrid_cache/linker_pool_assembler.py`).
  This checkout is a single-node TP4 layout with no hierarchical host tier and no mooncake:
  `grep -rn 'mooncake\|hicache\|hi_cache\|enable-hierarchical'` → **no hits**. NO-OP on the pin. Both
  sit adjacent to the previously recorded mem_cache window-sub-pool train (#48009–#48014, #27267),
  so they will ride along in the same pin-migration re-derive.
- **#49318** touches only `.github/workflows/nightly-test-npu.yml`, Ascend e2e jinja/aisbench
  harness and one NPU test file. No product code.
- Watcher census: count is now literally consistent with a fresh full-list scan (10 hits / 76
  commits) including both dspark commits. The residual defect I flagged earlier is still there: the
  query window is a **fixed** `BASE_CUTOFF`, so the alert line stays "10 DSV4-relevant commits" on
  every later run regardless of delta — a quiet run reads as a hit forever. Proposal (operator
  decision, not applied): persist the last-seen head sha (committed stamp, not `/tmp`, which is
  wiped by upgrades) and count only commits newer than it, or change the alert wording to
  "cumulative". Until then treat the count as cumulative and confirm the delta by subject scan.

### Action items

1. **No maintenance window triggered; nothing to do for the fleet.** This automation does not
   schedule or perform deploys, and did not contact the cluster.
2. Next-pin candidate for migration #2 remains **`5e6995dd`** (#49182). No new candidate this run:
   #47627 is config-gated off, the rest are NO-OP. bp6 prefill hunks still collapse at that pin;
   decode decision (torch override vs DeepGEMM decode A/B in-window) unchanged.
3. **Decode gap stays open** (dspark verify host-sync, diffuse, ~44 ms/step). #47627 isolates a
   microsecond-class metadata build cost and does not touch the host-sync path — no reason to
   re-open or re-order the dspark verify rework on account of this run.
4. **#49174** unacknowledged >24 h. Holding action: anything unsafely aborting/resubmitting long
   generations on this line (bd headstriking / `FINISH_ABORT` loops) still risks the negative-residue
   pool desync; keep pool reasoning on such an experiment.
5. Watch `dspark_schedule.py` / verify-variant edits at the next pin (#47581 and #47627 both move
   inside that area); pin-migration conflict churn is now higher than the last entry assumed.
6. `gh` auth dead for the automation identity (**4th** consecutive run) — re-auth `aotsukiqx` (or
   another account) at the next opportunity, otherwise REST-only access continues. Budget this run:
   22/60 core requests (extra fetch/retry due to the truncation noted above).

## 2026-11-09T20:35Z (2026-11-10 04:35 +08) — watch run #5

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0, no stdout; alert file raised with the same
three cumulative trigger lines (#49173 "2 comments" + "CLOSED"; "11 DSV4-relevant commits"). Log
entries `20:17:41Z–20:17:49Z`. `gh auth status` → **token for `aotsukiqx` still invalid** (**5th**
consecutive run): `gh issue view` / `gh pr view` unusable, unauthenticated REST used throughout.
Also read: `sgl-project/DeepGEMM` tags/raw files (for the wheel-bump diff) and PyPI metadata for
`sgl-deep-gemm` (availability only, no install). Budget: 21/60 core requests.

### Issues / PRs — no new maintainer activity

- **#49173** — closed 2026-11-09T07:26:52Z by BBuf; **2 comments**, last still
  2026-11-08T15:48:22Z. Resolution remains the #49182 merge; nothing new to read.
- **#49174** — open, **0 comments**, untouched since 2026-11-08T14:34:50Z (>30 h, no maintainer
  response). **But it now has a referencing fix PR: #49306** (see below) — the comment-count
  trigger cannot see PR bodies, so "0 comments" here no longer means "no activity".
- **#49182** — unchanged since run #4: `merged=true`, `merged_at=2026-11-09T07:26:50Z`,
  `merged_by=BBuf`, merge commit **`5e6995dd`**, `head` still **`a49ff4d7`** = the approved head
  → **merged verbatim, no post-approval content change**. Next-pin candidate record stands.

### PR #49306 — community fix for #49174 (open, unmerged)

- Title: `[Fix] scheduler: Prevent pool accounting desync on finished_len trim`; author
  `devtyagi3909` (not a maintainer); created **2026-11-09T12:56:03Z**, last updated 12:56:58Z;
  head **`4e154403`**, base `main`, `draft=false`, `mergeable_state=unknown`.
- Body says `closes #49174` and matches our filed mechanism exactly: post-hoc `finished_len`
  trim shrinks `owned_kv_len` below `cache_protected_len`, `_release_overallocated_kv_indices`
  then frees tree-protected prefix pages, invariant checker reports negative residue.
- Diff is small and on our core surface: `managers/schedule_batch.py` **+37/−4**
  (`Req.finish_with_trim` API + floor `owned_kv_len` at `cache_protected_len` + contract doc) and
  `scheduler_components/invariant_checker.py` **+8/−2** (signed residue in message), plus unit
  tests. `bp-class` in shape, but it is **a fix for an invariant violation we did not trigger on
  bp5** — no reason to backport it as a runtime patch.
- CI in the PR body is **red on every job** (Base/Extra/AMD ROCm 10 `:x`). Not merged, not a
  pin candidate. Watch only.

### sglang main since 2026-11-08T14:00Z

**84 commits** at 20:34Z (+8 vs the 76 recorded at 16:20Z). All 8 read by subject/file scope:

| Commit | PR | Subject | Class |
|---|---|---|---|
| `dfaca5681ddc` | #46934 | `[Fix] hicache: Restore sub-window SWA L3 hits under a mid-tree anchor` | **NO-OP at bp5** (HiCache off; mem_cache pin-cluster rider) |
| `4373137b6f80` | #49020 | `[CI] diffusion: Add H100 computable runner baselines…` | NO-OP (CI baseline JSON) |
| `6a79760533e7` | #49227 | `[CI] [CPU] cpu-graph: Import STATE_PART from decl…` | NO-OP (test import) |
| `0f989aa64022` | #49135 | `[Perf] kda: shorten the MTP verify phase-1 critical path…` | **NO-OP** (Kimi K3 KDA kernel; census false positive on "verify") |
| `a478cd21e070` | #48984 | `[Perf] mhc: one-pass DeepGEMM prenorm GEMM; revert the sm107 decode MegaMoE SM reserve` | **pin-migration-class — DeepGEMM 0.2.6→0.2.7** |
| `d959849938eb` | #49205 | `[Feat] [AMD] custom-all-reduce: DPX hierarchical 2shot_push on MI450 (gfx1250)` | NO-OP (AMD arch, carv2 off) |
| `67ff3b3938b2` | #48418 | `[Fix] diffusion: Count reclaimable page cache as available inside a cgroup cap` | NO-OP (multimodal_gen runtime, not srt) |
| `6d143e51b2a4` | #47750 | `[Fix] score: Validate logical item and MIS hidden-state row counts` | NO-OP (scoring mixin; not served here) |

Notes / evidence:

- **#48984 is the event of the run**: it bumps the tree's official kernel dependency from
  `sgl-deep-gemm==0.2.6` to **`0.2.7`** in `python/pyproject.toml` **and** `docker/Dockerfile.cu134`
  (`DEEPGEMM_REF=v0.2.7`; wheels build from `sgl-project/DeepGEMM`). Code side: +16 lines adding
  `tf32_hc_prenorm_gemm_dual` to `srt/layers/deep_gemm_wrapper/entrypoint.py`, used by
  `kernels/ops/layernorm/mhc.py` only under **`current_arch().is_sm(10)`** — SM12x keeps the old
  two-call `tf32_hc_prenorm_gemm` path. The `mega_moe.py` hunk removes the SM107 decode-reserved-SM
  special case; our MoE is `b12x_next` (`SGLANG_OPT_DEEPGEMM_MEGA_GATE` default False), so **MegaMoE
  remains a NO-OP branch**. Consistency check on our side: `adapter/hc_fused.py` patches
  `hc_mix_stats_sinkhorn` (a different symbol), no local code imports the dual API.
- **Direct answer to the bp6 round-2 arm "(ii) try deep-gemm 0.2.7 (same assert?)" — YES, same
  assert.** Read at both tags: `csrc/apis/attention.hpp` line **50** (v0.2.6) and line **49**
  (v0.2.7) both carry
  `weights.scalar_type() == (arch_major == 10 ? kBFloat16 : kFloat32)` and both set
  `out_dtype = arch_major == 10 ? kBFloat16 : kFloat32`. So 0.2.7 does **not** lift the arch-12
  fp32-weights contract that put bp6v2 decode at 36.7. A version bump alone cannot recover the
  winning DeepGEMM+bf16 combo; the remaining arms are the wheel-local assert patch or an in-window
  decode measurement.
- Migration viability of the bump is otherwise fine: PyPI publishes **`sgl_deep_gemm-0.2.7-py3-none-manylinux2014_aarch64.whl`**
  (5.1 MB), so the arm64 image row can move as it did for 0.2.6. Still a pin-pre-flight item:
  wheel row + in-image import/API audit (the wrapper now exports one more function; `import
  deep_gemm` remains satisfied by the wheel, no source vendored here, so no apply check was done).
- **#46934** touches `srt/mem_cache/` (`buffer_mode/pipeline.py` +26, `unified_cache/components/swa.py`,
  `unified_radix_cache.py`, rust radix tree). It is HiCache staged-prefetch / storage-attachment
  plumbing (`prepare_staged_prefetch`, `init_hicache`), and HiCache/host-tier is disabled on this
  single-node layout — NO-OP at bp5. It partially reverts
  `_storage_attachment.resolve_prefetch_threshold` integration from the earlier storage train, i.e.
  it rides the recorded mem_cache window-sub-pool pin cluster rather than fighting it.
- **Census quality, stated plainly**: of the 8 new commits only 2 are on-surface (#48984, #46934)
  and **both missed the keyword filter** (`mhc`/`DeepGEMM`, `hicache`); the only keyword hit
  (#49135) was a false positive on "verify". Cumulative count is now 11/84. Fixed-cutoff wording
  defect persists — alert text stays "11 commits" on an otherwise quiet run.

### Action items

1. **No maintenance window triggered; nothing done to the fleet.** This run was GitHub + local
   checkout only (no SSH, no docker, no schedule, no deploy). Destructive actions: none.
2. **Next-pin candidate remains `5e6995dd` (#49182…)**, merged from the approved head — but the
   pre-flight for any pin at ≥ `a478cd21e070` now **must also move `sgl-deep-gemm` 0.2.6→0.2.7**
   in the image row (aarch64 wheel exists). Gates themselves are unchanged by #48984.
3. **Decode plan update**: 0.2.7 provably does not relax the arch-12 fp32-weights assert, so the
   "winning combo" remains blocked. Do **not** schedule a window on a version-bump hypothesis; if
   decode measurement is resumed, go straight to the wheel-local assert patch or the r3 bisect
   variant plan already prepared.
4. **#49306** (community fix, closes #49174, CI red) is open and unmerged. Monitor; if a maintainer
   adopts it, absorb at that pin. Holding action unchanged: anything unsafely aborting/resubmitting
   long generations on this line (bd headstriking / `FINISH_ABORT` loops) is still exposed to area-1
   pool desync.
5. Adjacent **open, unmerged** PRs surfaced while verifying references — no preset action, recheck
   only: **#43009** (`deepseek-v4`: rebuild evicted SWA window on cache hit), **#49319** (SM12x FA4
   split-KV outside the arch decode policy), **#45264** (mhc `BLOCK_K=32` hc_mix on SM120),
   **#41898** (dsa: reject unsupported k-pool backends before weight load).
6. `gh` auth dead for the automation identity (**5th** run); re-auth `aotsukiqx` at the next
   operator opportunity. REST-only is viable but the #49306 discovery shows body-level references
   are easy to miss with comment-count triggers.

## 2026-11-10T04:17Z (2026-11-10 12:17 +08) — watch run #6

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0; raised `/tmp/dsv41-upstream-alert`
(`#49173 has 2 comments`, `#49173 CLOSED`, `15 DSV4-relevant commits`). Log lines 04:17:28Z–04:17:31Z.
`gh auth status` → **token for `aotsukiqx` invalid (6th consecutive run)**; every read below is
unauthenticated GitHub REST (`gh issue view` / `gh pr view` unusable by design, not attempted).
Core budget 56/60 at start, 28/60 after triage (28 requests). Alert consumed (`rm`) after acting.

### Issues — no new upstream comments this run

- **#49173** — **closed** `2026-11-09T07:26:52Z`, `state_reason=completed`. Still **2 comments**,
  both dated 2026-11-08 (15:21:48Z intake + 15:48:22Z "fix is up in #49182"), already recorded in
  runs #3/#4. No new content; the alert re-raises because the state/count triggers are sticky.
  Referencing objects with the tool were exactly two: #49173 itself and the merged PR **#49182**
  (merge commit `5e6995dd`, recorded in run #3 as the next-pin candidate).
- **#49174** — open, **0 comments**, `updated_at` still `2026-11-08T14:34:50Z` → **third
  consecutive silent run**. Reference search returns exactly 2 objects (#49174 + PR #49306).
  **#49306** unchanged since last run: head `4e1544039e3f84ba534a58613787c6f620756d5e`,
  `updated_at 2026-11-09T12:56:58Z`, 1 commit, 4 files, **+116/−7** of which the source hunks are
  the previously recorded `schedule_batch.py` +37/−4 and `invariant_checker.py` +8/−2; unmerged,
  CI red. Nothing to absorb; watch only.

### sglang main since 2026-11-08T14:00Z

**120 commits** (complete census), **+36 vs the 84 recorded at 20:34Z** (84 + 36 = 120 ✓). Only
2 of these are actionable; all 36 read by subject + file scope:

| Commit | PR | Subject | Class |
|---|---|---|---|
| `b5a7900a` | #49411 | `[Feat] kernels: Kimi-K3.1 kernel changes split out of #48414` | **pin-migration-class** (see below) |
| `0e7d3ab1` | #49362 | `[Chore] deps: Bump sgl-flash-mla to 1.0.4` | **pin-migration-class** (wheel row) |
| `c4b6338f` | #37472 | `[Fix] dspark: Keep verify bookkeeping in range for rows without candidates` | **bp-class** (decode-gap adjacent, see below) |
| `790cc463` | #49262 | `[Perf] [NVIDIA] dsv41: Write the MegaMoE input from the fused decode mHC post` | NO-OP at bp5 (MegaMoE off) |
| `83b3e2e9` | #49388 | `[Fix] [NVIDIA] mega_moe: Accept MXFP4 activations at DSV4.1-Flash's 2304 intermediate size` | NO-OP at bp5 (MegaMoE off) |
| `66ad8925` | #48860 | `[Fix] cuda_graph: Retain capture inputs for full graph replay` | bp-shape, NO-OP now (runner-split backend) |
| `8d5411e4` | #48613 | `[Perf] kda: Run the fused CuTe DSpARK verify kernel on a BF16 SSM state` | NO-OP (KDA/K3; census FP on "dspark") |
| `d76112d3` | #48605 | `[Perf] moe/paged-experts: decide block-wide and skip empty page-ins` | NO-OP (paged-experts backend) |
| `4a88c741` | #40962 | `[Fix] speculative: build target graphs outside draft contexts` | NO-OP at bp5 (adaptive controller path) |
| `34f49c9d` | #49253 | `[Fix] scheduler: Retract every KV holder on pause and order a few releases behind in-flight steps` | NO-OP (pause path; adjacent to #49174 family) |
| `eafc4914` | #44176 | `[Fix] dsa: Route mixed-chunk batches to the RAGGED top-k transform for bf16-only sparse prefill` | NO-OP (`dsa_backend`, not our attention backend) |
| `fe30d1e0` | #42790 | `[Feat] mamba: Allow skipping per-draft intermediate SSM state` | NO-OP (no SSM) |
| `33d0465c` | #46213 | `[Perf] flashinfer-megamoe: Add an optional decode workspace` | NO-OP (flashinfer-megamoe off) |
| `4e9e7e82` | #49386 | `[Fix] scheduler: Record every PP result completion event after the local forward` | NO-OP (no PP) |
| `ea712f56` | #48556 | `[Fix] moe: Fix illegal memory access in the W4A16/W8A16 Triton MoE kernel (unmasked K tail)` | NO-OP (quant path unused) |
| `56ee8ed5` | #47943 | `[Fix] ZeroMQ: make blocking send wait for queue space` | NO-OP (transport not on this line) |
| `5604b31b` | #49000 | `[Fix] [NPU] dsv4: Use explicit compressor state on Ascend 950` | NO-OP (NPU; census FP on "dsv4") |
| `211b22c6` | #48616 | `[Fix] gemm: Order the CuTe DSL kernels' TMEM release after the epilogue's tcgen05.ld` | NO-OP (CuTe DSL / datacenter Blackwell) |
| `5f7f971d` | #48126 | `[Feat] [NPU] attention: Support Modelslim GQA mxfp8 kvcache for 950 NPU` | NO-OP (NPU) |
| `7b150558` | #47529 | `[Fix] RL: Fix crash when returning routed experts under DCP or KV sharding` | NO-OP (RL trainer) |
| `78e6e9fc` | #49399 | `[Fix] ernie55: Import FlashMLA from the sgl-flash-mla wheel` | NO-OP (ernie model) |
| `e85fac45` | #45859 | `[Feat] rust/grpc: Report orchestrator launch facts in api.v1 discovery` | NO-OP (Rust control plane) |
| `e886acfa` | #47620 | `[Fix] hicache: vote the prefetch timeout once per tick, not per admission poll` | NO-OP (HiCache off) |
| `bcd6a814` | #49398 | `[CI] whl-release: Prefix sgl-project/whl release tags with the package name` | NO-OP (CI) |
| `60867f43` | #49394 | `[CI] chore: add CI permissions for aaronlinear` | NO-OP (CI) |
| `9fe1dbb8` | #49350 | `[Fix] test: Drop the removed SGLANG_HICACHE_TMA_TRANSFER from the page-unified HiCache TMA test` | NO-OP (test) |
| `98d32a4f` | #49235 | `[Test] comfyui: Attach loaded plugin modules to their parents so mock.patch works on Python 3.10` | NO-OP (test) |
| `340f70fb` | #49346 | `[Test] gemma4: Cover the multimem.ld_reduce rounding in the AR fusion residual check` | NO-OP (test) |
| `5175c315` | #48956 | `[Chore] CODEOWNERS: Update score, decisions, and embeddings API owners` | NO-OP (meta) |
| `1be97a0a` | #46052 | `[Doc] cookbook: Fix the dead MiniMax-H3 "Learn more" anchors` | NO-OP (doc) |
| `e1cd0c55` | #49401 | `[Doc] cookbook: Move Rubin hardware row above AMD` | NO-OP (doc) |
| `e0187268` | #45599 | `[Perf] [AMD] batch-invariant: Segmented split-K bf16 matmul for gfx950` | NO-OP (AMD) |
| `4a5c6b8f` | #47585 | `[Fix] [AMD] quark: Widen the Quark MXFP4 W4A16 MoE gate to consumer RDNA …` | NO-OP (AMD) |
| `f25bb8a1` | #33059 | `[Fix] [AMD] dsa: Align Aiter paged rows with DP-attention metadata` | NO-OP (AMD) |
| `d766f193` | #49225 | `[Fix] diffusion: Use K/V gather when auto SP cannot split the DiT's heads` | NO-OP (multimodal_gen) |
| `5379f2cb` | #49224 | `[Fix] diffusion: Let FLUX.3 Image start with --num-gpus alone` | NO-OP (multimodal_gen) |

Notes / evidence:

- **`c4b6338f` #37472 is the only bp-class kernel fix, and it is *not* the host-sync fix the
  decode gap needs.** It clamps `ell = verify_lens - 1` to `>= 0` in `cap_correct_len` (torch +
  Triton variants) and replaces "copy every draft, then scatter the bonus" with a `k < correct`
  where-select in `build_out_tokens` / `_static_accept_tail_kernel`, plus zero-initialised
  chain-verify buffers in `dflash_utils._get_or_create_chain_verify_buffers`. All of it is index
  hygiene for **padded graph rows** (`verify_lens == 0`); no new device sync is added or removed,
  so the ~44 ms/step `cudaEventSynchronize` attribution (#48831-33 rework, r2 trace) stands. Value:
  it is a real correctness anchor on the exact files our r3 V1 revert (#48833) will collapse — the
  two touch `dspark_accept.py` / `dspark_verify_window.py`, so **prep must re-check that revert
  patch still applies after this lands**. Do not add it to bp5 standalone: wrong direction
  (correctness, not throughput) and it depends on the upstream rework we did not adopt.
- **`b5a7900a` #49411 is a 2,460-line transducer on our kernel surface** — the Kimi-K3.1 halves of
  #48414 split out, but it carries anchors we actually ride: `kernels/jit/csrc/deepseek_v4/topk_v2.cuh`
  (+200: `page_bits` interleave + `global_position()` + coarse-bin cluster path, new
  `topk_transform_paged_candidates_v2` **entry point only** — no default-path switch observed),
  new `kernels/ops/attention/mla_window_indices.py`, new `_prepare_dflash_mamba_commit_kernel` in
  `kernels/ops/speculative/dflash.py` (added, existing accept/bonus code untouched in the shown
  hunks), and additive `out=` overloads for `concat_mla_absorb_q` / `concat_mla_absorb_q_general`.
  Classed pin-migration because of size + JIT-cuh surface, not because a behavioural delta was
  proven. Not cherry-pickable as a bp; absorbed at the next pin.
- **`0e7d3ab1` #49362** moves `python/pyproject.toml` only; PyPI has
  `sgl_flash_mla-1.0.4-cp310-abi3-manylinux2014_aarch64.whl` (and x86_64), so the image row can
  move. It does not interact with the decode gap: bp6r2 already showed fmla 1.0.0 recovered only
  +2.1 of the ~7 gap.
- **`66ad8925` #48860** (full graph replay pins each shape's capture inputs) is a genuine leak-class
  bug fix, but the file lives in upstream's newer `srt/model_executor/runner_backend/` split; the
  bp5 pin predates that layout (no `runner_backend` reference anywhere in this checkout). Treat as
  "exists only at pin-migration ≥ that refactor; verify path at apply time", i.e. NO-OP now.
- The three "dsv41/MegaMoE" hits are all on paths disabled on this line: **#49262** only triggers
  when `next.mega_moe_input is not None` (`deepseek_v4_mhc.py`, `_post_fusion`) and **#49388** only
  changes a MegaMoE shape pre-dispatch check; our MoE is `b12x_next` with
  `SGLANG_OPT_DEEPGEMM_MEGA_GATE` off. Recorded anyway: if a MegaMoE arm is ever opened,
  DSV4.1-Flash's 2304 intermediate *is* now MXFP4-legal (`mxf4xmxf4` added to the align-128 set).
- **Census quality**: 5/36 keyword hits, of which 3 are false or gated (KDA DSpARK, NPU `dsv4`
  compressor, MegaMoE shape check) and 2 real (#37472, #49262). **Four material commits missed the
  filter entirely**: #49411 (pin-class), #49362 (pin-class), #48860, #49253 (plus #40962 conditional).
  Cumulative 15/120. Fixed-cutoff wording defect unchanged — the alert's 15 is the whole window's
  hit count, not this run's.

### Action items

1. **No maintenance window triggered; nothing done to the fleet.** GitHub REST + this checkout only
   (no SSH, no docker, no build, no serve, no schedule change). Destructive actions: none besides
   consuming the alert file.
2. **Next-pin pre-flight now carries four rows** (candidate remains `5e6995dd`): (a) DeepGEMM
   0.2.6→0.2.7 (#48984, recorded run #5 — arch-12 fp32 assert unchanged, no decode unlock);
   (b) **sgl-flash-mla 1.0.3→1.0.4** (#49362, aarch64 wheel present); (c) **#49411** land on the
   pin and re-run adapter anchors (`b12x_next` untouched, but inspect `deepseek_v4/topk*`, dflash,
   `mla_qk_prep`); (d) confirm whether the pin carries `runner_backend/full_cuda_graph_backend.py`
   before evaluating #48860. Also pre-flight bookkeeping: verification of the post-49182 shape —
   `#49262` (mHC post) and `#49388` (shape check) change files (`deepseek_v4_mhc.py`, `mega_moe.py`)
   that a pin will pick up and that gate checks must not trip.
3. **Decode gate discipline unchanged**: bp6 pool+prefill are done, decode gap still attributed to
   the dspark verify rework's host-sync pattern. #37472 is index hygiene, not sync removal; no
   window for it. If r3 resumes, re-base the qeval A/B plan on the post-#37472 tree.
4. **#49306** (#49174 fix, CI red) still open, unmerged, body-referenced only — no maintainer
   comment on #49174. Holding exposure of #49174 unchanged (aborted/resubmitted long generations
   on this line).
5. `gh` auth still dead for the automation identity (**6th run**). REST-only remains viable;
   operator re-auth `aotsukiqx` is the standing fix.
6. Adjacent open/unmerged PRs unchanged, no preset action: #49306, #43009, #49319, #45264, #41898.

## 2026-11-10T08:27Z (2026-11-10 16:27 +08) — watch run #7

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0 (twice: unpatched at 08:17Z, patched at 08:27Z);
raised `/tmp/dsv41-upstream-alert` with three lines (sticky `#49173` count + CLOSED + commit count).
Log lines 08:17:52Z–08:17:55Z and 08:27:42Z–08:27:52Z. `gh auth status` → **invalid token for
`aotsukiqx` (7th consecutive run)**; everything read via unauthenticated GitHub REST (`gh issue view`/
`gh pr view` not attempted). Main HEAD at capture `b2616cabbf` (`2026-11-10T08:25:48Z`). Alert
consumed (`rm`) after acting. **Only in-repo change this run besides this file: the watcher census.**

### Watcher census — fixed (count), semantics intentionally unchanged

The DSV4 window since `2026-11-08T14:00Z` now holds **131 commits**, but the script fetched a single
`per_page=100` page. Full pagination shows **18** keyword hits, not the 14 the page-limited query
reports; the four omissions are all in the `11-08T14:00Z–11-09T02:5x` stretch already triaged in runs
#1–#2, so nothing material was missed — the number was simply wrong, and this alone explains part of
the run-to-run wobble (15 → 14) earlier entries credited only to the fixed cutoff.

`scripts/verify/upstream_watch.sh` now pages until exhausted (bounded at 10 pages) and logs
`dsv41-relevant-since-watch=18 window-commits=131`; the alert body quotes both. Trigger semantics are
deliberately untouched: the count stays **window-cumulative**, so any hit since 11-08 keeps the alert
sticky. A per-run delta would require persisting a head SHA; not done here (the prior entries' "leave
script fixes to the next script touch" note now applies to that item).

### Issues — no new upstream comment or state change

- **#49173** — closed `2026-11-09T07:26:52Z`, `state_reason=completed`; still exactly **2 comments**,
  both dated 2026-11-08, unchanged since run #1. Alert lines are sticky on the state/count, by design
  of the script.
- **#49182** (its fix PR) — unchanged: `merged=true`, `merged_at 2026-11-09T07:26:50Z`, merge commit
  **`5e6995dd0953189b538bc35152954414335f4970`**, already recorded in run #3 as the next-pin
  candidate. Nothing re-read.
- **#49174** — open, **0 comments**, `updated_at` still `2026-11-08T14:34:50Z` → **fourth consecutive
  silent run**. **#49306** (community fix) unchanged: open, unmerged, head
  `4e1544039e3f84ba534a58613787c6f620756d5e`, `updated_at 2026-11-09T12:56:58Z`.

### sglang main — 10 commits between run #6 and capture, +1 during write-up

| Commit | PR | Subject | Class |
|---|---|---|---|
| `73bf84e90f` | #49436 | `[Fix] dspark: Write deferred draft KV only for carried requests` | **pin-migration-class** (see below) |
| `29aaf75899` | #49420 | `[Fix] [NVIDIA] jit: Build sm_107a on CC 10.7, keeping sm_100f only for modules that use CUTLASS` | NO-OP here; pin bookkeeping (JIT arch API) |
| `ae1347a5c3` | #45263 | `[Perf] [NVIDIA] fp8: Add Triton block-fp8 GEMM configs for RTX PRO 5000` | **NO-OP** (device-keyed to RTX PRO 5000) |
| `b2616cabbf` | #44047 | `[Fix] openai: Validate sampling bounds and preserve chat output limits` | NO-OP for DSV4; **API-contract tightening** |
| `fa703d685a` | #46944 | `[CI] [AMD] minimax-m3.1: MI35x MXFP4 + DSpark nightly accuracy gate` | NO-OP (AMD CI; bare keyword hit) |
| `0c44fbff44` | #44566 | `[Perf] [AMD] Qwen3-Next: consume the fused AR+norm+quant output` | NO-OP (AMD) |
| `d565013acd` | #47857 | `[Feat] [NPU] attention: enable mixed-chunk for pure MLA on the ascend backend` | NO-OP (NPU) |
| `6dc7e3ea7e` | #44152 | `[Feat] score/embedding: raw embeddings, score rids, ... schedule-decision logging` | NO-OP |
| `c678ef8dc4` | #49470 | `[Fix] CODEOWNERS: Restore prefill-only test ownership` | NO-OP |
| `e03f9eef09` | #49461 | `[Fix] rust-server: Pass architectures to ModelConfig in test_rust_server_crash_dump` | NO-OP |
| `2fb4f3db43` | #49453 | `[Doc] [AMD] cookbook: Match the Kimi-K3.1-Pro MI35x cells ...` | NO-OP |

Notes / evidence:

- **#49436 is the only material commit, and it is not the decode-gap fix.** Structure (12 files,
  squash `73bf84e90f`, Liangsheng Yin):
  `dspark_components/dspark_deferred_commit.py` → **`dspark_deferred_draft_kv.py`**,
  `DeferredContextCommit` → **`DeferredDraftKVStore`**; removes the worker's
  `_maybe_build_deferred_commit` (`dspark_worker_v2.py` +25/−94), `note_static_verify`,
  `after_draft_forward` and the scalar host `staged_bs`; adds `note_staged(reqs)` which stores
  `(req, retraction_count)` rows and a `before_draft_forward(graph_bs, reqs)` that writes **only rows
  of requests this draft carries, with unchanged retraction_count**, leaving the staging pending
  across an intervening prefill. Env names are untouched
  (`SGLANG_DSPARK_DEFER_CONTEXT_KV`, `SGLANG_OPT_DSV41_DSPARK_ONE_LAUNCH_COMMIT_STAGE`, gated on
  `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION`); only their comments changed.
  Full-diff scan: **zero production `cudaEventSynchronize` / `.item()` / `.cpu()` deltas** — the new
  work is Python tuples/sets over the batch → the ~44 ms/step verify host-sync attribution stands.
- **#49436 collision surface**: the bp6r3 V1–V3 reverts target `dspark_verify_window.py`,
  `ragged_verify_kernels.py`, `decode_cuda_graph_runner.py`, `ragged_verify.py` (+ new test) — **no
  file overlap**, so those patches do not need re-basing for it. What dies is any future dspark patch
  anchored on `DeferredContextCommit` / `note_static_verify`. Cost: two O(bs) Python loops per step at
  the pin; noise vs a 44 ms sync, but measure if the decode gate ever nears target.
  Lineage: `docs/pin-migration-plan-20261024.md` already listed `dspark_deferred_commit` in the
  reorganized dspark_components, so the machinery predates this window and must be assumed present
  on the bp5 pin until the in-image tree is inspected.
- **#49420** gates #45911's `sm_100f` fallback per module: `arch.MODULE_USES_CUTLASS =
  ["deep_select_topk"]`, `get_compile_arch(kernel_name)`, module name threaded through `load_jit` and
  the ninja `-gencode`, build key includes the target. CC 10.7-only (Vera Rubin) → **no SM120
  behaviour**; recorded because it changes the JIT compile-arch surface and build-key hashing a pin
  would absorb. Payload useful later: the DSV4 `dpsk_v4_topk_v2` cluster kernels are 49–56 KB static
  smem and cannot build under `sm_100f` with nvcc 13.4 (they are `sm_90a/100a/103a/107a`); if a pin
  build fails inside `topk_v2`, suspect the arch target before the source.
- **#45263** is 8 JSONs under `kernels/ops/gemm/configs/`, all keyed
  `device_name=NVIDIA_RTX_PRO_5000_72GB_Blackwell`, `block_shape=[32,32]`. Our device name is GB10 →
  the lookup never matches → NO-OP, and upstream ships **no GB10-keyed Triton fp8 configs**. Keep
  anyway as the enumerated DSV4.1-Flash dense shape list (`1792×5120, 25600×6144, 32768×1280,
  4096×1280, 5120×1024, 5120×288, 5120×8192, 576×5120`) and as the tuning method if the Triton w8a8
  path is ever exercised here.
- **#44047** validates sampling bounds in `protocol.py` / `sampling_params.py` / `min_new_tokens.py`.
  Nothing DSV4, but it **rejects out-of-bounds sampling parameters that used to pass through**, and
  restores chat output-limit enforcement. At the next pin a client sending e.g. `temperature > 2` or
  an invalid top-p/top-k pair gets a 400 instead of silent clamping; check our eval/probe scripts'
  payloads before a window.
- The one hit the keyword set does **not** see is `#45263` (subject says only "RTX PRO 5000"); it was
  found by reading the subject list, not by the counter. This is the known census weakness in its
  least harmful form — a device-keyed config PR, not a code change for this line.

**Census quality**: 18/131 by the script's keyword set (was 14/100 before the fix); a widened local
scan (adds `topk|deferred|moe|deepep` stems) finds 24/131. Of the 18, 3 are AMD/NPU/CI noise
(#46944, #44566, #47857-plus-#49453 class). This run is otherwise complete: every DSV4-material commit
matched on its subject (the "materials missed the filter" failure mode of run #6 did not repeat).

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + this
   checkout only (no SSH, no docker, no build, no serve, no schedule change). Destructive actions:
   none besides consuming the alert file.
2. **Next-pin pre-flight rows now six** (candidate remains **`5e6995dd`**): (a) DeepGEMM 0.2.6→0.2.7;
   (b) sgl-flash-mla 1.0.3→1.0.4; (c) #49411 kernel-split/dflash additions; (d) confirm whether the
   pin has `runner_backend/full_cuda_graph_backend.py` before evaluating #48860; (e) **#49436** —
   audit the pin tree for `DeferredContextCommit`/`note_static_verify` references outside `srt/`, then
   validate the deferred-KV contract under prefill↔decode interleaving (a prefill now leaves a staging
   pending, and retracted rows are skipped); (f) **#49420** — JIT compile-arch API/build-key change;
   verify only if that pin postdates #45911.
3. **Decode gate unchanged**: bp6 pool+prefill done, decode gap still the dspark verify host-sync
   pattern. This window contains no sync removal; the r3 bisect plan stays as written and any resumed
   run must be re-based on a tree that includes #49436's renames if the pin moves first.
4. **#49306** is still open/unmerged (CI red, body-referenced only); #49174 remains silent. Exposure to
   aborted/resubmitted long generations on this line is unchanged.
5. Adjacent **open, unmerged** PRs noticed this run — watch only, no preset action: **#49499**
   (`dsv4: Size expand_prefill_causally's token block by batch size`), **#47685** (`dflash: Top up
   decode KV pages for the whole batch on allocation steps`), **#47478** (`dsv41-engram: Share Engram
   tables across P/D using EGM`). Previous list unchanged otherwise (#43009, #49319, #45264, #41898).
6. `gh` auth dead for the automation identity (**7th run**); operator re-auth of `aotsukiqx` remains
   the standing fix. REST-only stays viable for issues/PRs, but #49306 showed body-level references
   are invisible to the comment-count trigger.
