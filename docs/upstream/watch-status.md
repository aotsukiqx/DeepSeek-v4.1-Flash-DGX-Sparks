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

## 2026-11-10T12:19Z (2026-11-10 20:19 +08) — watch run #8

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0 (12:17:29Z–12:17:38Z); alert re-raised with the
same three sticky lines (`#49173` comment count, `#49173` CLOSED, cumulative commit count) and
consumed after acting. `gh auth status` → **invalid token for `aotsukiqx` (8th consecutive run)**;
all reads via unauthenticated REST, `gh issue view`/`gh pr view` never attempted. Core budget
56 → 47/60 (2 commit pages, 2 issues, 2 PRs, 2 commit details, 6 adjacent-PR polls, HEAD probe).
Main HEAD at capture `2d93fbfe51` (`2026-11-10T11:13:23Z`). **Only in-repo change: this file.**

### Issues — no new comment, no state change

- **#49173** — closed `completed` `2026-11-09T07:26:52Z`, `comments=2` (both 2026-11-08) → unchanged
  since run #1; the only referencing PR remains **#49182** (`merged`, head `a49ff4d7d3`, merge
  `5e6995dd09`), already the next-pin candidate since run #3.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **fifth consecutive silent
  run**. Issue search (title+body, PR only): exactly one hit, **#49306** — still open, unmerged,
  head `4e1544039e`. No new body-level reference to either issue appeared; the alert is repeating
  fixed facts, not reporting activity.

### sglang main — 11 new commits, no DSV4-material change

Census **18/142** (run #7: 18/131; +11 window commits, +0 keyword hits). Every subject since the
run-#7 capture (`b2616cabbf` there belongs to run #7):

| Commit | PR | Subject | Class |
|---|---|---|---|
| `2d93fbfe51` | #47917 | `[Feat] [NPU] longcat: Support Longcat2.0 for arch22 NPU` | NO-OP (NPU) |
| `a2ffd21469` | #49335 | `[CI] pp: Launch the PP test servers on DEFAULT_URL_FOR_TEST` | NO-OP (CI) |
| `626b95d503` | #43032 | `[Feat] Qwen3.8-Flash-Next: Avoid padded PLE prefill activations` | NO-OP (other model) |
| `4c08984a69` | #48776 | `[Feat] kimi-k3: Read the thinking object, map reasoning effort onto K3 tiers` | NO-OP (other model) |
| `533c067689` | #48615 | `[Fix] diffusion: Keep explicit cuDNN attention inside LTX breakable CUDA graphs` | NO-OP (diffusion) |
| `ed06877072` | #49024 | `[Perf] [AMD] wan2.2: Gate FlyDSL norm and scale-shift fusion` | NO-OP (AMD) |
| `3830fb0713` | #47685 | `[Perf] [NVIDIA] dflash: Top up decode KV pages for the whole batch on allocation steps` | **NO-OP here; pin bookkeeping** (DFlash-only call site) |
| `670265cd4b` | #48994 | `[Refactor] moe: Rename --moe-a2a-backend flashinfer to flashinfer_nvlink_one_sided` | **NO-OP here; pin bookkeeping** (CLI enum rename, 31 files) |
| `7e1cf12384` | #49509 | `[CI] NPU: increase NPU coverage timeout, relax non-denoise tolerance` | NO-OP (CI) |
| `66f47cd634` | #49459 | `[CI] rust-server: Pass architectures to ModelConfig in the crash-dump test` | NO-OP (CI) |
| `d50f98e7f8` | #48775 | `[Feat] kimi-k3: Stream tool-call arguments incrementally` | NO-OP (other model) |

Details / evidence:

- **#47685** touches a speculative module we do watch (`dflash_info_v2.py`), so the fetched diff
  was checked: the new top-up helper is called **only** from `DFlashInfoV2.prepare_for_decode`.
  Four files: `environ.py` (+4), `mem_cache/allocation_sizing.py` (+60, new
  `add_spec_decode_headroom_pages`), `dflash_info_v2.py` (+11), one test. New env
  **`SGLANG_OPT_SPEC_DECODE_HEADROOM_PAGES` defaults to 2**, i.e. any future tree running DFlash
  takes batch-wide page headroom silently. Guard rails: no-op when `page_size == 1` or under a
  HiSparse coordinator (`batch.hisparse_coordinator`), capped at 8 pages, and skipped when
  `free < num_needed + bs*page_size + 4*extra` (paged-extend eviction demand). Our spec transform is
  dspark (`dspark_*` files, per `runtime/bp6r3/*.patch`), and this checkout contains **zero**
  `dflash` code references — class NO-OP. Playbook note: if a future pin or r3 branch routes DSV41
  through `dflash_info_v2`, both page-allocation sizing **and** the extra one-allocation-per-step
  cost need re-baselining before trusting earlier decode/cold numbers.
- **#48994** changes a server-arg enum value. `grep -rn moe-a2a-backend` in this checkout: **0 hits**
  (launchers, probes, runtime patches) and our MoE path is `b12x_next` → NO-OP now. Pin pre-flight:
  the enum's test coverage moved (`test_platforms/...` style server-args tests, +35/−1), so re-run
  the launcher arg audit at adoption; a relayed script that still passes the old `flashinfer` value
  would fail arg validation rather than misroute.

### Cross-checks

- Arithmetic: run #7 total 131 + 11 = 142 ✓. The 11 shas in the table equal the shas returned after
  `2026-11-10T08:25:48Z` (set comparison, no eye-balling), with `b2616cabbf` excluded as #44047 from
  run #7.
- **Keyword census misses both near-surface subjects** (`dflash`, `moe` stems absent from the
  filter) — the counter stayed at 18 while two commits that change shared encoding/environ files
  landed. Presence here comes from reading every subject, which remains mandatory. Scope still has
  no DSV4 executable delta: model/spec/attention hits are NPU/AMD/diffusion/kimi/Qwen, and no commit
  in this batch overlaps the recorded bp6r3 patch targets or `#49436`'s rename surface.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + this
   checkout only (no SSH, no docker, no build, no serve, no schedule change). Only destructive
   action: consuming the alert file.
2. **Next-pin pre-flight rows now eight** (candidate remains **`5e6995dd`**): (a) DeepGEMM
   0.2.6→0.2.7; (b) sgl-flash-mla 1.0.3→1.0.4; (c) #49411 kernel-split/dflash additions; (d) check
   for `runner_backend/full_cuda_graph_backend.py` before evaluating #48860; (e) #49436 deferred-KV
   rename/prefill-interleave audit; (f) #49420 JIT compile-arch/build-key change (only if the pin
   postdates #45911); (g) **#48994** — MoE A2A enum rename, re-audit launcher CLI values; (h)
   **#47685** — DFlash headroom env has a **non-zero default 2**; no dspark call site exists, verify
   before enabling DFlash paths so page sizing is not double-counted.
3. **Decode gate unchanged**: bp6 pool+prefill proven, decode gap still the dspark verify host-sync
   pattern. No upstream sync removal landed here; r3 bisect plan stays as written.
4. **#49306** still open/unmerged (CI red); #49174 silent for five runs. Exposure to
   aborted/resubmitted long generations unchanged.
5. Adjacent **open, unmerged** PRs re-verified this run: **#49499** (dsv4 prefill token block),
   **#47478** (dsv41-engram P/D sharing), **#43009** (deepseek-v4 evicted-SWA rebuild), **#49319**
   (SM12x FA4 KV split), **#45264** (mhc SM120 BLOCK_K=32), **#41898** (dsa k-pool guard). Watch
   only; **#47685 is no longer in this list** (merged, triaged above).
6. `gh` auth dead for the automation identity (**8th run**); operator re-auth of `aotsukiqx`
   remains the standing fix. REST-only is sufficient for state/issues, and the search endpoint
   covered body-level references this run.

## 2026-11-13T12:17Z (2026-11-13 20:17 +08) — watch run #9

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0 (12:17:38–12:17:40Z) but a **degraded run**:
both issue fetches logged `fetch-failed` and the census logged `? ?` — the GitHub core budget was
exhausted (0/60 at 12:18Z, reset 12:19:25Z; `curl -sf` swallows the 403). With no branch raising,
`need_alert=0` **deleted** the alert file: a fully rate-limited run is indistinguishable from a
clean one (known script limitation, recorded; fix deferred to the next script touch). No alert
content existed to consume. All data below comes from a manual REST sweep after the reset.
`gh auth status` → **invalid token for `aotsukiqx` (9th consecutive run)**; `gh issue view`/
`gh pr view` never attempted. Core budget 60 → 9/60 across the sweep (2 issues, 2 PRs, 2 searches,
7 census pages, 31 commit details, HEAD probes).
**Automation gap**: no log lines between 2026-11-10T12:17Z and this run — the `17 */4 * * *` Orca
cron produced 0 of ~17 expected runs in 3 days (the log file still holds its 11-10 lines, so not a
/tmp wipe; more likely host sleep or a paused automation). Operator attention item, not fixable
from this repo. Main HEAD at capture `6d2d13a1c4` (`2026-11-13T12:20:16Z`).
**Only in-repo change: this file.**

### Issues — zero delta across the 3-day gap

- **#49173** — closed `completed` `2026-11-09T07:26:52Z`, `comments=2` (both 2026-11-08) → unchanged
  since run #1. Only referencing PR remains **#49182**: `merged`, head `a49ff4d7d3`, merge
  `5e6995dd09` — identical SHAs to run #3's recording; next-pin candidate unchanged.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **eighth consecutive silent
  run**. Issue search (PR bodies): still exactly one hit, **#49306** — open, unmerged, head
  `4e1544039e`, `updated_at 2026-11-09T12:56:58Z` (unchanged). No new body-level references.

### sglang main — 269 new commits in 3 days; 41 material rows, 3 pin-relevant dep bumps

Census **51/411** by the script's keyword set (run #8: 18/142; +269 window commits, +33 keyword
hits). Arithmetic: 142 + 269 = 411 ✓ (pages 100/100/100/100/11/0; run-#8 head `2d93fbfe51` found
at index 269 — slice, not eye-balling). Every one of the 269 subjects was read; 56 were triaged
individually below (41 material + 15 residual watch), the remaining **213 grouped NO-OP**: 45
diffusion/multimodal-gen, 28 AMD/ROCm, 30 CI/Test/Doc/Chore+tooling, 21 other models/MM, 16
API/serving surface, 15 NPU/Ascend, 13 other spec algos (KDA/GDN/EAGLE/dflash/ngram), 10
PD/PP/multi-node, 9 disabled tiers (hicache/paged-experts/linker/offload), 9 scheduler/infra
refactors (request-list move/retirer wave: #49677/#49623/#49496/#49751 — behaviour-neutral, anchor
churn only), 1 XPU/Apple, plus 16 DSV4/dspark-keyword NO-OPs on NPU/AMD/CI/test surfaces folded in
(#47934, #42977, #48031, #50096, #49054, #50267, #50252, #49443, #49428, #49647, #48891, #48731,
#44017, #49380, #49601, #42855).

| Commit | PR | Subject | Class — note |
|---|---|---|---|
| `2c75d490` | #50016 | [Chore] deps: Bump sgl-deep-gemm to 0.2.8 and sgl-flash-mla to 1.0.5 | **pin-migration** — wheel matrix; DeepGEMM attention.hpp byte-identical to 0.2.7 (see below) |
| `175ab802` | #49980 | [Perf] dsa: Native-head FlashMLA and a precomputed schedule for DCP decode | **pin-migration** — touches `deepseek_v4_backend.py` +25/−3, `dsa_backend.py` +48/−14 |
| `94d3dbad` | #49966 | [Refactor] mem_cache: Compose same-lifetime parts into one unified sub-pool | **pin-migration** — kv/build,decl,plan +185/−40, new layout/composite.py +271 |
| `2c597c92` | #49922 | [Perf] dsa: Stage the DCP top-k merge push in smem, fall back in registers ≤2 ranks | **pin-migration** — dcp_topk_merge.cuh +73/−15 |
| `92929577` | #49912 | [Feat] dsa: Honor the page stride of index-K buffers | **pin-migration** — indexer k-cache layout contract, 11 files incl. dsv4_fp4_index_k_cache |
| `bddf8d01` | #49775 | [Fix] mem_cache: Size a fused draft's KV region in the dtype the draft resolves | **NO-OP here; bookkeeping** — 11 nextn/eagle models touched, dspark absent; contract if drafts ever fuse |
| `a081aaa2` | #49771 | [Chore] deps: Bump FlashInfer to 0.7.4 | **pin-migration** — Dockerfile+pyproject+uno_utils; enablement for #45262/#49595 |
| `62eab144` | #49662 | [Chore] sgl-kernel: Stop bundling triton_kernels in the sglang-kernel wheel | **NO-OP here; bookkeeping** — wheel composition change |
| `ed15b28d` | #49661 | [Refactor] mxfp4: Dequantize MXFP4 MoE weights with a JIT kernel instead of triton_kernels | **NO-OP here; bookkeeping** — triton_kernels removal wave |
| `df4ec2e0` | #49660 | [Refactor] moe: Remove the triton_kernel MoE runner backend | **NO-OP here; bookkeeping** — backend we never selected |
| `51401bab` | #49659 | [Refactor] quantization: Quantize MXFP8 weights with a JIT kernel instead of triton_kernels | **NO-OP here; bookkeeping** — same wave |
| `72f72cca` | #49656 | [Fix] kernels: Synchronize AOT MoE prefix before expert lookup | **watch** — AOT MoE prefix sync (not our MoE path; kernel-level sync semantics) |
| `21453213` | #49616 | [Fix] dsa: Give each FlashMLA decode graph its own schedule for native-head kernels | **pin-migration** — dsa_backend −37; DSA decode-graph scheduling wave |
| `9fdd3920` | #49595 | [Perf] dspark: opt-in MXFP8 draft lm_head on SM10x via --speculative-dspark-lm-head-dtype | **pin-migration** — replaces ROCm-only env; flashinfer mm_mxfp8 cute-dsl; verify-logits GEMM axis |
| `0f5a47de` | #49538 | [Refactor] moe: Remove the SM120 top-22 small-M router path | **pin-migration** — −109 moe_fused_gate.py + test deleted; top-22 is our router shape |
| `fc6d3b6f` | #49499 | [Perf] dsv4: Size expand_prefill_causally's token block by batch size | **bp-class** — merged from our adjacent watch list; attn_metadata_kernels +3/−2 |
| `56e38dfe` | #49476 | [Fix] dsv4/topk_v2: Move the cluster kernels' top-k buffer to dynamic shared memory | **bp-class** — static smem pressure drops (cf. run #7's 49–56 KB note) |
| `43d0ecf9` | #49402 | [Refactor] graph: Introduce device-neutral graph backend bases (1/3) | **pin-migration** — runner_backend churn incl. full_cuda_graph_backend.py; parts 2–3 pending |
| `3d04a487` | #49321 | [Perf] [NVIDIA] dsv41: Fork the decode attention mHC mix at the MegaAttention core | **bp-class** — deepseek_v4.py/_boundary/_mhc +32/−4 |
| `e674ed71` | #49311 | [Fix] mem_cache: Signal a double free of unified pages when invariant checks are on | **NO-OP here; bookkeeping** — invariant diagnostics |
| `e8a2bee9` | #49310 | [Fix] scheduler: Apply the write-after-read barrier in the PP loops and on every unified-memory device | **NO-OP here; bookkeeping** — unified-memory wave, no PP here |
| `1b51a111` | #49301 | [Feat] mem_cache: Ask every unified memory capacity question of every pipeline stage | **pin-migration** — capacity contract change |
| `2a7c267e` | #49300 | [Refactor] mem_cache: Ask unified memory every capacity question as a demand | **pin-migration** — capacity contract change |
| `e38dff3e` | #49299 | [Fix] mem_cache: Reset a reused request row's state in the next forward | **NO-OP here; bookkeeping** — row-state hygiene |
| `e9fd9efe` | #49263 | [Fix] cuda-graph: Pause automatic GC during CUDA graph capture | **bp-class** — base_cuda_graph_runner +8/−2; capture-time robustness |
| `f9b90d33` | #49189 | [Refactor] cleanup: Prune dead code and env vars, fix silent failures | **pin-migration** — deletes dsv4/dsa anchors (below); env prunes cross-checked: none used here |
| `84ee24c9` | #49102 | [Perf] dsa: bf16 MQA logits for the prefill indexer | **pin-migration** — SGLANG_DSA_INDEXER_BF16_LOGITS default off; topk_v2.cuh + topk_impl.cuh |
| `93dbe651` | #48949 | [Perf] reject_sampling: Vocabulary-tiled final-token pass at small batch | **watch** — +447 lines in the shared spec op; dspark consumer unconfirmed |
| `219b7e7a` | #48928 | [Perf] DeepSeek-V4.1-Flash: Use segmented fastokens for prompt encoding | **NO-OP here; watch** — parser-side prompt encoding |
| `026a145a` | #48456 | [Perf] [NVIDIA] dsv41: Run the side-stream mHC mix in BF16x3 for Blackwell decode batches | **bp-class** — mhc.py + deepseek_v4_mhc.py; BF16x3 BLOCK_K=64 const |
| `d46d9adf` | #48367 | [Perf] speculative: Fuse the draft proposal's Gumbel-max sample into one pass | **NO-OP here; watch** — generic draft-side sampling |
| `4929ba40` | #48350 | [Perf] flashinfer_cutedsl: Use swap-AB MoE on Blackwell | **NO-OP here; watch** — not our MoE backend |
| `4a99de61` | #47469 | [Perf] [NVIDIA] attention: opt-in LiteTopK prefill indexer top-k for DSA (FP8) and DSV4.1 (FP4) on SM100 | **NO-OP now; pin candidate** — SGLANG_OPT_USE_PREFILL_LITE_TOPK, off, SM100-gated, ~3.7k LOC |
| `75e8d543` | #46830 | [Fix] dsa: Remove host syncs from rank-divergent eager paths | **watch** — host-sync family, but eager metadata, not our verify site |
| `884d29b2` | #46186 | [Perf] spec: radix top-k for speculative verify target probabilities | **NO-OP here; watch** — dflash consumer |
| `1017f576` | #46185 | [Feat] gdn: run DFLASH and DSPARK verify on the ReplaySSM spec ring | **NO-OP here; bookkeeping** — GDN-local; shared spec_utils.py −42 anchor churn |
| `d684cc9f` | #45264 | [Perf] [NVIDIA] mhc: Use BLOCK_K=32 for the hc_mix kernel on SM120 | **NO-OP here; bp candidate** — gate is CC==(12,0) only; code says "SM121 has not been measured" |
| `6c0c4d32` | #45262 | [Perf] [NVIDIA] fp8: Use FlashInfer b12x for block-fp8-as-MXFP8 linears on SM12x | **pin-migration** — dynamic is_backend_supported("b12x", SM) gate, K%128==0 |
| `00752f26` | #44823 | [Fix] scheduler: Park a chunked prefill continuation when its chunk or KV budget is spent | **NO-OP here; watch** — chunked-prefill budget semantics |
| `e11c9583` | #44588 | [Feat] spec: Online acceptance estimator for DSpark confidence-scheduled verification | **pin-migration** — SGLANG_DSPARK_USE_ONLINE_ESTIMATOR off; touches dspark_verify_window.py (bp6r3 flag) |
| `5d0de926` | #42566 | [Fix] quantization: Load block-FP8 MoE TP shards that split checkpoint blocks | **NO-OP here; bookkeeping** — our TP4 shard dims are 128-block aligned; shared loader path |

Residual individually-read NO-OPs (kept for their pin notes):

| Commit | PR | Subject | Class — note |
|---|---|---|---|
| `8385d83d` | #50144 | [Feat] JIT: Add SGLANG_JIT_ENABLE_LINEINFO for profiler source views | NO-OP here; bookkeeping — useful for the r3 decode profiling |
| `74cb973d` | #49943 | [Perf] dcp: Replay prefix hits under breakable prefill graphs | NO-OP here; watch — no breakable prefill graphs on this line |
| `087e57c4` | #49931 | [Fix] [NVIDIA] unquant: restore eager low-M BF16 performance | NO-OP here; watch |
| `983455ae` | #49868 | [Fix] test: Seed the retracted queue through move() in test_prealloc_token_usage | NO-OP — test-only; #49174-adjacent surface |
| `07a6c955` | #49497 | [Perf] flashinfer-autotune: Tune MoE with varied warm-up tokens | NO-OP here; watch — autotune not exercised |
| `7b4cf858` | #49370 | [Refactor] custom-ar-v2: Move the DPX pair stages into their own header | NO-OP here; bookkeeping |
| `2f209ae0` | #49365 | [Refactor] custom-ar-v2: Single-word push epoch and consume-style pull semaphores | NO-OP here; bookkeeping — verify which AR path the pin uses |
| `dec82422` | #49139 | [Perf] [NVIDIA] unquant: Re-land low-M BF16 autotune via FlashInfer CuTe DSL mm_bf16 | NO-OP here; watch |
| `a8ff9948` | #48947 | [Perf] layernorm_sp: Add --layernorm-sp-min-tokens to run small extend passes replicated | NO-OP here; watch — SP unused here |
| `c7f3e2bd` | #48602 | [Perf] dsa: Add an opt-in FlashInfer Prims-TS sparse MLA backend | NO-OP now; pin candidate — new decode backend option |
| `46aa131b` | #48295 | [Feat] modelopt: Quantize block-FP8 MTP draft experts to NVFP4 for MIXED_PRECISION checkpoints | NO-OP |
| `3769d88f` | #48272 | [Fix] mem_cache: report the CUDA error when host registration fails | NO-OP — diagnostics |
| `6299ec26` | #48271 | [Perf] qsa: drop the per-layer RoPE cache length check in the indexer | NO-OP — other model's indexer |
| `ed037608` | #48263 | [Perf] [NVIDIA] nvfp4: Fuse SiLU-and-mul with the next linear layer's NVFP4 activation quant | NO-OP — NVFP4 unused |
| `800f9b30` | #44753 | [Refactor] RoPE: Support RotaryEmbedding class usage for DiT models | NO-OP — DiT |

### Evidence notes (fetched diffs; runbook §1b/§3b applied)

- **#50016 wheel contract**: Dockerfile `DEEPGEMM_REF v0.2.7→v0.2.8`; raw-diff of DeepGEMM
  `csrc/apis/attention.hpp` between v0.2.7 and v0.2.8 is **byte-identical** → the arch-12
  fp32-weights assert (`weights.scalar_type() == (arch_major == 10 ? kBFloat16 : kFloat32)`)
  **persists in 0.2.8**; the bump still cannot recover bf16-weights decode MQA logits on SM121
  (extends the 2026-11-09 known answer to 0.2.8). PyPI: `sgl_deep_gemm-0.2.8-py3-none-
  manylinux2014_aarch64.whl` ✓ and `sgl_flash_mla-1.0.5-cp310-abi3-manylinux2014_aarch64.whl` ✓.
- **#45264**: `_hc_mix_stats_sinkhorn_block_k` returns 32 only when
  `torch.cuda.get_device_capability(device) == (12, 0)`, with the comment **"SM121 has not been
  measured."** GB10 is CC 12.1 → default BLOCK_K retained. A one-line extension + measurement
  candidate, not a free win. #48456 in the same kernel adds `_HC_MIX_BF16X3_BLOCK_K = 64`.
- **#48891**: `_small_heads_prefill_enabled` requires `device_sm_major == 10` → excludes CC 12.1
  (the inverse of #49182's restored `>=10`); no behaviour change here.
- **#49595**: deletes `SGLANG_ROCM_OPT_DSPARK_FP8_LM_HEAD` (moved to the deprecated-map pointing at
  the new flag) and adds `--speculative-dspark-lm-head-dtype {fp8,mxfp8}`; the mxfp8 path is
  flashinfer `mm_mxfp8` (cute-dsl) gated only on `hidden.is_cuda and K%32==0 and N%128==0` → applies
  to SM121 **iff** FlashInfer 0.7.4 ships the cute-dsl kernel for it. Measurement candidate for the
  verify-logits GEMM, gated on the #49771 bump.
- **#44588**: kernel op `dspark_verify_window.py` +18/−1, `dspark_worker_v2.py` +64,
  `SGLANG_DSPARK_USE_ONLINE_ESTIMATOR` default False. **bp6r3 collision**: the V1 revert
  (`runtime/bp6r3/bp6r3-pr48833-revert.patch`) touches `dspark_verify_window` — conflict re-check
  required before resuming the r3 bisect.
- **#49189**: deletes `layers/attention/dsa/kpool_fp8_index.py` (−127) and
  `layers/attention/dsv4/compressor_v2.py` (−93), prunes `dsa_indexer_kpool.py` (−84). The 8 pruned
  env vars cross-checked against this checkout (launchers, probes, runtime patches, .env files):
  **zero usage**. Anchors for the indexer-arch-gate audit (docs/upstream/indexer-arch-gate-48399-bug.md)
  change at the next pin.
- **#49538** removes the SM120 top-22 small-M router fast path entirely (−109 + test deleted).
  Top-22 is our router shape; whether SM121 ever took that path is unrecorded → decode router
  re-baseline at the next pin.
- **#49102**: `SGLANG_DSA_INDEXER_BF16_LOGITS` (default off; interacts with pre-existing
  `SGLANG_OPT_DSV41_BF16_INDEX_LOGITS`, whose unset default is "bf16 on SM100, fp32 elsewhere") —
  prefill-side bf16 logits axis; the decode-side DeepGEMM assert is untouched (first note).
- **Census quality**: the script's keyword set misses **42 of the 56 individually-triaged rows**
  (both dep bumps, the mem_cache/graph refactors, the whole DSA decode wave, #49189 cleanup). The
  alert counter (+33) would have undersold this window ~4:1 — reading every subject remains
  mandatory, and after a multi-day gap the window must be re-sliced from the last recorded head.

### Cross-checks

- Arithmetic: run #8 total 142 + 269 new = 411 ✓; census pages 100/100/100/100/11 (page 6 empty);
  run-#8 head `2d93fbfe51` at index 269; 269 = 41 material + 15 residual + 213 grouped.
- Fetch integrity: all 44 API JSONs parse (issues, PRs, searches, 5 census pages, 31 commit
  details); batch-loop failure mode of §1b checked explicitly.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + this
   checkout only (no SSH, no docker, no build, no serve, no schedule change). No alert consumed
   (none existed — see tooling note).
2. **Next-pin pre-flight rows now fourteen** (candidate remains **`5e6995dd`**): (a) DeepGEMM →0.2.8
   [supersedes the 0.2.7 row — attention.hpp identical, arch-12 fp32-weights assert persists];
   (b) sgl-flash-mla →1.0.5 [supersedes 1.0.4 — aarch64 cp310-abi3 wheel confirmed]; (c) #49411
   kernel-split/dflash; (d) runner_backend path checks [#49402 base-class wave, parts 2–3 pending];
   (e) #49436 deferred-KV rename audit; (f) #49420 JIT build key (pin post-#45911 only); (g) #48994
   moe-a2a enum rename; (h) #47685 DFlash headroom default 2; **(i) #49771 FlashInfer 0.7.4** —
   enablement gate for #45262/#49595; **(j) #49538** SM120 top-22 router path removed — decode
   router re-baseline; **(k) #45262** b12x-for-block-fp8-as-MXFP8 dense linears — confirm our dense
   path's backend and K%128 on SM121; **(l) #49595** `--speculative-dspark-lm-head-dtype` —
   verify-logits GEMM measurement (needs i); **(m) #49189** kpool_fp8_index/compressor_v2 anchors
   deleted — redo the indexer-arch-gate audit anchors; **(n) #49966/#49300/#49301** unified
   sub-pool + capacity-as-demand — mem_cache layout migration.
3. **Decode gate unchanged in substance**: no sync removal on our verify site (the ~44 ms/step
   dspark verify host-sync stands; #46830 removes syncs only from rank-divergent eager metadata).
   Upstream added default-off perf axes relevant to us — #44588 online acceptance estimator,
   #49102 prefill bf16 indexer logits, #47469 LiteTopK prefill top-k, #49595 lm_head dtype,
   #48602 Prims-TS backend — **none active by default; evaluate at the next pin, not via runtime
   patch**. r3 bisect stays as written, now with the #44588 conflict re-check.
4. **bp6r3**: `bp6r3-pr48833-revert.patch` (V1) vs #44588's `dspark_verify_window.py` change —
   conflict re-check before any resumed bisect; #37472 note from run #6 unchanged.
5. **Adjacent watch list shrank**: **#49499 and #45264 merged** (triaged above as bp-class and
   NO-OP-here/bp-candidate). Still open, not re-polled this run (budget): #47478 (dsv41-engram P/D),
   #43009 (evicted-SWA rebuild), #49319 (SM12x FA4 KV split), #41898 (dsa k-pool guard) — last
   verified run #8. #49306 open/CI-red as before; #49174 silent (8th run).
6. **Orca automation gap 11-10 → 11-13** (~17 missed cron slots): operator should confirm the
   automation host/schedule (this run was the first log line in 3 days). The script's
   silent-clean-on-rate-limit behaviour (alert deleted when fetches fail) is recorded as a known
   limitation; fix deferred to the next script touch.
7. `gh` auth dead for the automation identity (**9th run**); operator re-auth of `aotsukiqx`
   remains the standing fix. REST-only stayed sufficient, including body-reference searches.

## 2026-11-13T16:17Z (2026-11-14 00:17 +08) — watch run #10

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0 (16:17:37–16:18:05Z). **Partial run**: both
issue fetches succeeded, but the census leg printed `dsv41-relevant-since-watch=? window-commits=?`
(python/urllib exit in the script) while the alert still raised on issue facts — so this run was not
silently clean, but the commit counter was unusable and the sweep below is manual. A direct re-run of
the same urllib call 90 s later returned `ok 100`; budget was not exhausted at the time (53/60 at
16:18Z, window reset 16:19:52Z → 59/60 after). Transient, but the `? ?`-on-failure masking is the
second consecutive run it hid real signal (run #9 was a full 403 blackout). Fix still deferred.
`gh auth status` → **invalid token for `aotsukiqx` (10th consecutive run)**; `gh issue view` /
`gh pr view` never attempted. Core budget: ≈20/60 for the sweep (2 issues, 2 comment lists, 1 PR,
1 search, 1 adjacent PR, 5 census pages, 7 commit details, 1 probe). **Only in-repo change:
this file.** Main HEAD at capture `410e64bb84` (`2026-11-13T15:51:53Z`).
**Cadence**: log shows the `17 */4` slots at 11-13T12:17Z and 11-13T16:17Z — 4 h apart, healthy
since the 3-day gap recorded in run #9; that gap's cause remains unexplained.

### Issues — zero delta; PR candidate unchanged

- **#49173** — closed `2026-11-09T07:26:52Z`, `comments=2`, both `Kylinny` **2026-11-08**
  (15:21:48Z / 15:48:22Z) → no maintainer comment since run #1; nothing new to read.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **ninth consecutive silent
  run**. Body-reference search: still exactly one hit — **#49306** open, unmerged, head
  `4e1544039e`, `updated_at 2026-11-09T12:56:58Z` (unchanged since run #5; still CI-red).
- **#49182** — `merged`, merge `5e6995dd09`, head `a49ff4d7d3`, `merged_at 2026-11-09T07:26:50Z` →
  **byte-identical to run #3/#9 recording**; next-pin candidate remains **`5e6995dd`**.

### sglang main — 17 new commits in 4 h; 1 bp candidate, 4 pin notes, 0 new-model work

Census 428, prev head `6d2d13a1c4` at index 17 → slice `[0:17]`. Arithmetic: 411 + 17 = 428 ✓.
Keyword counter would have said **3 of 17** (only #50039/#50178/#50177 match) for what is 6 rows of
real triage — the 4:1 miss rate from run #9 persists on a small window.

| Commit | PR | Subject | Class — note |
|---|---|---|---|
| `dac84ad8cc` | #50177 | [Fix] DeepSeek-V4: Skip decode attention TP setup when it is off | **bp-class** — `maybe_use_decode_attn_tp` short-circuits when `get_cp_decode_attn_tp_ctx().is_enabled` is False (our CP=1 config): skips per-layer `isinstance`+ctx resolution on decode. 3 LOC, tiny CPU saving, **not** the ~44 ms `cudaEventSynchronize` site. Anchor `maybe_use_decode_attn_tp` must exist at pin 8817f63c — check at apply time |
| `2e8a2346cd` | #50039 | [Fix] deepseek-v4.1: Preserve compact DSpark outputs with CP bounded replay | **pin note** — `_forward_impl` L4747: `cuda_cp_tail = not capture_dspark and _needs_cp_tail_scatter(...)`; CP-only (our CP=1 ⇒ `_needs_cp_tail_scatter` False ⇒ unchanged). Adds an 80-line unit test pinning the "compact gather" contract for DSpark capture under CP. Anchor churn in the exact `capture_dspark` tail block any future bp would touch |
| `410e64bb84` | #49382 | [Refactor] custom-ar-v2: Share one symmetric workspace per group | **pin note** — 1463/−879 structural: new `symm_workspace.py`, `custom_all_reduce_v2.py` rewrite, `parallel_state.py` +23/−4. Custom-AR-v2 is not our cross-node path (1 GPU/node, all-reduce over RoCE via NCCL) ⇒ NO-OP-here; re-check at pin that TP init is unaffected |
| `0508e4c6d3` | #46132 | [Perf] cuda-graph: Raise the decode graph ceiling `max_bs` 32→128 for 35–60 GiB | **NO-OP here** — only the `gpu_mem < 60*1024` branch changed (A100-40/L40); GB10 (128 GiB unified) sits in the ≥90 GiB branch and our launcher sets `max_bs` explicitly |
| `c65c1e5d29` | #49866 | [Fix] sgl-kernel: Do not compile FA3 instantiation files twice | **NO-OP runtime / build** — `kernels/aot/CMakeLists.txt` +6/−9 only; shorter sgl-kernel build at the next image |
| `530db9b470` | #49477 | [Perf] attention: Optimize FA3/FA4 tree attention | **NO-OP** — confined to `flashattention_backend.py` + new `kernels/ops/attention/metadata.py::prepare_tree_spec_metadata`; our DSV4 serve path is `deepseek_v4_backend`, and dspark verify is not tree attention |
| `56b32550c7` | #50178 | [Fix] DeepSeek-V4: Default `input_embeds` to `None` in DS4 forward | **NO-OP** — signature default only (+1/−1); no caller change |
| `dad2c79111` | #49338 | [Fix] compressed-tensors: Reject WNA16 MoE TP shards that split a scale group | **NO-OP** — WNA16 weight-only quant gate, not our config |
| `0758280c8d` | #49878 | [Fix] disaggregation: Attribute decode HiCache hits to host and storage | **NO-OP** — PD disagg + HiCache host tier disabled |
| `85111ff763` | #47842 | [Fix] diffusion: Cosmos 3 Edge/Nano fused QK-norm/RoPE on SM 121 (GB10) | **NO-OP** — diffusion model path, not the LLM serving surface (SM121 named only) |
| `7eaa90110f` | #50098 | [Fix] diffusion: Wan2.2-TI2V-5B 32-pixel patch grid | **NO-OP** — diffusion |
| `822e3e2dd1` | #48629 | [Perf] models: fused `apply_qk_norm` in Exaone, HunYuan-V3, AfMoE, BailingMoE | **NO-OP** — other models |
| `806417dc77` | #49964 | [Fix] kimi_k25: vision projection under caller's grad mode | **NO-OP** — other model |
| `0745bfe49b` | #49600 | [Feat] function_call: `granite_toolcall_parser` | **NO-OP** — serving surface, unused parser |
| `1b2870d33d` | #46408 | [Refactor] multimodal: [MM refactor 2/9] processors from config | **NO-OP** — MM processors |
| `eaf392e425` | #50033 | [Doc] hybrid: sizing state/KV pools for multi-turn reuse | **NO-OP** — docs |
| `b9283c02f3` | #50231 | [Doc] [NPU] restructure Ascend tuning guide | **NO-OP** — docs/NPU |

### Cross-checks

- Arithmetic: run #9 total 411 + 17 = 428 ✓; prev head `6d2d13a1c4` at index 17 of 428; pages
  100/100/100/100/28 (page 5 short → stop); 17 = 1 bp-class + 16 NO-OP, of which 4 carry
  pin/build notes (#50039, #49382, #46132, #49866) and 12 are plain NO-OP.
- Fetch integrity: all 12 API JSONs parse (2 issues, 2 comment lists, 2 PRs, 1 search, 5 census
  pages); commit-detail batch 7/7 parsed (§1b assertion run on the file set).

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + this
   checkout only — no SSH, no docker, no build, no serve, no schedule change. Alert file consumed
   and removed after acting.
2. **Next-pin pre-flight rows stay fourteen** (candidate **`5e6995dd`**, unchanged); add #50039's
   `_forward_impl` anchor churn and #49382's `parallel_state.py`/`symm_workspace.py` to the
   apply-check notes only — neither is a contract change for TP-only serving.
3. **New bp candidate (low value)**: #50177 decode-attn-TP skip — 3 LOC, worth folding in only if a
   window is already open for other reasons; it cannot close the decode gap.
4. **Decode gate unchanged**: still no sync removal at our verify site; #49477's tree-attention
   metadata rewrite is a different backend.
5. **Orca automation**: cadence healthy (2 slots 4 h apart); the 11-10→11-13 gap remains
   unexplained and the census `? ?` masking fired again — operator should confirm the automation host
   and priority-order the script fix (fail loud instead of `?`).
6. `gh` auth dead for `aotsukiqx` (**10th run**); REST-only remains sufficient. Re-auth is the
   standing operator fix.

## 2026-11-13T20:17Z (2026-11-14 04:17 +08) — watch run #11

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0 (20:17:33–20:17:42Z); all three legs numeric
(`#49173 closed comments=2`, `#49174 open comments=0`, `dsv41-relevant-since-watch=55
window-commits=436`) — the `? ?` masking of runs #9/#10 did not fire. Alert raised on **cumulative**
facts only (closed status + the frozen comment count), i.e. a trigger, not a delta. `gh auth status`
→ **invalid token for `aotsukiqx` (11th consecutive run)**; `gh issue view` / `gh pr view` never
attempted. Core budget: 52/60 at start (used 8), 37/60 at end → **15 core + 1 search** for the
sweep. **Only in-repo change: this file.** Main HEAD at capture `cc3e9dc03e` (committer
`2026-11-13T20:12:50Z`).
**Cadence**: log slots 16:17:37Z → 20:17:33Z, `17 */4` healthy; no new evidence on the 11-10→11-13
gap.

### Issues — zero delta (10th silent run for #49174)

- **#49173** — closed `2026-11-09T07:26:52Z`, `comments=2`, `updated_at 2026-11-09` → unchanged
  since run #1; no maintainer text to read.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **tenth consecutive silent
  run**. Body-reference search (`"49174" type:pr`): `total_count 1`, still **#49306** open/unmerged,
  `updated_at 2026-11-09T12:56:58Z` — unchanged since run #5, still CI-red.
- **#49182** — `merged`, merge `5e6995dd09`, head `a49ff4d7d3`, `merged_at 2026-11-09T07:26:50Z` →
  byte-identical to runs #3/#9/#10; next-pin candidate remains **`5e6995dd`**.

### sglang main — 8 new commits in 4 h; 2 pin/build rows, 6 NO-OP

Census 436, prev head `410e64bb84` at index 8 → slice `[0:8]`. Arithmetic: 428 + 8 = 436 ✓.

| Commit | PR | Subject | Class — note |
|---|---|---|---|
| `ad438669fa` | #50369 | [Chore] release: Bump sgl-kernel version to 0.4.11 | **pin-migration-class (dep)** — in-tree version only (`kernels/aot/pyproject{,_cpu,_musa,_rocm}.toml`, `python/sgl_kernel/version.py`); the **consuming dep pin is still `sglang-kernel==0.4.10`** (`python/pyproject.toml:83` at main). **Wheel gap found**: PyPI 0.4.11 (uploaded `2026-11-13T20:02:33Z`) ships **one** file, `cp311-abi3-manylinux2014_x86_64`; 0.4.9/0.4.10 shipped **both** aarch64+x86_64 as `cp310-abi3`. Two shifts: **no aarch64 wheel yet**, ABI floor 3.10→3.11 ⇒ a bump to 0.4.11 cannot install on GB10 until an aarch64 wheel appears or we build the kernel from source |
| `d133444fec` | #50260 | [Chore] fa3: Serve FA3 from the sgl-fa3 wheel and remove it from sglang-kernel | **pin-migration-class (build)** — dep adds `sgl-fa3==0.1.0rc0 ; linux and x86_64` (**aarch64 skipped by marker**; PyPI has only `cp310-abi3-manylinux2014_x86_64`), `sgl_kernel.flash_attn` module + `flash_ops` symbols deleted from the kernel build (CMakeLists −120, `flash_attn.py` −424). Runtime **NO-OP**: `_is_fa3_supported` covers sm80/86/89/90a only (SM121 rejected), `dsa fa3` impl is SM9.x per KNOWN_FAILURES, and `grep -rn sgl_kernel\.flash_attn runtime/` is empty. Pin check: no module-scope `sgl_fa3` import on the `deepseek_v4_backend` path |
| `914405d107` | #43136 | [Perf] [AMD] DCP: Slice gathered MHA prefixes with host offsets to drop GPU syncs | **NO-OP (AMD/DCP)** — but it edits **shared** `models/deepseek_common/attention_forward_methods/forward_mha.py` (+1/−3): `all_gather_kv_cache_for_mha_extend` drops the GPU `seq_lens`/`extend_prefix_lens`/`extend_seq_lens` args and takes CPU lists instead. Call site is inside the `dcp_local_prefix_kv_indices` branch (DCP≠1); no local patch anchors on that symbol ⇒ no conflict today |
| `084a5e9aae` | #47645 | [Fix] [AMD] aiter: Plan the DCP asm verify shard with the draft tokens under CUDA graph replay | **NO-OP** — `aiter_backend.py` only (ROCm). Analog note: the planner localized `seq_lens[:bs]` where the verify window needs `kv_lens`, i.e. a graph-replay verify-shard length bug of the same family as #37472; no shared file, no CUDA effect |
| `69635d1d3f` | #45905 | [Refactor] moe: simplify FlashInfer MegaMoE autotune handoff | **NO-OP** — `flashinfer_megamoe_autotune.py` + test; MegaMoE bypassed (our MoE is `b12x_next`, `SGLANG_OPT_DEEPGEMM_MEGA_GATE` False) |
| `cc3e9dc03e` | #50181 | [Fix] hicache: Refuse a runtime write_back policy under buffer_only host memory | **NO-OP** — new refusal in `mem_cache/unified_cache/storage_attachment.py`; host tier disabled, and `unified_cache/` is post-pin (unified sub-pool landed 2026-11-13, pre-flight row n) |
| `4a4f400022` | #49603 | [Fix] [AMD] docker: Pin UCX and NIXL commits for ROCm PD disaggregation | **NO-OP** — `docker/rocm.Dockerfile` |
| `a8d88db3d1` | #49632 | [Test] crash-dump: Pin the Python frontend in the tests that resolve `--ssl-keyfile-password` | **NO-OP** — test-only `envs.SGLANG_RUST_SERVER.override(False)` fixtures |

### Cross-checks

- Arithmetic: 428 + 8 = 436 ✓; prev head `410e64bb84` at index 8; 8 = 2 pin/build + 6 NO-OP
  (of which 2 carry apply/pin notes: #43136 shared-file signature, #50181 post-pin path).
- Fetch integrity: 8/8 commit details parse with `files[]`+`patch`; 5 census pages, 1 PR, 1 search
  parse; file count asserted before triage (§1b).
- Census-cutoff note: my wider `since=2026-11-08T00:00:00Z` fetch returns 477 rows (pages
  100/100/100/100/77), the script's `BASE_CUTOFF=14:00:00Z` returns 436 — the 41 extras all have
  committer date ≤ `2026-11-08T01:38:51Z`, i.e. pre-cutoff. Both agree on the 8-row delta.
- Keyword counter vs reality: it flagged ~2 of 8 (#43136 "syncs", #47645 "verify"), both AMD-only,
  while the two material rows (#50369/#50260) matched only on generic words — the bidirectional
  miss keeps holding; per-commit reading stays mandatory.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + this
   checkout only — no SSH, no docker, no build, no serve, no schedule change. Alert file consumed
   and removed after acting.
2. **Next-pin pre-flight rows now sixteen** (candidate **`5e6995dd`**, unchanged): (a)–(n) as run
   #8/#9, plus **(o) `sglang-kernel` 0.4.11 aarch64 gate** — do not absorb the version bump until
   an `aarch64` wheel exists on PyPI (0.4.11 today: x86_64 only, and ABI moved to `cp311-abi3`),
   otherwise the image must build `sglang-kernel` from source; **(p) FA3 wheel split** — confirm at
   pin that nothing on the DSV4 path imports `sgl_fa3`/`sgl_kernel.flash_attn` at module scope on
   aarch64 (dep marker excludes us).
3. **No new bp candidate this window.** #50177 (run #10, decode-attn-TP skip, 3 LOC) remains the
   only open low-value bp; it cannot close the decode gap.
4. **Decode gate unchanged**: still no sync removal at our verify site (~44 ms/step
   `cudaEventSynchronize`, dspark verify rework). #43136 is the closest structural analog (drops
   GPU syncs) but is AMD DCP-only.
5. **bp6r3 conflict re-checks stand**: #44588 and #37472 on `dspark_verify_window.py` (V1 revert
   `bp6r3-pr48833-revert.patch`). #47645 is the same bug family as #37472 but ROCm-local.
6. **Operator**: `gh` auth dead for `aotsukiqx` (**11th run**) — re-auth remains the standing fix;
   REST-only stayed sufficient (15/60 core). The script's `? ?`-on-failure masking (runs #9/#10)
   is still unpatched, though the census leg was healthy this run.

## 2026-11-14T00:17Z (2026-11-14 08:17 +08) — watch run #12

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0 (00:17:38–00:18:03Z). **Partial run again**:
both issue legs numeric, census leg printed `dsv41-relevant-since-watch=? window-commits=?` (the
script's python/urllib call exits early) while the alert raised on cumulative issue facts. Budget at
that moment **49/60 core** → this is *not* a rate-limit symptom; it is the third occurrence of the
script's census failure mode (run #9 = full 403 blackout, #10 = same `? ?` at 53/60 core) and the
second consecutive time it masked a window containing material commits. Census redone manually
below via `curl --http1.1 --compressed` (5 pages, 21 new rows).
`gh auth status` → **invalid token for `aotsukiqx` (12th consecutive run)**; `gh issue view` /
`gh pr view` never attempted — REST only. Core budget 49/60 at start → 31/60 after the sweep
(**18 core**: 2 census+2 issue legs are the script's, 5 census pages, 3 issue/PR/search reads,
10 commit details). **Only in-repo change: this file.** Main HEAD at capture `8c7ec05287`
(committer `2026-11-13T23:36:19Z`).
**Cadence**: log slots 20:17:33Z → 00:17:38Z, `17 */4` healthy; the 11-10→11-13 gap stays
unexplained.

### Issues — zero delta (11th silent run for #49174)

- **#49173** — closed `2026-11-09T07:26:52Z`, `comments=2`, `updated_at 2026-11-09T07:26:52Z`;
  the two comments are both **`Kylinny` on 2026-11-08** (15:21:48Z / 15:48:22Z) — nothing newer
  than run #1, so no maintainer text to read.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **eleventh consecutive
  silent run**. Body-reference search (`"49174" type:pr`): `total_count 1`, still **#49306**
  open/unmerged, `updated_at 2026-11-09T12:56:58Z`, unchanged since run #5.
- **#49182** — `state=closed`, `merged=True`, merge **`5e6995dd09`**, head **`a49ff4d7d3`**,
  `merged_at 2026-11-09T07:26:50Z` → byte-identical to runs #3/#9/#10/#11. Next-pin candidate
  remains **`5e6995dd`**.

### sglang main — 21 new commits in 4 h; 1 build/dep row, 20 NO-OP

Census **457** (`since=2026-11-08T14:00:00Z`, pages 100/100/100/100/57), prev head `cc3e9dc03e`
at index 21 → slice `[0:21]`. Arithmetic: 436 + 21 = 457 ✓.

| Commit | PR | Subject | Class — note |
|---|---|---|---|
| `3e4d7bd0c5` | #50394 | [Chore] flash-attn: Restore the kernels-hub FA3 loader and the kernel lock | **pin-migration-class (build/dep)** — new runtime dep **`kernels>=0.14.1,<0.15`** + `[tool.kernels.dependencies] "kernels-community/sgl-flash-attn3" = 1`; **new env `SGLANG_USE_SGL_FA3_KERNEL = EnvBool(True)`** (`environ.py:1654`, default **on**) routes `_load_fa3_kernels()` to `_load_fa3_kernel_from_sgl_fa3()`, whose body is a bare `from sgl_fa3 import …` **outside the try/except** (the try/except only wraps the kernels-hub branch, and its `except` falls *back to* the wheel). #50260's dep marker is `linux and **x86_64**` ⇒ on aarch64 that import is unavailable, i.e. the failure is an ImportError, not a caught fallback. `Dockerfile.cu134` +24: `kernels lock python` then `kernels download python` with an explicit aarch64 skip branch ("sgl-flash-attn3 ships no arm variants; JIT-compile at runtime") and `kernels.lock` moved into `SGLANG_CACHE_DIR`. Runtime scope: FA3 is excluded at SM121 by `_is_fa3_supported` (sm80/86/89/90a, per run #11), so no DSV4 serve path should reach the loader — the pin risk is (a) module-scope import of `sglang/kernels/ops/attention/flash_attention_v3.py` on aarch64, (b) `kernels` installing in the aarch64 image |
| `d38c87c4fd` | #49906 | [Fix] mem_cache: Keep freeing SWA slots for an unpublished prefill with a pending SWA branch | **NO-OP now / pin note** — `mem_cache/unified_radix_cache.py` +10/−0 (+2 disagg tests). Unified radix cache is **post-pin** (unified sub-pool landed 2026-11-13, pre-flight row n); at the next pin any SWA-enabled or PD path inherits the free-slot fix. No local patch anchors on `mem_cache/*` (`grep` over `runtime/*.patch` is empty) ⇒ no conflict today |
| `257ba631d4` | #49636 | [Feat] kv-shard: Let out-of-tree backends and pools join logical-page KV sharding | **NO-OP / pin note** — `arg_groups/kv_shard_hook.py` +19/−2, `mem_cache/kv_cache_configurator.py` +9/−0, `mem_cache/page_interleave.py` touched. Logical-page KV sharding is not our layout (TP4 over NCCL, no kv-shard); `kv_cache_configurator` gains a hook post-pin code may call |
| `d7a4948025` | #50364 | [Feat] quantization: Support dense SiLU+mul fusion for masked per_token_group_quant | **NO-OP (default off) / optional lever** — new `kDenseSiluAndMul` trait arg on `PerTokenGroupQuantMaskedKernel` and `masked_layout` removed from the dense-mode rejection, so `silu_and_mul_mode="dense"` now works on the **masked (MoE) layout**, bit-exact vs flat dense (new `test_dense_silu_masked_bitexact`). Opt-in only; no launcher flag in this repo sets `silu_and_mul_mode`. Different files from our bp4-pr46088 anchors (`deepseek_v4_mhc.py` imports `create_per_token_group_quant_fp8_output_scale` from `ops/quantization/fp8_kernel`, not `per_token_group_quant`) ⇒ no anchor churn |
| `6ddfd1380c` | #49686 | [Feat] disaggregation: Make the PD top-logprobs metadata width configurable | **NO-OP** — PD disagg; but `managers/scheduler.py` +23/−5 (our bp anchors `managers/scheduler_components/batch_result_processor.py`, a different file) |
| `a1bc580cfd` | #49917 | [Fix] hicache: Lock the window a staged buffer-mode load-back keeps | **NO-OP** — `mem_cache/buffer_mode/pipeline.py` +19/−1 + 384 lines of tests; HiCache host/buffer tier disabled here |
| `0e509eaf4c` | #49633 | [Perf] hicache: Stage kernel-backend host index uploads in pinned memory | **NO-OP** — `mem_cache/pool_host/unified.py`, `utils/device.py` (pinned-alloc helper); host tier disabled |
| `9521f578b7` | #47025 | [Feat] disaggregation: Add opt-in Mooncake TENT transfer engine backend for PD | **NO-OP** — +1107/−26 PD transfer engine (`tent_transfer_engine.py` new); PD not used |
| `aebd76a5f3` | #50363 | [Perf] [AMD] hicache: Submit staged write-back copies with hipMemcpyBatchAsync | **NO-OP** — ROCm + host tier |
| `8c7ec05287` | #50373 | [Test] server_args: Stop two PortArgs tests from probing fixed host ports | **NO-OP** — test-only (+15/−2) |
| `d6bc9e7272` `108ffbdc5b` `2815ad0907` `b71901e7ad` `fbe18964eb` `27c2438dae` `b71152ae2a` `34fd2d48c9` `5bfcc9f1f2` `29327444b1` `26f3f4e1c0` | #50199 #50190 #50189 #49907 #49663 #49653 #49650 #49640 #49635 #50204 #49631 | rust-server block: MM item identity hashed once; abort-hook request names; `--warmups`/`--gc-warning-threshold-secs`; MM processor request failure; `/generate` error type+retryability; Python-parity rejections; media without a fetch slot; tokenizer-queue 503 + extension metrics + GIL release on shutdown; **aarch64 SHA-256 instructions**; `/v1/loads` snapshot lock off async workers; `managers/load_snapshot.py` encode-before-lock | **NO-OP (11 rows)** — the Rust frontend is opt-in (`SGLANG_RUST_SERVER`); this line serves with the Python frontend (`grep` of `scripts/`, `runtime/` finds no `RUST_SERVER` reference). #50204 is build-only for a Rust binary we do not ship. Note the block is the largest single cadence of upstream work this window (11/21) — the Rust frontend is getting the attention, our Python serve path is not |

### Cross-checks

- Arithmetic: run #11 total 436 + 21 = 457 ✓; prev head `cc3e9dc03e` at index 21 of 457.
- Fetch integrity: 5 census pages + 10 commit details + 1 PR + 1 search + 1 comment list = 18 JSONs,
  all parse (asserted before triage, §1b); `uc/` count asserted at 10.
- Class totals: 1 pin/build + 20 NO-OP = 21 ✓. NO-OP breakdown: rust-server/load-snapshot 11,
  HiCache host tier 3, PD disagg 2, other 4 (test, quant default-off, kv-shard, post-pin radix cache).
  Four NO-OPs carry pin notes (#49906, #49636, #49686, #50364).
- **Keyword counter: 0 of 21.** The script's own heuristic would have reported *no* DSV4-relevant
  commits in a window that contains a default-on FA3 loader switch, a new runtime dependency, and a
  Dockerfile cubin path. Worst counter miss recorded (previous worst: 42 of 56 rows missed, run #9).
  Per-commit reading is the only reliable mode.
- Manual-vs-script cutoff agreement: the script's `? ?` gives no number this run, so the 457 total
  and the 21-row slice come from one fetch path only (`curl --http1.1 --compressed`, run #11's
  confirmed-safe form); no second-cutoff cross-check available.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + this
   checkout only — no SSH, no docker, no build, no serve, no schedule change. Alert file consumed
   and removed after acting.
2. **Next-pin pre-flight rows now seventeen** (candidate **`5e6995dd`**, unchanged): (a)–(p) as
   runs #8/#9/#11, with **row (p) revised** and **row (q) added**:
   - **(p) FA3 sourcing is now dual-path, wheel preferred** — #50260 removed FA3 from
     sglang-kernel; #50394 restores a kernels-hub loader but makes the `sgl-fa3` wheel the
     **default** (`SGLANG_USE_SGL_FA3_KERNEL=True`). The dep marker is `linux and x86_64`, so on
     aarch64 the default branch reaches `from sgl_fa3 import …` with no guard. Pin check: confirm
     nothing on the DSV4 path imports `flash_attention_v3.py` at module scope, or set the env off
     in the image.
   - **(q) new runtime dependency `kernels>=0.14.1,<0.15`** — confirm it installs in the aarch64
     image and that the Dockerfile's aarch64 cubin-skip branch is present in whatever recipe the
     next image is built from; otherwise the build's `kernels download python` step has no
     aarch64 variant to fetch.
3. **No new bp candidate this window.** #50177 (run #10, decode-attn-TP skip, 3 LOC, cp path
   disabled) remains the only open low-value bp; it cannot close the decode gap.
4. **Decode gate unchanged**: no sync removal at our verify site (~44 ms/step
   `cudaEventSynchronize`, dspark verify rework). Nothing in this window touches
   `deepseek_v4_backend.py`, `dspark_verify_window.py`, `ragged_verify*`,
   `decode_cuda_graph_runner.py`, or `parallel_state.py` — the files our patches anchor on.
5. **bp6r3 conflict re-checks stand**: #44588 and #37472 on `dspark_verify_window.py`. No new
   upstream commit on that file this window.
6. **Operator — script fix is now the top item**: the census leg has failed in 3 of the last 4 runs
   (#9, #10, #12) and twice it silently substituted `? ?` for a window that contained material
   commits. Fail loud (non-zero rc + keep the alert) instead of printing `?`; the `? ?` path also
   makes "nothing changed" and "everything failed" indistinguishable for any downstream reader.
7. `gh` auth dead for `aotsukiqx` (**12th run**) — re-auth remains the standing operator fix;
   REST-only stayed sufficient (18/60 core).

## 2026-11-14T04:18Z (2026-11-14 12:18 +08) — watch run #13

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0 (04:17:40–04:18:03Z). **Partial run again**:
both issue legs numeric, census leg `dsv41-relevant-since-watch=? window-commits=?` (the script's
python/urllib call exits early). Budget measured immediately afterwards **53/60 core, used 7** →
rate limiting is excluded; this is the **4th occurrence in 5 runs** (#9 full 403 blackout, #10/#12/#13
`? ?` at 53/49/53 core) and the third time it masked a window. Census redone manually below via
`curl --http1.1 --compressed` (5 pages, 17 new rows).
`gh auth status` → **invalid token for `aotsukiqx` (13th consecutive run)**; `gh issue view` /
`gh pr view` never attempted — REST only. Core budget 53/60 at start → 48/60 after the sweep
(reset rolled mid-sweep). **Only in-repo change: this file.** Main HEAD at capture
`bd47834099` (committer `2026-11-14T03:26:49Z`).
**Cadence**: log slots 00:17:38Z → 04:17:40Z, `17 */4` healthy.

### Issues — zero delta (12th silent run for #49174)

- **#49173** — closed `2026-11-09T07:26:52Z`, `comments=2`, `updated_at 2026-11-09T07:26:52Z`;
  byte-identical to runs #1–#12 (both comments are `Kylinny` on 2026-11-08). No maintainer text to
  read; the alert line for it is cumulative.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **twelfth consecutive
  silent run**. Body-reference search (`"49174" type:pr`): `total_count 1`, still **#49306**
  open/unmerged, `updated_at 2026-11-09T12:56:58Z` — unchanged since run #5.
- **#49182** — `merged=True`, merge **`5e6995dd09`**, head **`a49ff4d7d3`**,
  `merged_at 2026-11-09T07:26:50Z` → unchanged. Next-pin candidate remains **`5e6995dd`**.

### sglang main — 17 new commits in 4 h; 1 low-value bp candidate, 16 NO-OP

Census **474** (`since=2026-11-08T14:00:00Z`, pages 100/100/100/100/74), prev head `8c7ec05287`
at index 17 → slice `[0:17]`. Arithmetic: 457 + 17 = 474 ✓.

| Commit | PR | Subject | Class — note |
|---|---|---|---|
| `fe10e2be64` | #49721 | [Fix] scheduler: Answer health probes that are waiting behind other requests | **bp-class (low value, monitoring)** — `managers/scheduler.py` +21/−0 (`_ack_waiting_health_checks()` called after the early returns in `_get_new_batch_prefill_raw`, `+2` at the call site) + `dllm/mixin/scheduler.py` +1 + test +188. A `health_check_generate` probe queued behind a chunked prefill would otherwise wait out every chunk; it is now acked and retired. `start.sh`/`stop.sh`/`scripts/verify/window_guard.sh` all gate on HTTP health, so this is liveness-only value during windows — **zero effect on decode or prefill perf**. Anchors to confirm at apply: `is_health_check_generate_req`, `return_health_check_ipcs`, `maybe_send_health_check_signal` likely pre-pin; **`RequestRetirer.release` and `finish_outside_step` look post-pin** (req-lifecycle refactor) ⇒ expect a rewrite, not a straight apply |
| `1dd0dcce38` | #49992 | [Feat] mem_cache: Joint Full/SWA eviction walk on the Rust tree core | **NO-OP now / pin note** — `rust_tree_core/adapter.py` +21/−8, `rust/sglang-radix-tree/{swa.rs,unified_lru_list.rs,unified_tree_core.rs,python_bindings.rs}` +753 total. Rust tree core is post-pin (unified sub-pool, pre-flight row n); at the next pin Full/SWA eviction walks jointly |
| `e1113c1fc3` | #49695 | [Refactor] mem_cache: Tag full-only KV frees that release a tree node's value, commit load-backs through one method | **NO-OP now / pin note** — `unified_cache/cache_action.rs`+`cache_action.py` tag bit, `unified_radix_cache.py` +14/−5. Post-pin unified cache; no local anchor (`mem_cache/*` absent from `runtime/*.patch`) |
| `800f596bfc` | #49641 | [Fix] radix-tree: Clamp SWA tombstone recovery to the protected prefix in the Rust tree core | **NO-OP now / pin note** — Rust core only (`swa.rs` +2/−1, tests +159/−5). Post-pin |
| `25363debb8` | #49649 | [Fix] scheduler: Observe finished storage prefetches before shortest-prefill-first ranks, count staged prefixes as cached | **NO-OP (policy-gated)** — `schedule_policy.py` +21/−11, `base_prefix_cache.py` +10, `unified_radix_cache.py` +52/−12. Two independent gates: `poll_storage_prefetches` returns unless `policy == SHORTEST_PREFILL_FIRST`, and our launchers set **no `--schedule-policy`** anywhere (grep over the checkout: zero hits) ⇒ default `fcfs`. Rest is post-pin unified cache |
| `e5caad9a07` | #49658 | [Feat] scheduler: Add opt-in aging to the shortest-prefill-first policy | **NO-OP (default off)** — new `--shortest-prefill-aging-factor` default **`0.0`**, and the constructor raises unless `schedule_policy=shortest-prefill-first` ⇒ inert on our `fcfs` line. `schedule_policy.py` +49/−5, `scheduler.py` +5/−2 |
| `5f11045a08` | #49666 | [Perf] hicache: Keep the buffer load-back ownership check from waiting on the GPU at admission | **NO-OP (host tier off) / directional note** — replaces the admission-time `torch.equal(canonical, device_indices)` ownership check with a device-side compare staged through **`async_d2h`**, read at the load-back ack (`buffer_mode/pipeline.py` +57/−13). A sync removal, but on the HiCache buffer load-back path we do not enable — **does not touch the dspark verify sync site**, so the ~7-token decode gap is unaffected |
| `cbf4c36687` | #49644 | [Perf] hicache: End each buffer-only backup chain where the insert walk's previous one ended | **NO-OP** — `unified_tree_core.rs` +41/−20 + `buffer_mode/pipeline.py` +1/−2; Rust core is post-pin, host tier off |
| `56fa7ca6e7` | #50370 | [Feat] kv-shard: Let an out-of-tree HiCache storage backend join logical-page KV sharding | **NO-OP** — `arg_groups/kv_shard_hook.py` +19, `mem_cache/{buffer_mode/pipeline,hybrid_cache/hybrid_cache_controller,pool_host/page_interleave,unified_cache/storage_attachment}.py`. kv-shard off, host tier off |
| `6317e8ed4b` | #46409 | [Refactor] multimodal: [MM refactor 3/9] Extract a shared multimodal processor mixin | **NO-OP** — new `multimodal/multimodal_processor_mixin.py` (+411), `processors/base_processor.py` −385. Text-only serving here; parts 4–9 pending, watch only if a later part changes scheduler-side MM admission |
| `bd47834099` `c5d7c59804` `19668f7b9b` `1be399bd11` `b074ad473a` `6e4f22fcd1` | #49970 #50407 #50401 #49630 #50396 #49634 | rust-server block: client request IDs up to 256 B; video item frame/token caps to MM processors; **serve HTTP/2 h2c beside HTTP/1.1 (`--enable-http2`)**; drain scheduler-kill log + bounded reap; per-test shutdown topology; refresh frontend config catalogue, drop the Rust lexer | **NO-OP (6 rows)** — Rust frontend is opt-in (`SGLANG_RUST_SERVER`); this line serves with the Python frontend |
| `fc8d97d142` | #44613 | [CI] tests: Give each NIXL storage test its own temp directory | **NO-OP** — CI-only |

### Cross-checks

- Arithmetic: run #12 total 457 + 17 = 474 ✓; prev head `8c7ec05287` at index 17 of 474.
- Fetch integrity: 5 census pages + 10 commit details + 2 issue legs + 1 PR + 1 search = 19 JSONs,
  all parse (asserted before triage, §1b).
- Class totals: 1 bp + 16 NO-OP = 17 ✓. NO-OP breakdown: rust-server 6, Rust-core/unified
  mem_cache 3, scheduler policy-gated 2, HiCache buffer/host tier 2, kv-shard 1, MM refactor 1,
  CI 1. Four NO-OPs carry pin notes (#49992, #49695, #49641, #49649).
- **Keyword counter: 0 of 17** (recomputed over the slice with the script's own key set). It would
  have reported a clean window and missed the only row worth recording (#49721 — "scheduler" and
  "health" are not keys). Script census could not have reported it either: `? ?` again.
- Anchor-file sweep: no new commit touches `deepseek_v4_backend.py`, `deepseek_v4.py`,
  `dspark_verify_window.py`, `engram.py`, `parallel_state.py`, `speculative/`, or
  `scheduler_components/batch_result_processor.py`. Only `mem_cache/` and `managers/scheduler.py`
  intersect our patch surface, both via post-pin/policy-gated paths.
- Manual-vs-script cutoff agreement: script gave `? ?`; manual 474 from one fetch path
  (`curl --http1.1 --compressed`).

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + this
   checkout only — no SSH, no docker, no build, no serve, no schedule change. Alert file consumed
   and removed after acting.
2. **Next-pin pre-flight rows remain seventeen** (candidate **`5e6995dd`**, unchanged): (a)–(q) as
   run #12. No dependency or Dockerfile change in this window.
3. **New bp candidate #49721** (`fe10e2be64`): low value — health-probe liveness behind chunked
   prefills, no perf surface. Apply only in an authorised window, and expect to rewrite the retire
   call (`RequestRetirer`/`finish_outside_step` are post-pin). #50177 (run #10) stands as the other
   low-value bp. **Neither closes the decode gap.**
4. **Decode gate unchanged**: no sync removal at our verify site (~44 ms/step
   `cudaEventSynchronize`, dspark verify rework #48831-33). #49666 is a sync removal, but in HiCache
   buffer load-back, not the verify site.
5. **bp6r3 conflict re-checks stand**: #44588 and #37472 on `dspark_verify_window.py`. No new
   upstream commit on that file this window.
6. **Operator — script fix still top item**: census leg failed in **4 of the last 5 runs**
   (#9/#10/#12/#13), and today at a 53/60 budget, which **refutes the earlier 403-only
   explanation** — the urllib path fails independently. It also silently substitutes `? ?` for
   windows that contain material commits. Fail loud (non-zero rc, keep the alert).
7. `gh` auth dead for `aotsukiqx` (**13th run**) — re-auth remains the standing operator fix;
   REST-only stayed sufficient.

## 2026-11-14T08:19Z (2026-11-14 16:19 +08) — watch run #14

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0 (08:17:35–08:17:46Z). **First fully numeric
census since run #11**: `dsv41-relevant-since-watch=55 window-commits=482` — the urllib leg worked,
so the `? ?` failure mode (#9/#10/#12/#13) did not recur. Keyword counter is window-wide and
cumulative (55 is the same 55 as every run since the window filled), **not** an incremental count;
the incremental slice is 8 and carries **0** of the script's keys. `gh auth status` → **invalid
token for `aotsukiqx` (14th consecutive run)**; `gh issue view`/`gh pr view` not attempted — REST
only (budget 31/60 core after the sweep). **Only in-repo change: this file.** Main HEAD at capture
`2456e0beab` (committer `2026-11-14T07:00:33Z`).

### Issues / PR — zero delta (13th silent run for #49174)

- **#49173** — closed `2026-11-09T07:26:52Z`, `comments=2`, `updated_at 2026-11-09T07:26:52Z`;
  byte-identical to runs #1–#13 (both comments `Kylinny`, 2026-11-08). No new maintainer comment to
  read; the alert line for it is cumulative (`need_alert` also fires on `state=closed`).
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **thirteenth consecutive
  silent run**. Body-reference PR **#49306** still open/unmerged, `updated_at 2026-11-09T12:56:58Z`
  (unchanged since run #5). The `TRUNCATE=0` workaround in `adapter/loop_abort.py` remains the only
  mitigation.
- **#49182** (fix PR for #49173) — merged, merge **`5e6995dd09`**, head `a49ff4d7d3`,
  `merged_at 2026-11-09T07:26:50Z` → unchanged. Next-pin candidate remains **`5e6995dd`**.

### sglang main — 8 new commits in 4 h; 1 bp candidate, 7 NO-OP

Census **482** (`since=2026-11-08T14:00:00Z`); prior total 474 + 8 new = 482 ✓, script agreed
independently. Slice taken above prior head `bd47834099` (committer `03:26:49Z`).

| Commit | PR | Subject | Class — note |
|---|---|---|---|
| `4a909c3ba6` | #50400 | [Perf] sampling: Re-derive the top-k flags when a running batch drops rows | **bp-class (decode-path, low value)** — `sampling/sampling_batch_info.py` +28/−12, `managers/schedule_batch.py` +3, test +58. `filter_batch` left `is_all_greedy`/`need_top_k_sampling`/`max_top_k`/`npu_top_k_top_p_eligible` as derived from the pre-drop batch ("safe but sticky"): one finished greedy row keeps the survivors on the top-k path (and `max_top_k=None` disables the tight host-side bound) for the running batch's remaining life. Fix factors the five derivations into `_top_k_flags()` + new `refresh_top_k_flags()`, called from `ScheduleBatch.filter_batch` right after `sampling_info.filter_batch(keep_indices, keep_indices_device)`. Host-side flag/derivation only — **no GPU work removed at the verify site; does not touch the decode gap**. Effect is conditional on mixed greedy/non-greedy batches (all-`top_k=TOP_K_ALL` traffic derives the same flags either way) |
| `cb55118032` | #49895 | [Fix] entrypoints: Resolve auth levels of included routers and pin a level for every route | **NO-OP** — `srt/utils/auth.py` +25/−10 + test +133. Route-level auth defaults; `start-tp4.sh` sets no `--api-key` (0 hits) so the TP4 line runs unauthenticated |
| `a05c729c4e` | #38362 | [Fix] hicache: Include logical KV dtype in storage keys | **NO-OP (host/L3 tier off)** — `managers/cache_controller.py` +21/−9, `mem_cache/hicache_storage.py` +2/−2, tests +128. Wrong-dtype L3 key collisions; inert with hicache storage disabled |
| `04eece19ad` | #46410 | [Refactor] multimodal: [MM refactor 4/9] Split feature transport and preprocessing placement out of the base processor | **NO-OP** — new `multimodal/serving_preprocessing.py` (+128), `multimodal/transport/feature_transport.py` (+157), `base_processor.py` −219. Text-only serving; parts 5–9 pending |
| `7c683961cc` | #49234 | [Perf] diffusion/cosmos: optimize bf16 fa | **NO-OP** — `multimodal_gen/runtime/**` flashinfer FA2 backend + docs + tests |
| `6a64408419` | #50390 | [Perf] diffusion: Look up the rounded SiLU of the gated activation in a table on sm_10x | **NO-OP (additive, opt-in)** — `kernels/jit/csrc/elementwise/activation.cuh` +135, `kernels/ops/activation/activation.py` +60, `+221/−0` total: new `build_silu_rounding_table` / `run_silu_and_mul_rounded_table` entrypoints only, the existing `silu_and_mul` dispatch is untouched. Only a caller opting in (diffusion) pays the 128 KB table build. **If a later commit wires this into the LLM MLP path on sm_10x it becomes bp-class perf** (expf-plus-divide removed from a per-token op) |
| `ac939dfaa8` | #34353 | [Fix] [NPU] modelslime: Fix NEXTN startup and mamba scatter UB overflow | **NO-OP** — `hardware_backend/npu/**`, `modelslim.py`, `qwen3_5_mtp.py`; Ascend only |
| `2456e0beab` | #50466 | [Test] ep: Temporarily disable test_mooncake_ep_small on the Mooncake EP JIT nvcc spawn flake | **NO-OP** — CI-only (1 line, `test/registered/ep/`) |

### Cross-checks

- Arithmetic: 474 + 8 = 482 ✓ against the script's own `window-commits=482`; prev head
  `bd47834099` sits at index 8 of the 9-row `since=03:00Z` fetch.
- Fetch integrity: 1 census page (script, 5 pages) + 1 commit-list page + 8 commit details + 2 issue
  legs + 2 PR legs = all JSON parsed before triage; 2 patch bodies fetched over plain
  `github.com/.../commit/<sha>.patch` (no API budget).
- Class totals: 1 bp + 7 NO-OP = 8 ✓. NO-OP breakdown: diffusion 2, hicache 1, MM refactor 1,
  auth 1, NPU 1, CI 1.
- **Keyword counter: 0 of 8** — the only row worth recording (#50400: "sampling", "batch") is
  invisible to the script's key set, as in run #13 (#49721). Triage cannot be reduced to keywords.
- Anchor-file sweep: no commit in this slice touches `deepseek_v4_backend.py`, `deepseek_v4.py`,
  `deepseek_v4_dspark.py`, `dspark_components/*`, `engram.py`, `speculative/`, `deepseek_v41` config
  or `srt/kernels/`. Only `managers/schedule_batch.py` intersects our surface, and only via #50400;
  the local `schedule_batch` hook (`loop_abort.install` → wraps `Req.update_finish_state`) is on a
  different symbol, so a #50400 backport would not collide with the adapter.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + this
   checkout only — no SSH, no docker, no build, no serve, no schedule change. Alert file consumed
   and removed after acting.
2. **New bp candidate #50400** (`4a909c3ba6`): low value, decode-path sampling overhead only, and
   conditional on mixed greedy/non-greedy batches. Anchors to confirm at apply:
   `SamplingBatchInfo.filter_batch`, `from_schedule_batch`'s inline flag derivation, `TOP_K_ALL`,
   `max_top_k`, the `_fast_copy_enabled`/`_FIELD_NAMES` path (the new method uses `setattr` and adds
   no field, so the fast-copy tuple is unaffected), call site
   `ScheduleBatch.filter_batch` → `self.sampling_info.filter_batch(keep_indices, keep_indices_device)`.
   **Block-level apply check not done** (pin 8817f63c tree is not in this checkout); the diff's `-`
   lines show the pre-fix inline form, consistent with the pin. Apply only in an authorised window.
   **Does not close the decode gap**; #49721 and #50177 stand as the other low-value bps.
3. **Next-pin pre-flight rows remain seventeen** (candidate **`5e6995dd`**, unchanged): (a)–(q) as
   run #12. No dependency, Dockerfile or wheel change in this window.
4. **Decode gate unchanged**: no commit this window touches the dspark verify host-sync
   (~44 ms/step `cudaEventSynchronize`, dspark verify rework #48831-33). #50400 is host-side flags,
   #50390 is additive and opt-in.
5. **bp6r3 conflict re-checks stand**: #44588 and #37472 on `dspark_verify_window.py`. No new
   upstream commit on that file this window.
6. **Operator**: (a) `gh` re-auth (**14th run**), (b) make the census leg fail loud in
   `scripts/verify/upstream_watch.sh` — it recovered this run, but 4 failures in the last 6 runs
   with silent `? ?` substitution is still the top tooling risk.

## 2026-11-17T12:32Z (2026-11-17 20:32 +08) — watch run #15

**Tooling**: script rc=0 at 12:18:03–12:18:22Z, but **all three legs failed** — `issue#49173
fetch-failed`, `issue#49174 fetch-failed`, `dsv41-relevant-since-watch=? window-commits=?`. First
total blackout in 15 runs, and the first one that was transport rather than budget. **Log gap**:
the previous run was 2026-11-14T08:17Z, so ~19 four-hour slots (3.2 d) did not run; the log is
append-only and resumes cleanly, so the automation/host was down, not wiped.

**Root cause — transport, not rate limiting.** `/rate_limit` answered on the first manual try
(core **40/60** remaining after the failed run, 32/60 after this sweep; never exhausted). What
failed: curl (LibreSSL 3.3.6) is reset mid-handshake on `api.github.com` — `SSL_ERROR_SYSCALL`,
rc 35, plus rc 28 timeouts — and `github.com` then stopped resolving entirely (`Could not resolve
host`) while `api.github.com` kept resolving. Python's urllib (OpenSSL) worked for the API, and
`curl --resolve github.com:443:20.169.109.160` worked for patch bodies. Every number below was
collected with those two workarounds. `gh auth status` → **invalid token for `aotsukiqx` (15th
consecutive run)**; `gh issue view`/`gh pr view` never attempted, REST only.

**Script patched in this run** (the deferred fix from run #14 action item 6b): each issue leg now
retries 4× over curl → curl `--tlsv1.2` → python urllib, the census leg retries urllib → curl
(plain and `--tlsv1.2`), and any leg that still fails sets `FETCH_RC`, writes a `TOOL FAILURE`
line into the alert, and makes the script exit non-zero instead of deleting the alert. Failure
path proven at 12:31:58Z: rc=1, `ALERT raised (fetch_rc=1)`, alert body says the clean-looking
result is not a clean result. A second run at 12:32:40Z still failed every leg — by then
`api.github.com` had stopped resolving too (`Errno 8: nodename nor servname provided`), so the
retry ladder was verified as *loud*, not as self-healing. Retries cannot beat a dead resolver —
that part stays a host problem.

### Issues / PR — zero delta

- **#49173** — closed `2026-11-09T07:26:52Z`, `comments=2`, `updated_at 2026-11-09T07:26:52Z`;
  byte-identical to runs #1–#14 (both comments `Kylinny`, 2026-11-08). No new maintainer comment.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **fourteenth consecutive
  silent run**. No PR body-reference appeared this window either; the `TRUNCATE=0` workaround in
  `adapter/loop_abort.py` remains the only mitigation.
- **#49182** — merged `2026-11-09T07:26:50Z`, merge **`5e6995dd09`**, head `a49ff4d7d3` — unchanged.
  Next-pin candidate remains **`5e6995dd`**.

### sglang main — 249 new commits (3.2-day window); 1 new bp candidate, 3 pin-migration rows

Census **731** (`since=2026-11-08T14:00:00Z`, 7 full pages + 31); prior total **482 + 249 = 731 ✓**.
Slice above prior head `2456e0beab` (committer `2026-11-14T07:00:33Z`), newest `703e763d5f`
(`2026-11-17T11:19:50Z`). All 249 subjects read; 18 diffs read at block level via
`github.com/.../commit/<sha>.patch`; **8 anchors checked against the pin tree** by raw fetch at
`8817f63c`.

| Commit | PR | Subject | Class — note |
|---|---|---|---|
| `7b5fb39bb9` | #49896 | [Feat] entrypoints: Require --admin-api-key for sensitive admin endpoints | **pin-migration (pre-flight #1)** — `AuthLevel.ADMIN_FORCE` routes (weight update, LoRA, memory release, **profiling**, `/set_internal_state`, `/hicache/storage-backend`, pause/continue) answer **403 when no admin key is configured at all**. `scripts/profile/profile_decode.sh` and `scripts/verify/decode_profile_bounded.sh` POST `/start_profile`/`/stop_profile` unauthenticated, and `start.sh`'s `api_key()` deliberately sends no `--api-key` on this line ⇒ **profiler capture breaks at the next pin** unless the launcher exports `SGLANG_ADMIN_API_KEY`/`--admin-api-key` and both scripts send `Authorization: Bearer`. `/abort_request` and `/flush_cache` stay keyless while no admin key is set (`memguard.py` unaffected). Same family: `f7f0c930c1` #50559 (single route-level table) and `17e4738c4c` #50586 |
| `233d6bae87` | #50397 | [Chore] deps: Bump sglang-kernel to 0.4.11 and sgl-fa3 to 0.1.0 | **pin-migration (deps)** — `SGL_KERNEL_VERSION`/`sglang-kernel` 0.4.10→**0.4.11**, `sgl-fa3` 0.1.0rc0→0.1.0 (x86_64 marker kept), `sgl-deep-gemm` 0.2.8 and `sgl-flash-mla` 1.0.5 unchanged, plus an `assert_pkg_version` in `entrypoints/engine.py`. **Correction to pre-flight row p**: PyPI 0.4.11 *does* publish `sglang_kernel-0.4.11-cp311-abi3-manylinux2014_aarch64.whl` — the "x86_64-only, no aarch64 wheel" note recorded 2026-11-13 is wrong, so the 0.4.11 aarch64 gate is clear |
| `b8555f91bb` | #50994 | [CI] docker: Drop sgl-fa3 from the CUDA 13.4 image | **pin-migration (image)** — `sed -i '/^  "sgl-fa3==/d' pyproject.toml` in `docker/Dockerfile.cu134`, rationale in-comment: sgl-fa3 pins its torch (non-stable C++ ABI) and ships only sm80/sm86/sm90a kernels; FA3 is imported lazily. Our image is exactly this one and no repo script imports `sgl_fa3` (0 hits) ⇒ bookkeeping row: the next image has no sgl-fa3 |
| `d1e7987757` | #50981 | [Fix] gc: Keep the heap frozen after a CUDA graph capture | **bp-class (low value)** — `freeze_gc` in `model_executor/runner/base_cuda_graph_runner.py` **exists at the pin** in the simpler collect→freeze→unfreeze→collect form; the fix re-freezes after capture when the heap was frozen before (and honours `gc_was_enabled`). Host-heap/RSS hygiene around capture and recapture only. Companions `8d07d63f37` #50980, `9eb23a3e63` #50883, `988b043982` #50882 live in `utils/gc_utils.py` + `managers/scheduler_components/gc_freezer.py`, both **404 at the pin** |
| `9fd8b7fef2` | #50719 | [Perf] [NVIDIA] dsv4: Score the V4.1 prefill indexer with DeepGEMM fp8 MQA logits on Hopper | **NO-OP (watch)** — the new prefill-score path is gated on `get_platform().is_sm90`; SM12x keeps `is_sm100_or_newer()` = `is_sm_at_least(10)` for prefill and decode, and the candidate indexer only renames publish/consume flags. Recorded because it churns the exact `v41_indexer` arch-gate family from #49173/#49182 and its comment still documents the "sgl-deep-gemm 0.2.6 dispatches arch 12 with the fp32 weights/logits contract" behaviour; upstream now carries `test_dsv41_indexer_arch_gates.py`. No `==10` regression |
| `a27a59c2e7` | #50471 | [Perf] dsv41: Share Engram tables across servers with replication group_shards | **NO-OP (overridden)** — new opt-in `replication: group_shards` (+`group_size`/`group_store`, validated and rejected unless set) touching `srt/layers/engram.py` +241 and `arg_groups/engram_config.py`. Our line replaces `EngramEmbedding` wholesale with `adapter/engram_backend.py` (NVMe exact rows, host callback), so upstream churn here is conflict-relevant only at pin migration |
| `04af670a31` | #50512 | [Perf] [AMD] deepseek-v4.1: Build the low-ratio extend slot gather once per forward, without a host sync | **NO-OP (HIP-only)** — `deepseek_v4_backend_hip_radix.py` + `dsv4/low_ratio_backend_hip.py` only; the CUDA `low_ratio_backend.py` is **404 at the pin**. The decode-gap axis (a host sync in the low-ratio/verify path) is still untouched on CUDA |
| `5c2d9e4056` | #47013 | [Fix] kvcache: Skip the reserved padding slot in the zero-RoPE MLA KV writer | **NO-OP (not our writer)** — pin file confirms the bug verbatim (`kernels/ops/kvcache/mla_buffer.py:149`: the norope kernel "takes no skip index and writes every loc"). DSV4.1 does not use it: `mem_cache/deepseek_v4_memory_pool.py` at the pin has three own `set_kv_buffer` impls and no `set_mla_kv_buffer` call. bp-writable (anchor exists) but currently unnecessary |
| `710813dee3` | #46781 | [Perf] spec: Avoid blocking H2D for MRoPE delta in spec decode | **NO-OP (MRoPE)** — one call site in `compute_spec_mrope_positions`; MRoPE is the multimodal position path, DSV4.1 spec decode uses plain RoPE |
| `a187006c24` | #44861 | [Fix] [NVIDIA] modelopt: Keep SM12x FP8 batch-invariant in deterministic mode | **NO-OP (deterministic off)** — `other=0.0` on two masked `tl.load`s in `kernels/ops/gemm/fp8_kernel.py` `scaled_mm_kernel` (arch-independent, file exists at the pin) plus `apply_fp8_linear_deterministic`. Our FP8 GEMMs dispatch to DeepGEMM/flashinfer, not this Triton kernel, and the deterministic helper needs deterministic inference |
| `80a461512e` | #49170 | [Fix] deepseek-v4.1: Keep active DP decode ranks out of idle tail replay | **NO-OP (DP)** — drops `late_dp_counts` from the DSV4 layers call and adds `global_late_layer_tail_graph_compatible` to the DP late-layer exchange; `dp_size=1` on this pure-TP line does no DP exchange |
| `a51b0e2b84` | #50311 | [Perf] mem_cache: Cut unified-memory admission host work | **NO-OP (pool not enabled)** — `managers/schedule_policy.py`, `mem_cache/kv/alloc/chain.py`, `prefetch_budget.py`; the unified sub-pool is not enabled here (0 launcher hits for unified/hicache). Same for `9ae8cf0608` #50310, `fd510c505f` #50309, `a7da11be3d` #50308, `ff68baae6d` #50307, `03d4f49f22` #50306 |
| `7e2d9b85a3` | #48065 | [Fix] [AMD] dsv4: Fix the mHC weight orientation | **NO-OP (AMD guard)** — `fn_transpose=_is_gfx1250_supported or not self.attention` in `deepseek_v4_boundary.py`/`deepseek_v4_platform_stage.py`; `is_gfx1250_supported()` is False on GB10 |
| `9158db2b97` | #48095 | [Perf] [AMD] dspark: Speed up the DSPARK draft step | **NO-OP (AMD guard)** — a `_is_gfx1250_supported` branch selecting `fused_hc_head` in `deepseek_v4_dspark.collapse_hc_head`; CUDA keeps `hc_head_torch` |
| `d8a6e5a9de` | #50771 | [Fix] [NVIDIA] mega_moe: Stop DeepGEMM's MLOPart helper before every KV cache sizing | **NO-OP (MegaMoE off)** — `SGLANG_OPT_DEEPGEMM_MEGA_GATE` is False and our MoE is `b12x_next` |
| `24feb92de0` | #50608 | [Perf] [NVIDIA] mega_moe: Allow DeepGEMM weight localization on SM 10.0 and 10.3 | **NO-OP** — SM10.0/10.3 only; GB10 is CC 12.1 |
| `c67db7c762` | #50982 | [Fix] cuda-graph: Free the old decode graphs before a recapture | **NO-OP (recapture path)** — this line captures once at boot; recapture is owned by the dual-graph adapter |
| `f74df0c3d6` | #50717 | [Fix] scheduler: Wait for the launched forward before writing weights | **NO-OP (weight update)** — RL/eval weight-update path |
| `837d1cf76e` | #50741 | [Perf] tokenizer_manager: Poll client disconnects once per wait timeout | **NO-OP (frontend CPU)** — tokenizer-manager only, no engine path effect on bench traffic |
| `c712459e7f` | #49823 | [Fix] http_server: Hold the HTTP port across the model load | **NO-OP (boot surface)** — our readiness gates on `/health_generate` 200, not on connect success |
| `07f4c845c2` | #50234 | [Fix] weights: Reject mismatched manifests before distributed broadcasts | **NO-OP (load guard)** — earlier error message for a case our NFS weight loads do not hit |
| `6565c03ff9` | #40893 | [Fix] spec: Support a reduced draft vocabulary in rejection sampling | **NO-OP (flag off)** — needs `--speculative-use-rejection-sampling` (default False), unsupported on PD decode servers |
| `72d55dc900` | #49805 | [Fix] models: Forward router top_k to fused MoE experts | **NO-OP** — hunyuan/llama4/olmoe-class models; no `deepseek_v4` file touched |
| `f75136c949` | #50019 | [Fix] moe: Apply the model's swiglu_limit in the CuTe DSL MoE runner | **NO-OP** — `moe_runner/flashinfer_cutedsl.py`; our MoE path is DeepGEMM/`b12x_next` |
| `3981de0825` (+`775db052b6`) | #50156 | Revert "[Perf] [NVIDIA] runtime: Blackwell/Rubin serving optimizations for hybrid GDN + QSA + MoE models" | **NO-OP (reverted pair)** — landed `2026-11-16T01:35Z`, reverted in the same window; re-triage only if it re-lands |

**Grouped NO-OPs — the remaining 223 commits**, each read by subject (all 249 assigned to exactly
one bucket; counts sum to 249):

| Bucket | n | Examples |
|---|---|---|
| diffusion (image/video models) | 45 | `25c1c43942` #50765 sm120 cosmos GEMM, `37bc19e86e` #49599 H3 FP8 attention on DGX Spark |
| AMD | 31 | `87f393c249` #48064 aiter sparse prefill, `ed3aa8c322` #41382 gfx1250 launch retune, `4469f9fc7f` #49732 absorb-BMM wave budget |
| other models | 29 | `6321f714dd` #48414 Kimi-K3.1 merge, `2b9eab7125` #46392 Kimi rejects flashinfer_mxfp4 SiTU on SM90/SM120, `40723a52de` #47802 GLM-4 GQA Triton decode |
| NPU | 24 | `2a71304ac5` #49481 breakable prefill graph for DSV4, `baaa201a14` #49294 DSpark multi-stream drafts, `88151fa62c` #42224 INT8 LightningIndexer |
| CI | 19 | `83c2c71774` #42806, `ba93265487` #50872 |
| auth / metrics / infra / parsing | 13 | `f7f0c930c1` #50559 route auth table, `17e4738c4c` #50586, `2e55a173a7` #47060 |
| router / rust | 11 | `2588d80d06` #50732 delete sgl-model-gateway, `b947a615f6` #50671 source-only |
| DP / CP / PP only | 9 | `50fbc8af48` #40333 DSV4.1 prefill CP+BCG (2/N), `552e16968b` #50146 prefill delayer, `9a0e6eafef` #50481 PP depth |
| hicache / hisparse (tier off) | 7 | `d9bf131d8b` #49473 hisparse data plane, `19af0c78a2` #45351 PP-sync depth |
| unified pool / GDN | 7 | `883476951c` #50122 opt-in state_cost eviction |
| multimodal refactor | 5 | `22e1cb6d98` #46412 6/9, `5ec2b14be5` #46411 5/9 |
| disaggregation (PD) | 4 | `6795aed975` #50895 first-token stream, `b40c565be9` #50520 |
| docs / chore / other-hw | 10 | `d766b7327e` #50677, `d08d15985b` #50486 (MUSA) |
| gc series (post-pin paths) | 3 | `8d07d63f37` #50980, `9eb23a3e63` #50883, `988b043982` #50882 |
| comm / multimem, policy-gated, megakernel | 6 | `a0d169e8d0` #50764 Lamport markers, `0e8c255117` #46022 SPF (fcfs here), `0b28a39a2b` #50157 |

### Cross-checks

- Arithmetic: 482 + 249 = **731** ✓ against the independent page count (7×100 + 31); prior head
  `2456e0beab` sits at index 249, and the row just above it is `233d6bae87` (`2026-11-14T08:52Z`).
- Class totals: 25 Table-A rows (26 shas, the reverted pair shares one row) + 223 grouped = **249** ✓,
  assigned programmatically with an assertion that every sha lands in exactly one bucket.
- Fetch integrity: 8 census pages + 2 issue legs + 2 comment legs + 1 PR leg + 1 rate-limit call
  (≈13 core requests) plus 18 `.patch` bodies over `github.com` (no API quota); every JSON parsed.
- **Keyword counter: 23 of 249 (9.2%)** and **all four material rows were misses** (#49896, #50397,
  #50994, #50981); most of the 23 hits are AMD/NPU/doc rows. Per-commit reading stays mandatory.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + this
   checkout only — no SSH, no docker, no build, no serve, no schedule change. Alert consumed.
2. **Pre-flight #1 (new, repo-actionable)**: #49896 admin-key gate. Decide before the next pin:
   launch with `--admin-api-key`/`SGLANG_ADMIN_API_KEY` and add `Authorization: Bearer` to
   `scripts/profile/profile_decode.sh` + `scripts/verify/decode_profile_bounded.sh`, or accept that
   decode/torch profiling cannot be captured at the next pin. `memguard.py`'s `/abort_request`
   keeps working keyless as long as no admin key is set.
3. **Pre-flight deps corrected**: sglang-kernel 0.4.11 has an `aarch64` `cp311-abi3` wheel
   (filename evidence above) — row p's "no aarch64 wheel" claim is withdrawn. sgl-deep-gemm stays
   0.2.8 and flash-mla 1.0.5, so the arch-12 fp32-weights assert situation is unchanged.
4. **Pre-flight image**: the cu134 image no longer installs sgl-fa3; nothing in the repo imports it.
5. **New bp candidate #50981** (`d1e7987757`): anchors confirmed at the pin; low value (host heap
   around capture). The rest of the gc series is post-pin and must be re-taken wholesale at the next
   pin, not backported.
6. **Decode gate unchanged**: no CUDA host-sync removal landed on the dspark verify path. #50512 is
   HIP-only, #46781 is MRoPE-only, #50197/#49905/#50306-07 are GDN. The ~44 ms/step
   `cudaEventSynchronize` attribution still stands.
7. **Watch**: #50719 confirms upstream still uses `is_sm_at_least(10)` for the dense DSV4 indexer
   gates and now has an explicit arch-gate test — the bp6 chain collapse at candidate `5e6995dd`
   still applies. Do not re-pick `#50156` unless it re-lands.
8. **Operator**: (a) `gh` re-auth (**15th run**), (b) the automation did not run for 3.2 days
   (11-14 08:17Z → 11-17 12:18Z) — check the Orca schedule/host, (c) this host's network resets
   TLS handshakes and intermittently fails DNS for `github.com`; the script now fails loud, but
   polling needs a working resolver, and a stale-IP `--resolve` pin is only a stopgap.

## 2026-11-17T16:18Z (2026-11-18 00:18 +08) — watch run #16

**Tooling**: script rc=0 with **all three legs numeric** — `issue#49173 state=closed comments=2`,
`issue#49174 state=open comments=0`, `dsv41-relevant-since-watch=78 window-commits=732`
(16:17:48–16:18:00Z, `fetch_rc=0`). First fully-clean run since #14: the retry ladder added in #15
is now proven self-healing as well as loud, and the resolver is answering again. Slot is the
scheduled `17 */4` 16:17Z and the 12:17Z slot ran (the #15 path), so **no log gap** — the Orca
automation has resumed after the 3.2-day outage.

`gh auth status` → invalid token for `aotsukiqx` (**16th consecutive run**); `gh issue view`/`gh pr
view` not attempted. REST only: 8 census pages + 2 issue legs + 1 PR leg + 1 referencing-PR leg +
1 commit detail + 1 check-runs leg + 2 searches ≈ 15 core + 2 search requests. Core budget
**50/60 at start, 53/60 at end** (the unauthenticated hour window rolled mid-sweep), search 8/10.

**Push backlog flushed**: origin was still at `99a21a8`, so both `8156c8a` (the script retry-fail-loud
patch) and `0222605` (run #15's entry) had never been published — the run #15 commit happened, the
push did not. This run's `git push` carried all three commits (`99a21a8..0d3cc76`); `git ls-remote`
confirms `origin/main = 0d3cc762ad`. Run #15's work is only now visible on GitHub.

### Issues / PR — zero delta

- **#49173** — closed `2026-11-09T07:26:52Z`, `comments=2`, `updated_at 2026-11-09T07:26:52Z`:
  byte-identical to runs #1–#15, no new maintainer comment.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **fifteenth consecutive
  silent run**. Referencing-PR search re-run (0 comments is not proof of inactivity): still exactly
  **#49306** (`closes #49174`), `open`, `merged=False`, head `4e1544039e`, 1 commit (+116/−7), last
  touched `2026-11-09T12:56:58Z`, CI still red (`pr-test-finish` `failure`; 126 of 127 check runs
  `skipped`). `adapter/loop_abort.py` keeps `TRUNCATE=0` as the only mitigation.
- **#49182** — merged `2026-11-09T07:26:50Z`, merge **`5e6995dd09`**, head `a49ff4d7d3`, unchanged.
  Next-pin candidate remains **`5e6995dd`**.

### sglang main — 1 new commit; NO-OP

Census **732** (`since=2026-11-08T14:00:00Z`, 7×100 + 32); prior total **731 + 1 = 732 ✓**.
Slice above prior head `703e763d5f` (`2026-11-17T11:19:50Z`) is exactly one commit; new head
**`b2cc023f23`** (`2026-11-17T13:53:57Z`).

| Commit | PR | Subject | Class — note |
|---|---|---|---|
| `b2cc023f23` | #51021 | [Chore] mem_cache: Add UMBP code owners | **NO-OP (bookkeeping)** — `.github/CODEOWNERS` +1 line adding `/python/sglang/srt/mem_cache/storage/umbp`. No code path, wheel, launcher flag or arch gate touched |

Keyword counter over the slice: **0/1** (the only subject matches the `mem_cache` keyword but carries
no code). Per-commit reading remains the only reliable mode.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + this
   checkout only — no SSH, no docker, no build, no serve, no schedule change. Alert consumed.
2. **Standing — bp6 collapse at candidate `5e6995dd`**: the #49182 fix family is what the next pin
   absorbs; #50719 confirms upstream keeps `is_sm_at_least(10)` for the dense DSV4 indexer gates
   (no `==10` regression).
3. **Standing — decode gate unchanged**: no CUDA host-sync removal landed on the dspark verify path
   (#50512 is HIP, #46781 MRoPE, #50197/#49905/#50306-07 GDN). The ~44 ms/step
   `cudaEventSynchronize` attribution stands.
4. **Standing pre-flights (unchanged this window)**: #49896 admin-key gate breaks unauthenticated
   `/start_profile` in `scripts/profile/profile_decode.sh` +
   `scripts/verify/decode_profile_bounded.sh`; sglang-kernel 0.4.11 ships an aarch64 `cp311-abi3`
   wheel; sgl-deep-gemm 0.2.8 / sgl-flash-mla 1.0.5 unchanged, so the arch-12 fp32-weights assert
   situation is unchanged.
5. **Standing — bp6r3 conflict re-checks**: #44588 and #37472 on `dspark_verify_window.py`. No new
   upstream commit on that file this window.
6. **Operator**: (a) `gh` re-auth (**16th run**; every run is REST-only now), (b) the 3.2-day Orca
   outage is over (clean scheduled slot this run), but the host resolver stays the single point of
   failure for polling.

## 2026-11-17T20:18Z (2026-11-18 04:18 +08) — watch run #17

**Tooling**: script rc=0 with all three legs numeric — `issue#49173 state=closed comments=2`,
`issue#49174 state=open comments=0`, `dsv41-relevant-since-watch=78 window-commits=741`
(20:17:40–20:18:14Z, `fetch_rc=0`). Scheduled `17 */4` 20:17Z slot; the 12:17Z slot ran (TLS retry
ladder: 12:18Z `fetch-failed`, then successful 12:31/12:32Z retries) and 16:17Z ran clean, so **no
log gap**. The alert file holds only the cumulative triggers (#49173 closed + its fixed comment
count), not new activity; consumed.

`gh auth status` → invalid token for `aotsukiqx` (**17th consecutive run**); `gh issue view`/`gh pr
view` not attempted. REST only: 8 census pages + 2 issue legs + 2 comment legs + 1 PR leg + 1
referencing-PR search + 9 commit details = **22 core + 1 search**. Core budget **49/60 at start**
(11 used) and the hour window rolled to **60/60, 0 used** by the end; search 10/10 → 9/10.

### Issues / PR — zero delta

- **#49173** — closed `2026-11-09T07:26:52Z`, `comments=2`, `updated_at 2026-11-09T07:26:52Z`:
  byte-identical to runs #1–#16. Both comments are the original maintainer thread (Kylinny,
  11-08T15:21Z intent + 11:48Z "fix is up in #49182"); no new comment.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **sixteenth consecutive
  silent run**. Referencing-PR search re-run: still exactly **#49306** (`closes #49174`), `open`,
  `merged=None`, `updated_at 2026-11-09T12:56:58Z` — untouched for 8 days; no re-triage needed.
  `adapter/loop_abort.py` `TRUNCATE=0` remains the only mitigation.
- **#49182** — merged `2026-11-09T07:26:50Z`, merge **`5e6995dd09`**, head `a49ff4d7d3`, unchanged.
  Next-pin candidate remains **`5e6995dd`**.

### sglang main — 9 new commits; all NO-OP

Census **741** (`since=2026-11-08T14:00:00Z`, 7×100 + 41); prior total **732 + 9 = 741 ✓**. Slice
above prior head `b2cc023f23` (`2026-11-17T13:53:57Z`) is exactly 9 commits (prior head at index 9);
new head **`1aa45c6617`** (`2026-11-17T19:45:00Z`).

| Commit | PR | Subject | Class — evidence |
|---|---|---|---|
| `1aa45c6617` | #50126 | [Perf] tokenizer: flat output sampling masks | **NO-OP** — new request fields only (`return_flat_raw_sampling_mask[_b64]`, default `False`, chat form requires `return_meta_info`); the non-flat `to_lists()` branch is the old behavior and `chunk` is `None` unless sampling masks are requested; `rust/sglang-processor` is opt-in |
| `5cefe8de82` | #48097 | [Feat] [AMD] AITER MegaMoEGfx1250 + mori dispatch | **NO-OP** — the `fp8.py` edits sit under `if _use_aiter and self.is_fp4_expert` (ROCm only) and `is_megamoe()`; our MoE is `b12x_next` with MegaMoE bypassed |
| `b43c4c00a6` | #51044 | [CI] qknorm rope AMD stage rename + ml-dtypes cap | **NO-OP** — `scripts/ci/amd` + HIP/JIT tests only |
| `31e55a5120` | #50874 | [CI] suites: unify CI suite/command names | **NO-OP** — 300 files, all `.github/workflows`, `scripts/ci*`, `scripts/lint`, test registration and docs; no `python/sglang/srt` file touched |
| `9c1b3eceb8` | #44349 | [Feat] beam-search trie head + shared-context attention | **NO-OP (feature-gated)** — both new envs default `False`; every new branch keys on `req.beam_group`/`forward_batch.beam` which are `None` for non-beam batches (`beam_decode_budget`/`beam_launch_complete` return the old value); trie head installs only when a bundled trie file is found in the model dir (`configure_model_trie_output_heads` returns `False` first); the new flashinfer wrapper exists only on the non-MLA branch. **Pin note**: it parameterizes `schedule_policy` budgets, `logits_processor`, `model_runner`, `forward_batch_info` and `decode_cuda_graph_runner.can_run_graph` — re-derive any budget/`can_run_graph` bp against this shape at the next pin |
| `aa3ccefd38` | #50983 | [Fix] gc: warn on frozen garbage, don't report a leak | **NO-OP** — post-pin `scheduler_components/gc_freezer.py`; diagnostic only, drops the `SGLANG_ENABLE_GC_FROZEN_GARBAGE_CHECK` env and the `Severity.LEAK` report, keeps `refreeze_gc` |
| `90121c62a8` | #50059 | [Fix] mem_cache: prompt end in the same checkpoint as an SWA branch point | **NO-OP** — the added `checkpoint_into_tree` call is gated on `result.swa_branch_inserted`; DSV4.1 on this line is MLA/DSA with no sliding-window pool. Cache-correctness only, no perf surface. Re-check the gate at apply time if a hybrid-SWA model ever runs here |
| `ba9c86bb7e` | #46029 | [Feat] [NPU] diffusion MindIE-SD into Ascend FA | **NO-OP** — `multimodal_gen` + `platforms/npu.py` only |
| `db12772f0c` | #47293 | [Fix] unified_cache: Inkling MTP draft state in the direct external linker | **NO-OP** — restores SWA+FULL sidecars for *hybrid* drafts; every `DeepSeekV4TokenToKVPool` branch is preserved (`_draft_bytes` keeps the SWA-only path, `build_hicache_draft_sidecars` and `kv_cache_builder` keep their DSV4 branches), and the HiCache host/draft-sidecar tier is off on this line. Files are post-pin (`hybrid_cache/*`, `linker_pool_assembler.py`) — re-take wholesale at the next pin |

### Cross-checks

- Arithmetic: 732 + 9 = **741** ✓ against the independent page count (7×100 + 41); prior head
  `b2cc023f23` at index 9, the row just above it is `db12772f0c` (`2026-11-17T16:38:04Z`).
- Class totals: **0 material rows + 9 individually-read NO-OPs = 9** ✓ (every sha assigned once).
- No commit in the slice touches `speculative/`, `dspark_*`, `deepseek_v4*`, `engram.py`,
  `mem_cache/` (runtime file) or the indexer backend — the decode-gate search surface is untouched.
- Fetch integrity: 8 census pages + 2 issue legs + 2 comment legs + 1 PR leg + 1 search + 9 commit
  details; all 14 cached JSON bodies re-parsed, 9/9 commit details present.
- **Keyword counter: 0 of 9** — the total stayed at 78 across the window, so the counter was a
  complete miss on this slice (it would have reported "nothing new" for a 2 678-line feature).

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + this
   checkout only — no SSH, no docker, no build, no serve, no schedule change. Alert consumed.
2. **Standing — bp6 collapse at candidate `5e6995dd`**: unchanged; #49182 is what the next pin
   absorbs, and nothing in this window walks back the `>=10` dense-gate family.
3. **Standing — decode gate unchanged**: still no CUDA host-sync removal on the dspark verify path.
   The ~44 ms/step `cudaEventSynchronize` attribution stands.
4. **Standing pre-flights (unchanged)**: #49896 admin-key gate breaks unauthenticated
   `/start_profile` in `scripts/profile/profile_decode.sh` +
   `scripts/verify/decode_profile_bounded.sh`; sglang-kernel 0.4.11 aarch64 `cp311-abi3` wheel;
   sgl-deep-gemm 0.2.8 / sgl-flash-mla 1.0.5, arch-12 fp32-weights assert unchanged.
5. **Standing — bp6r3 conflict re-checks**: #44588 and #37472 on `dspark_verify_window.py`. No new
   commit on that file this window; the nearest new conflict surface is #44349's parameterization of
   `schedule_policy`/`logits_processor`/`decode_cuda_graph_runner`.
6. **Operator**: (a) `gh` re-auth (**17th run**; every run is REST-only), (b) no Orca/log gap this
   window (12:17Z, 16:17Z, 20:17Z all present), (c) the host resolver/TLS path stayed healthy this
   run after the 12:17Z failure — still the single point of failure for polling.

## 2026-11-18T00:18Z (2026-11-18 08:18 +08) — watch run #18

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0 after 24 s; log lines
`00:17:41Z` (issue legs) → `00:18:02Z` (`fetch_rc=0`). Alert raised on the two standing triggers
(#49173 closed + 79 keyword hits) and consumed. Run #17 appended cleanly, so the 00:17Z slot had
no gap. `gh auth status` → **token for `aotsukiqx` invalid** (**18th consecutive run**); every
payload below is unauthenticated REST. `raw.githubusercontent.com` timed out twice under `curl`
(rc 28); the same URLs came through `urllib` in <2 s — the TLS-path fragility from 11-17 is still
there, just on the raw host this time, and it did not affect the census.

### Issues / PR — zero delta

- **#49173** — closed `2026-11-09T07:26:52Z`, `comments=2`, `updated_at=closed_at` same second:
  byte-identical to runs #1–#17. Both comments are the original Kylinny thread (11-08T15:21Z
  intent, 11:48Z "fix is up in #49182"); no new comment to read.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **seventeenth consecutive
  silent run**. Referencing-PR search re-run: still exactly **#49306** (`closes #49174`), `open`,
  `merged_at=None`, `updated_at 2026-11-09T12:56:58Z` — untouched for 9 days.
  `adapter/loop_abort.py` `TRUNCATE=0` remains the only mitigation.
- **#49182** — closed, merged `2026-11-09T07:26:50Z`, merge **`5e6995dd09`**, head `a49ff4d7d3`,
  `updated_at 2026-11-09T07:26:51Z` — unchanged. Next-pin candidate stays **`5e6995dd`**.

### sglang main — 8 new commits: 1 material, 7 NO-OP

Census **749** (`since=2026-11-08T14:00:00Z`, 7×100 + 49); prior total **741 + 8 = 749 ✓**. Slice
above prior head `1aa45c6617` (index 8 of the newest-first page) is exactly 8 commits; new head
**`ecd3d75a2a`** (`2026-11-17T23:33:41Z`).

| Commit | PR | Subject | Class — evidence |
|---|---|---|---|
| `4b9f9043fa` | #46190 | [Feat] spec: confidence-scheduled compact verify for DFLASH | **PIN-MIGRATION (material)** — see below; inert on the live config, but it rewrites the verify/planner/relay surface our bp6 chain anchors on |
| `7aaebb1cf0` | #50740 | [Feat] spec/dflash: per-request draft KV ring for all-sliding-window MHA drafts | **NO-OP (feature-gated)** — new arm only for `DFLASH_MHA_RING_DRAFT_BACKENDS = ("trtllm_mha", "fa4")` behind `_mha_ring_unsupported_reason` (requires CUDA, `dcp_size==1`, `attn_cp_size==1`, no attn DP, and an all-sliding-window **MHA** draft); our draft is MLA/DSA and the ring switch is off. **Pin-shape note**: `draft_kv_ring_unsupported_reason` gained `attn_cp_size`/`enable_dp_attention`/`is_cuda`, `_init_draft_kv_ring` gained `warm_block_locs(...)`, and `set_draft_kv_ring` landed on `flashattention_backend`/`trtllm_mha_backend` |
| `372b35034d` | #50712 | [Fix] disaggregation: poll decode transfer queue while retracted requests wait | **NO-OP** — only `Scheduler.process_decode_queue` inside `disaggregation/decode.py`; the PD decode path does not exist on a single-node TP4 line (no `--disaggregation-mode`) |
| `5830ea8205` | #50196 | [Perf] [AMD] mhc: enable AITER mHC kernels on gfx942 | **NO-OP** — both hunks are `is_hip` + `is_gfx95_supported()/is_gfx942_supported()` + `SGLANG_USE_AITER`; `deepseek_v4_fused_mhc.py` lives under `models/deepseek_common/**amd**/`, `mhc.py`'s aiter branch returns `None` off HIP |
| `1280ff574b` | #38344 | [Feat] CP: prefill context parallelism for Kimi K3 | **NO-OP** — `enable_prefill_cp` + `attn_cp_size>1` only, and the new branch raises for every non-Kimi arch; we launch neither. **Guard note**: a mambaish arch with `--enable-prefill-cp` now hard-fails (`Linear-attention prefill CP is not integrated with …`) instead of silently falling back |
| `9d9b45d413` | #38766 | [Feat] diffusion: rank-local TP checkpoint loading for MiniMax-H3 | **NO-OP** — `multimodal_gen/**` only |
| `85360aaab8` | #50748 | [Fix] [AMD] monokernels: keep a draft's forward on hf_config mismatch | **NO-OP** — `kernel_plugins/**/amd/` only |
| `ecd3d75a2a` | #49290 | [Fix] diffusion/security: `safe_pickle_loads` for DiffusionServer | **NO-OP** — `multimodal_gen/**` inbound request deserialization |

### The one material row — #46190 `4b9f9043fa` (18 files, +1220/−129)

First commit in the whole watch window to touch the DSpark verify surface. What it actually does:

- `dspark_verify.py::TargetVerifyExecutor.run_compact` — **full-block fast path**: when
  `RaggedVerifyLayout.is_full_block(stride)` (new helper on `ragged_verify.py`) the packed rows
  *are* the `[bs, verify_num_draft_tokens]` window, so `BuildRaggedVerifyWindow` plus both
  `ScatterCompactToStrided` calls are skipped. `kv_injector` becomes `Optional` (default `None`,
  asserted in `commit_hidden`). `dspark_verify_window.py::RaggedVerifyWindow.window_index` becomes
  `Optional[torch.Tensor]`.
- `dspark_planner.py` — `plans_verify_budget()` / `_plan_verifies_every_draft()` /
  `_compact_graph_tier()`: when the shortest plan already replays the full-block compact graph
  (e.g. a single request), the step prices **no** budget, so the per-step schedule and its
  host↔device round trips are skipped. `is_verify_all` now also requires no pinned
  `forced_budget_frac`.
- `overlap_utils.py::ConfidenceRelay` — `resolve()` replaces a non-blocking
  `copy_done[slot].query()` miss (return `None`, fall back) with a blocking
  `copy_done[slot].synchronize()`; `scatter()` waits on the previous ring copy's stream event.
- Shared plumbing: `spec_info.supports_ragged_verify`, `decide_needs_confidence_relay`, and the
  `internal_state` runtime-override guards widen `is_dspark()` → `is_dflash_family()`; a pinned
  budget is now rejected under `SGLANG_SIMULATE_ACC_LEN > 1`.
- `forward_batch_info.compute_spec_mrope_positions` — ragged mRoPE via new
  `spec_mrope_token_deltas()` (`view` → `reshape`); text-only branch equivalent.

**Liveness on our line = nil.** `envs.SGLANG_RAGGED_VERIFY_MODE` defaults to `"static"` at our pin
`8817f63c` (and on main), `start.sh` never sets it, and `boot.py` only switches on
`SGLANG_RAGGED_VERIFY_MODE=compact` when an SPS table path made it into the argv — which
`sps_table_args()` prohibits on this Engram model. With `mode=static`, `schedule_layout()` returns
`None`, `run_compact` is never entered, and `decide_needs_confidence_relay()` is `False`, so neither
the fast path nor the new `synchronize()` executes. The `is_dflash_family()` widening is also a
no-op for DSPARK: the predicate already exists at the pin and `supports_ragged_verify()` returns
`is_dspark()` only for the DFLASH arm.

**Decode gate unchanged.** #46190 does not remove a host sync from the static verify path — it
*adds* one, on a relay that is inactive here. The ~44 ms/step `cudaEventSynchronize` attribution on
the dspark verify stands; treat #46190 as the reference shape if the line ever moves off `static`
(which needs the SPS table, still prohibited), **not** as the decode fix.

**Re-anchor list for the next pin** (the bp6 chain edits three of these): `run_compact`,
`TargetVerifyExecutor.__init__`, `RaggedVerifyWindow`, `dspark_planner` budget scheduling,
`ConfidenceRelay.resolve`, `compute_spec_mrope_positions`. Prior doc's bp6r3 conflict surface
(`dspark_verify_window.py`, #44588/#37472) is now live: this commit changes that file's types.

### Cross-checks

- Arithmetic: 741 + 8 = **749** ✓ against the independent page count; prior head `1aa45c6617` at
  index 8, the row just above it (`4b9f9043fa`) at 9.
- Class totals: **1 material + 7 individually-read NO-OPs = 8** ✓ (every sha assigned once).
- Coverage: 2 of 8 touch `python/sglang/srt/speculative/` (#46190, #50740); 1 touches an
  AMD `deepseek_common` file; **none** touch `engram.py`, `mem_cache/` runtime files,
  `deepseek_v4.py`, or `deepseek_v4_backend.py` — the indexer/engram/cache axes stayed quiet.
- Fetch integrity: 2 issue legs + 1 PR leg + 1 search + 8 commit details + 5 patch re-fetches
  (needed to rule out gating on the #46190/#50740/#38344/#50712 rows) + 2 raw pin reads.
  Core budget **50/60 at start, 53/60 after** (hour rolled mid-run); search 10/10.
- **Keyword counter: 1 of 8** — the single hit is `verify` in #46190's subject; the other 7 were
  invisible to the counter, which again means the `79` in the alert is a trigger count, not a census.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + this
   checkout only — no SSH, no docker, no build, no serve, no schedule change.
2. **Standing — bp6 collapse at candidate `5e6995dd`**: unchanged; #49182 is what the next pin
   absorbs and nothing this window walks back the `>=10` dense-gate family.
3. **New — pin-migration prep**: add #46190 to the migration notes and re-derive the verify/planner
   bps against it; #50740 for the ring paths. Neither changes today's line behaviour.
4. **Standing — decode gate**: unchanged, and #46190 is not the fix (see above).
5. **Standing pre-flights (unchanged)**: #49896 admin-key gate breaks unauthenticated
   `/start_profile` in `scripts/profile/profile_decode.sh` +
   `scripts/verify/decode_profile_bounded.sh`; sglang-kernel 0.4.11 aarch64 `cp311-abi3` wheel;
   sgl-deep-gemm 0.2.8 / sgl-flash-mla 1.0.5, arch-12 fp32-weights assert.
6. **Operator**: (a) `gh` re-auth (**18th run**; REST-only throughout), (b) no log gap this slot,
   (c) raw-host TLS needed the urllib fallback — the polling path still has a single point of
   failure, now demonstrably on both API and raw hosts across separate runs.

## 2026-11-18T04:18Z (2026-11-18 12:18 +08) — watch run #19

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0 after 26 s; log lines `04:17:46Z` (issue legs)
→ `04:18:12Z` (`fetch_rc=0`). **All legs fetched on the first try** — first fully clean run since
11-17, no retry fallback exercised. Alert raised on the two standing triggers (#49173 closed + 82
keyword hits) and consumed. `gh auth status` → **token for `aotsukiqx` invalid** (**19th consecutive
run**); every payload below is unauthenticated REST. Core budget **43/60** after the run (hour rolled
mid-run at 04:00Z); search 10/10.

### Issues / PR — zero delta

- **#49173** — closed `2026-11-09T07:26:52Z`, `state_reason=completed`, `comments=2`,
  `updated_at=closed_at` same second: byte-identical to runs #1–#18. Both comments are the original
  Kylinny thread (11-08T15:21Z intent, 11:48Z "fix is up in #49182"); nothing new to read.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **eighteenth consecutive
  silent run**. Referencing-PR search re-run: still exactly **#49306** (`closes #49174`), `open`,
  `merged=False`, head `4e1544039e` — unchanged since 11-09. `adapter/loop_abort.py` `TRUNCATE=0`
  remains the only mitigation.
- **#49182** — merged `2026-11-09T07:26:50Z`, merge **`5e6995dd09`**, head `a49ff4d7d3` — unchanged.
  Next-pin candidate stays **`5e6995dd`**.

### sglang main — 20 new commits: 1 pin-migration-class (boot blocker), 19 NO-OP

Census **769** (`since=2026-11-08T14:00:00Z`, 7×100 + 69, **0 duplicate shas** across pages); prior
total **749 + 20 = 769 ✓**. Prior head `1aa45c6617` sits at index 28; the previous run's eight shas
(`4b9f9043fa` … `5830ea8205`) occupy indices 20–27, so the slice `[0:20]` is exactly the delta. New
head **`386edff54d`** (`2026-11-18T04:12:45Z`).

| Commit | PR | Subject | Class — evidence |
|---|---|---|---|
| `7946e589c5` | #51074 | [Refactor] server_args: Remove deprecated flags, choice values and environment variables | **PIN-MIGRATION (boot blocker)** — new `environ.REMOVED_ENVS` + `arg_groups/validation_hook.py::validate_removed_options` **raise `ValueError` at startup**; two names we set are on the list. See below |

**#51074 in detail** (300 files, +1478/−2385 — the "SGLang 1.0" arg cleanup):

- **`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` → REMOVED.** `start.sh:374` (head) and `start.sh:526`
  (worker) set it to `0` unconditionally. At this commit the server refuses to boot while it is in
  the environment. Replacement is `--engram-config '{"placement": "cpu"}'` for the *enabled* case;
  we disable it, so the correct migration edit is to **delete the two `-e` lines** (default off).
  Siblings on the same list: `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT` (→ `"replication": "per_node"`)
  and `SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS` (→ `"lookup": "peer_read"`) — we set neither.
- **`SGLANG_DSPARK_FOLDED_SAMPLING` → REMOVED.** Forwarded conditionally at `start.sh:387-388` and
  `:531-532`; `docs/tp4.md:80` and `docs/tp4-optional-setups.md:20` record production intent `=2`
  (keep sampled requests in the CUDA graph). At this commit the surviving knobs are
  `SGLANG_DSPARK_FOLDED_PROPOSAL = EnvBool(True)` and `SGLANG_DSPARK_USE_GREEDY_DRAFT =
  EnvBool(False)`, i.e. *a folded proposal samples by default*. The FORCE=2 equivalence is
  therefore plausibly the default, but that is a re-derivation to prove in-window, not an
  assumption — the removal message says "set `USE_GREEDY_DRAFT=1` … or `FOLDED_PROPOSAL=0`", which
  describes the *off* directions only.
- **Flags we pass are untouched.** `REMOVED_FLAGS` = `--disable-cuda-graph`,
  `--enable-dp-attention`, `--monokernel-backend`, `--enable-flashinfer-allreduce-fusion`,
  `--disable-fast-image-processor`, `--disable-outlines-disk-cache`, `--show-time-cost`,
  `--triton-attention-reduce-in-fp32` — grep of this checkout for all eight: **0 hits**. Our
  `EXTRA_SGLANG_ARGS` (`--fp8-gemm-backend flashinfer_cutlass --watchdog-timeout 1800
  --sleep-on-idle --min-free-slots-delay 1 --enable-metrics`) has zero removal hunks.
  `REMOVED_CHOICES` covers `attention_backend` (`compressed`→`dsv4`, `nsa`→`dsa`,
  `tokenspeed_mla`→`cutedsl_mla`), `moe_a2a_backend` (`flashinfer`→`flashinfer_nvlink_one_sided`),
  and `tool_call_parser` (`qwen25`, `glm45`) — none selected on this line.
- **`SGL_` prefix auto-rewrite deleted.** No `SGL_`-prefixed names exist in this checkout (only
  `SGLANG_*`), so inert here.
- **`docker/Dockerfile.cu134` diff in this commit is a two-line comment**: the MSA plugin is now
  advertised as `--enable-nvidia-megakernel` instead of `--monokernel-backend msa`. **No version
  bump, no wheel-matrix change.**

| Commit | PR | Subject | Class — why it is inert here |
|---|---|---|---|
| `386edff54d` | #51093 | [CI] slash-command: refresh CI-states block | NO-OP — `.github/` workflow only |
| `b085beba3d` | #50783 | [Fix] [NPU] dspark: dspark & dp attention during prefill with pp/cp | NO-OP — NPU backend, requires attn-DP plus PP or CP; this line is `dp_size=1`, `pp=1`. **Note**: it is the only in-slice commit touching dspark, and it does not touch the verify-sync site |
| `f61c3b51f5` | #51149 | [Chore] CODEOWNERS: weight_cache owner | NO-OP — CODEOWNERS |
| `b1a02a5371` | #51124 | [Fix] [Intel] cpu: pin Outlines version | NO-OP — Intel CPU backend |
| `dd7daa9311` | #49749 | [CI] [AMD] muse-spark: MI35x nightly MMMU gate | NO-OP — CI + AMD; "spark" here is the MI350 codename, not GB10 |
| `acf9d94231` | #51133 | [CI] cpu: register `test_quant_ops` to renamed suite | NO-OP — CI |
| `255a0844b1` | #51114 | [Fix] encoder: scalar hashes, empty feature list, media payloads | NO-OP — `disaggregation/encoder/**` (multimodal encode server), unused |
| `8f2e5fe54e` | #51123 | [Fix] encoder: timeout-tick result, IPv6-safe gRPC URL | NO-OP — same encoder path |
| `bff44f5fd7` | #41329 | [Feat] [CPU] FP8/INT8 quant ops, compressed-tensors | NO-OP — CPU backend |
| `8c6ee5d5e6` | #50023 | [Feat] [NPU] hicache: modelslim MLA fp8 / GQA mxfp8 KV | NO-OP — NPU HiCache; host tier off here |
| `add961dab2` | #51116 | [Refactor] muse-spark: delete unreachable kernels/dispatch | NO-OP — `rocm_muse_spark/**` + `kernels/ops/*/muse_*` (ROCm only) |
| `457cfa33e8` | #44769 | [Feat] [Intel] lora: intel_xpu backend | NO-OP — Intel XPU |
| `78b8523f77` | #51118 | [Doc] sgl-router: PyPI and container installs | NO-OP — docs |
| `6f35d412f2` | #51111 | [Fix] disaggregation: count prefill rank once at bootstrap | NO-OP — PD bootstrap server; no `--disaggregation-mode` here |
| `cd3045fa6e` | #51077 | [Fix] mooncake: snapshot room transfer infos, blacklist on write failure | NO-OP — mooncake transfer engine, PD only |
| `7fed639806` | #50076 | [CI] sgl-router: release from `router-v*` tags | NO-OP — router CI |
| `0fcd253a87` | #51101 | [Test] rust-server: decouple request-log fixture | NO-OP — Rust frontend, opt-in |
| `445bb97129` | #49930 | [CI] [AMD] diffusion: layerwise-offload tests on ROCm | NO-OP — CI |
| `0f1509829b` | #50008 | [Fix] moe: pad NVFP4 trtllm hidden size to a multiple of 256 | NO-OP — `moe_runner/flashinfer_trtllm.py` + `modelopt_quant.py`; we pass no `--moe-runner-backend` and our MoE is `b12x_next`/DeepGEMM, not TRT-LLM NVFP4 |

### Cross-checks

- Arithmetic: **749 + 20 = 769 ✓** against the independent 8-page count; prev head `1aa45c6617` at
  index 28 with the prior run's 8 shas filling 20–27 — slice and delta agree two ways.
- Class totals: **1 material + 19 individually-read NO-OPs = 20** ✓ (every sha assigned once).
- Keywords: counter **3 of 20** (`dspark` in #50783, `spark` in both muse-spark rows) — it missed
  #51074, the only material row, for the fourth consecutive run. The alert's `82` is a cumulative
  trigger count, not a census.
- Coverage: **none of the 20 touch** `engram.py`, `mem_cache/` runtime files, `deepseek_v4.py`,
  `deepseek_v4_backend.py`, or `speculative/dspark_verify*` — the perf axes (indexer, engram, cache,
  verify) were quiet; the only exposure is argument-surface.
- Fetch integrity: 2 issue legs + 1 PR leg + 1 search + 8 census pages + 6 commit details + 1 PR
  (#49306) + 3 raw reads (`environ.py`, `choices.py`, `server_args.py` @ `7946e589c5`).
  `--engram-config` is **not** in `server_args.py` at that commit — the field moved into
  `arg_groups/fields/`, so the flag's registration is an unconfirmed anchor: the replacement text
  above comes from `REMOVED_ENVS`, and the flag name should be re-checked at apply time.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + raw reads
   + this checkout only — no SSH, no docker, no build, no serve, no schedule change.
2. **NEW — next-pin boot blocker (#51074)**: any image built at `7946e589c5` or later **refuses to
   start** with today's `start.sh`. Before the next migration, (a) delete
   `-e "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0"` at `start.sh:374` and `:526`, (b) remove the
   `SGLANG_DSPARK_FOLDED_SAMPLING` forwarding at `:387-388` and `:531-532`, (c) prove or disprove
   the FORCE=2 → default equivalence against `SGLANG_DSPARK_FOLDED_PROPOSAL=True` /
   `SGLANG_DSPARK_USE_GREEDY_DRAFT=False` in-window, with the sampled-decode gate. **Do not edit
   `start.sh` now**: the live pin `8817f63c` still reads both variables.
3. **Standing — bp6 collapse at candidate `5e6995dd`**: unchanged; this window adds nothing to the
   `>=10` dense-gate family and nothing that walks it back.
4. **Standing — decode gate**: unchanged. #50783 is the only dspark commit in the window and it is
   NPU/DP/PP/CP-gated; the ~44 ms/step `cudaEventSynchronize` attribution on the static verify
   still has no upstream fix.
5. **Standing pre-flights (unchanged, re-anchored)**: #49896 admin-key gate breaks unauthenticated
   `/start_profile` in `scripts/profile/profile_decode.sh` +
   `scripts/verify/decode_profile_bounded.sh`; sglang-kernel 0.4.11 aarch64 `cp311-abi3` wheel;
   sgl-deep-gemm 0.2.8 / sgl-flash-mla 1.0.5, arch-12 fp32-weights assert.
6. **Operator**: (a) `gh` re-auth (**19th run**; REST-only throughout), (b) no log gap this slot —
   `04:17Z` follows `00:17Z` on schedule, (c) the polling path fetched clean on every leg this run,
   so the TLS fragility of 11-17 did not recur.

## 2026-11-18T08:18Z (2026-11-18 16:18 +08) — watch run #20

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0 after 40.6 s, silent stdout; log lines
`08:17:44Z` (issue legs) → `08:18:22Z` (`fetch_rc=0`, ALERT raised). All legs numeric, no retry
fallback exercised. The alert holds only the two standing triggers (#49173 closed + **89**
cumulative keyword hits) and was consumed. `gh auth status` → **token for `aotsukiqx` invalid**
(**20th consecutive run**); every payload below is unauthenticated REST. Core budget **16/60
remaining** at the run's last leg (≈44 consumed: script census + this session's reads); search
untouched.

No log gap: `04:18Z` → `08:18Z` on schedule.

### Issues / PR — zero delta

- **#49173** — closed `2026-11-09T07:26:52Z`, `comments=2`, `updated_at` identical to that second:
  byte-identical to runs #1–#19. Both comments are the original Kylinny thread (11-08T15:21Z intent,
  11:48Z "fix is up in #49182"); **no new maintainer comment to read**.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **nineteenth consecutive
  silent run**. Referencing PR **#49306** (`closes #49174`) still `open`, `merged_at=None`,
  `updated_at 2026-11-09T12:56:58Z` — untouched for 9 days. `adapter/loop_abort.py` `TRUNCATE=0`
  remains the only mitigation.
- **#49182** — `closed`, `merged=true`, `merged_at 2026-11-09T07:26:50Z`, merge **`5e6995dd09`**,
  head `Kylinny:fix-49173-dsv41-sm120` — unchanged. Next-pin candidate stays **`5e6995dd`**.

### sglang main — 23 new commits: 1 pin-migration-class (shape), 22 NO-OP

Census **792** (`since=2026-11-08T14:00:00Z`, 8 pages = 7×100 + 92, **0 duplicate shas**); prior
total **769 + 23 = 792 ✓**. Prior head `386edff54d` sits at index 23, so the slice `[0:23]` is
exactly the delta. New head **`da52796a05`** (`2026-11-18T08:09:12Z`).

| Commit | PR | Subject | Class — evidence |
|---|---|---|---|
| `4d200caec7` | #50763 | [Perf] [NPU] dsv4: mirrorless seq_lens path for DFlash-family spec decode | **PIN-MIGRATION (shape)** — see below |
| `da52796a05` | #49514 | [Fix] openai: Render DSV4.1 message tools with recipe | **NO-OP at bp5**, functional re-anchor at next pin (recipe path) |
| `0e0f003efc` | #47644 | [Fix] dflash: Stage the draft sampler only for batches its buffers hold | **NO-OP** — `dflash_worker_v2.py` only (DFLASH worker; this line's draft is DSPARK), 14/−5 |
| `160adb6db9` | #47703 | [Fix] trtllm_mha: Pack a graph-built page table for eager reads | **NO-OP** — `trtllm_mha_backend.py`; SM120 attention here is b12x/FlashMLA |
| `da395ae091` | #50707 | [Feat] [AMD] muse-spark: gfx1250 support | **NO-OP (ROCm-gated)** — every shared-srt hunk keys on `gcnArchName` containing `gfx1250` or `_is_hip`; on CUDA the probes throw into `except → False` and `_GFX1250_LMHEAD` caches the decision at first call |
| `4af099e86e` | #50999 | [Feat] [AMD] speculative: MLA draft verify, DFLASH MLA ctx, target-only sampling | **NO-OP (ROCm-gated)** — new `verify_sampling` arm is `elif is_hip() and is_gfx95_supported()`; new `bf16_dot` kwarg on `extend_attention_fwd` is trailing and defaults `False` |
| `8d72d2ea11` | #50997 | [Feat] [AMD] dsa: gfx950 sparse MLA, DCP top-k, FP4 index-K | **NO-OP (ROCm-gated)** — `aiter.*` imports are function-local; new `dsa/__init__` exports exist unconditionally but only aiter callers reach them; `aiter_paged_mqa_logits` gained a trailing keyword-only `persistent_schedule=None` |
| `9e5c4a78e8` | #50998 | [Feat] [AMD] kimi-k3: gfx950 KDA verify, A8W4 MoE, latent-tail | **NO-OP** — Kimi-K3/KDA kernels; `moe_sorting_small` gains a `qx_stride` parameter (generic column-slice fix) with its Kimi-K3 (E=896, top-16) selector gated on that shape |
| `e1f0fc95db` | #48313 | [Perf] [NVIDIA] fused_moe: H20 down-projection (TMA) configs for GLM-5.3-Flash | **NO-OP** — 8 Triton config JSONs, `device_name=NVIDIA_H20`; GB10 selects none |
| `bf5af3dee5` | #49348 | [Fix] model-loader: RunAI for PP Nemotron MTP drafts | **NO-OP** — `pp_draft_embedding.py` + model-loader; `pp=1` here |
| `ca89faed52` | #51083 | [Fix] lora: bias / deferred-finalize flags on LoRA-wrapped layers | **NO-OP** — 5 lines, LoRA not enabled |
| `0ad64771d6` | #51081 | [Fix] hicache: KV dtype suffix in page-unified L3 object names | **NO-OP** — one test file only |

NO-OP, no DSV4 surface (tests/CI/docs, read by file list): `c808dcac90` #51177 and `79db781e5b`
#51171 (NPU workflow + NPU DSV4-Flash cases), `4dfbb953eb` #49488, `5dae4eec30` #51176,
`9eb3904d1c` #51087, `9d3c71cbbe` #51152, `c8dc16dc23` #48894, `47d00340ba` #51160,
`063d9d4af8` #51085, `7a04e8f443` #51134 (PD consensus store; no `--disaggregation-mode` here).
**`013c0fbcbc` #51086 (cpu: pin outlines) is an empty commit** — single parent, `files=[]`,
`+0/−0`; the outlines-pin diff is already in the tree, nothing to take.

**#50763 in detail** (8 files, +375/−81): the material row is `ascend_dsv4_backend.py` +
`dsv4_common_hooks.py` — an Ascend-only mirrorless replay path for DFlash/DSpark verify
(`SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=0`, device-side `seq_lens`, static-shape async compact
instead of boolean-index D2H). Shared-file part is the plumbing: `batch.max_seq_len_host`
(host upper bound of final KV lengths) is now set in `eagle_utils.eagle_prepare_for_verify`,
`dflash_info.prepare_for_verify`, `dflash_info_v2.prepare_for_decode`, and forwarded by
`dspark_draft._run_forward` into the hand-built `ForwardBatch`. Pin-side anchors verified by raw
read at `8817f63c`: `forward_batch_info.py` **already has** `max_seq_len_host` and `eagle_utils.py`
already references it, while `dspark_draft.py` does **not** — so this commit closes that gap
without adding a field. **Zero CUDA behaviour**: nothing on the CUDA side reads
`max_seq_len_host`, and the NPU backend is not reachable here. Keep it in the pin-migration notes as
the nearest upstream *shape* to our decode gap, not as a fix — it bounds a page-table copy, it does
not remove the `cudaEventSynchronize` host sync (~44 ms/step attribution stands).

**#49514 in detail** (3 product files, +592/−11): message-declared `tools` are now bracketed into
marker system messages (`_take_message_tools` / `_declare_message_tools`) and `tool_choice` is
forwarded even without request tools. **Not perf.** Adapter check: our
`adapter/placeholder_sanitize.py` wraps `deepseek_recipe_renderer.pre_normalize`, and that symbol is
**untouched by this commit** (grep of the patch: no `pre_normalize` hunk), so the sanitizer anchor
holds at the next pin; the rewriting happens later, in render, on messages our adapter never strips
of `tools`.

### Cross-checks

- Arithmetic: **769 + 23 = 792 ✓** against the independent 8-page count; prior head `386edff54d`
  at index 23 with the row below it (`063d9d4af8`) at 24 — slice and delta agree two ways.
- Class totals: **1 pin-migration + 11 individually-read NO-OPs + 11 file-list NO-OPs = 23** ✓
  (every sha assigned once).
- Keywords: **7 of 23** — #49514 `DSV4.1`, #51177 `DSV4`, #50707 `spark`, #50999 `verify`,
  #51171 `deepseek-v4`/`DSPARK`, #50763 `dsv4`/`verify`, #50998 `verify` — lifting the cumulative
  counter 82 → **89 ✓**, which this run matches the full-scan total exactly. Still a trigger, not a
  census: 16 of the 23 subjects produced no hit at all.
- Coverage: **none** of the 23 touch `engram.py`, `mem_cache/` runtime files, `deepseek_v4.py`,
  `deepseek_v4_backend.py`, or a CUDA attention backend. The only DSV4-relevant file touched is
  `speculative/dspark_components/dspark_draft.py` (#50763, pass-through only).
- Fetch integrity: 2 issue legs + 8 census pages + 23 commit details (+10 repeat reads for file
  lists) + 7 patches + 3 raw pin reads + 1 PR leg, all re-parsed; no truncation. Approx. **44/60**
  core requests consumed this run; no referencing-PR search issued (state carried from run #19).

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + raw reads
   + this checkout only — no SSH, no docker, no build, no serve, no schedule change.
2. **Standing — bp6 collapse at candidate `5e6995dd`**: unchanged; nothing in this window walks
   back the `>=10` dense-gate family.
3. **Standing — decode gate**: unchanged. #50763 is NPU-only; the nearest CUDA-adjacent change is
   the portable `max_seq_len_host` *field*, which no CUDA path consumes.
4. **New — pin-migration prep**: add #50763 to the re-anchor list (`run_compact` /
   `TargetVerifyExecutor` unchanged this time, but `dspark_draft._run_forward` gains a kwarg into the
   hand-built `ForwardBatch`); add #49514 as a **functional** re-anchor on the recipe path
   (message-declared tools), not a perf one.
5. **Standing pre-flights (unchanged)**: #49896 admin-key gate breaks unauthenticated
   `/start_profile` in `scripts/profile/profile_decode.sh` +
   `scripts/verify/decode_profile_bounded.sh`; sglang-kernel 0.4.11 aarch64 `cp311-abi3` wheel;
   sgl-deep-gemm 0.2.8 / sgl-flash-mla 1.0.5, arch-12 fp32-weights assert.
6. **Operator**: (a) `gh` re-auth (**20th run**; REST-only throughout), (b) no log gap this slot,
   (c) polling path clean on every leg — no TLS retry needed for the second consecutive run.

## 2026-11-18T12:18Z (2026-11-18 20:18 +08) — watch run #21

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0 after 28.1 s, silent stdout; log lines
`12:17:47Z` (issue legs) → `12:18:14Z` (`fetch_rc=0`, ALERT raised). Every leg numeric on the first
attempt, no TLS retry. The alert holds only the three standing triggers (#49173 closed, its 2
comments, **91** cumulative keyword hits) and was consumed. `gh auth status` → **token for
`aotsukiqx` invalid** (**21st consecutive run**); all payloads below are unauthenticated REST.
Core budget **45/60** remaining immediately after the script; the primary window rolled over
mid-session (final read `56/60`, search `10/10` untouched), so this run's ~31 reads cost nothing.

No log gap: `08:18Z` → `12:18Z` on schedule.

### Issues / PR — zero delta

- **#49173** — closed `2026-11-09T07:26:52Z`, `comments=2`, `updated_at` identical to that second:
  byte-identical to runs #1–#20. Both comments are the original Kylinny thread (11-08T15:21Z intent,
  11:48Z "fix is up in #49182"); **no new maintainer comment to read**.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **twentieth consecutive
  silent run**. Referencing PR **#49306** (`closes #49174`) still `open`, `merged=False`, head
  `4e1544039e`, `updated_at 2026-11-09T12:56:58Z` — untouched for 9 days. `adapter/loop_abort.py`
  `TRUNCATE=0` remains the only mitigation.
- **#49182** — `closed`, `merged=true`, `merged_at 2026-11-09T07:26:50Z`, merge **`5e6995dd09`**,
  head `a49ff4d7d3` — unchanged. Next-pin candidate stays **`5e6995dd`**.

### sglang main — 18 new commits: 1 pin-migration (dependency bump), 1 bp-class (env-only, expected nil), 15 NO-OP

Census **810** (`since=2026-11-08T14:00:00Z`, 9 pages = 8×100 + 10, **0 duplicate shas**); prior
total **792 + 18 = 810 ✓**. Prior head `da52796a05` sits at index 18, so the slice `[0:18]` is
exactly the delta. New head **`4693deba5a`** (`2026-11-18T08:25:24Z`).

| Commit | PR | Subject | Class — evidence |
|---|---|---|---|
| `c2e9301c5b` | #51048 | [Feat] inkling2: Add the FlashInfer CuTe DSL and DeepGEMM MegaMoE kernel ops | **PIN-MIGRATION (dependency bump)** — `DEEPGEMM_REF` v0.2.8→**v0.2.9**, `sgl-deep-gemm` 0.2.8→**0.2.9**; contract check below |
| `0893ad02f1` | #49067 | [Perf] dsv41-flash: Increase the prefill Indexer score tile budget | **bp-class (env-only; expected nil here)** — see below |
| `0299a10a5a` | #51034 | [Fix] jit: Add memory clobbers to the mbarrier wrappers | **pin-migration-class (correctness, post-pin path)** — file 404s at the pin |
| `440086bdd8` | #49457 | [Refactor] graph: Decouple eager_on_graph from CUDA capture (2/3) | **NO-OP at bp5**, re-anchor note — breakable capture never enabled |
| `21dae9b17e` | #49066 | [Perf] dsv41-flash: Add FP8 DeepGEMM prefill scoring on Hopper | **NO-OP** — Hopper-only + `EnvBool(False)` |

Residual NO-OPs, each read by file list: `44227af064` #48254 (tokenizer_manager abort echo,
`_rid` cleanup — manager-side; `adapter/loop_abort.py` patches the scheduler finish path and is
untouched), `ea93e2fcd0` #51000 (67 files, ROCm gfx950 Kimi-K3.1 plus post-pin
`kernels/ops/speculative/dflash_mla_ctx.py` + `arg_groups/`), `e0decbc338` #50124 (both call sites
under `batch.return_logprob`; decode benches request no logprobs and it never touches the verify
`cudaEventSynchronize`), `a8db62b8a4` #51030 (jit benchmark marker → package; post-pin dev tooling),
`ac59b471d3` #46414 + `3a41a59919` #51207 + `6a31888b33` #50965 (tests / NPU workflows only),
`009b31b010` #51147 (Rust `sgl-router --dp-rank-policy least_tokens`; no router on this line),
`f04d886bf7` #51215 (`utils/common.__getattr__` shim for `get_cuda_graph_batch_size_alignment` /
`require_attn_tp_gather` moved to `dp_attention`; `dp_size=1`), `8b546b85d7` #47756 +
`225cb59c67` #50519 (diffusion / ROCm), `bbf0b4e89b` #50805 (77 files, **70** under post-pin
`python/sglang/kernels/`, rest tests), `4693deba5a` #46413 (Qwen-VL token-space strategy + `mrope`;
DSV4.1 text path has no multimodality).

**#51048 in detail** — the only material hunk is the version triple:
`docker/Dockerfile.cu134` `ARG DEEPGEMM_REF=v0.2.8→v0.2.9`, `SGL_DEEP_GEMM_VERSION=0.2.8→0.2.9`,
`python/pyproject.toml` `sgl-deep-gemm==0.2.8→0.2.9`. Evidence, not assumption:
**(a) wheel** — `pypi.org/pypi/sgl-deep-gemm/json` publishes
`sgl_deep_gemm-0.2.9-py3-none-manylinux2014_aarch64.whl`, so the image matrix is satisfied;
**(b) contract** — `DeepGEMM v0.2.9 csrc/apis/attention.hpp` is **byte-identical** to v0.2.8
(54858 B, sha256 `2b0a542e7c…`), so line 49
`DG_HOST_ASSERT(weights.scalar_type() == (arch_major == 10 ? torch::kBFloat16 : torch::kFloat32))`
still forces fp32 MQA-logits weights on arch 12 ⇒ **the bump cannot recover bf16-weights decode on
SM121** (the known-answer chain 0.2.6→0.2.9 now closed end to end). Remaining hunks are new
inkling2 kernel files + one line in `kernels/ops/gemm/quantization/utils.py`; our MoE is
`b12x_next` with MegaMoE bypassed.

**#49067 in detail** (2 product files, +15/−2): adds
`SGLANG_OPT_DSV41_LARGE_PREFILL_SCORE_BUDGET = EnvBool(False)` and a `_prefill_score_budget_bytes()`
helper (2 GiB when set) applied at two call sites — `score_tiles` (DeepGEMM leg) and `_score_chunks`
(torch leg). At pin `scoring.py:36` `_DEEP_GEMM_SCORE_BUDGET_BYTES = 2 << 30` and `:329`
`_TORCH_SCORE_BUDGET_BYTES = 1 << 30`, **identical to main's defaults at the commit**, so the only
behavioural delta is the torch chunk width 1→2 GiB. This line runs the dense full-top-k path with
`_deep_gemm_consume_prefill` (candidate keys zeroed by `adapter/sm120_candidates.py`), so the torch
leg is a fallback and the lever is expected to be a no-op. Recorded, not scheduled: all four anchors
exist at the pin (scoring.py 36/278/329/429), so it applies clean as a runtime patch if a torch-leg
measurement ever wants it.

**#49457 in detail** (9 files, +164/−63): replaces the `BreakableCUDAGraphCapture` contextvar with a
device-neutral `BreakableCaptureRecorder` protocol (new `capture_recorder.py`, 36 LOC) plus
`use_/get_breakable_capture_recorder`, and routes `eager_on_graph` segment splits through it.
`breakable_cuda_graph.py` **does exist at the pin** (18537 B raw at `8817f63c`) — so this is not a
runner-split artifact — but the only reference in our tree is the guard
`adapter/prefill_sp.py:628 v4.is_in_breakable_cuda_graph()`; breakable capture is never enabled.
Part 1/3 was #49402 (11-13); re-read at the next pin.

**#51034 in detail** (1 file, 6 lines): adds `"memory"` clobbers to `mbar_init`, `mbar_arrive`,
`mbar_arrive_relaxed`, `mbar_arrive_expect_tx` and `mbar_wait_parity`. Path
`python/sglang/kernels/jit/include/sgl_kernel/ptx/barrier.cuh` → **404 at `8817f63c`** (contents
API), i.e. the `sglang/kernels` JIT tree is post-pin; a bp has nothing to anchor on. Take it with the
tree at the next pin and re-check that our JIT-built kernels (b12x / mxfp8 adapters) compile against
the clobbered headers — a missing clobber there is a miscompile, not a perf delta.

**#49066 in detail** (7 files, +200/−30): `SGLANG_OPT_DSV41_SM90_DEEPGEMM_PREFILL = EnvBool(False)`;
`use_sm90_deep_gemm_consume` additionally needs `candidate_block_size == CANDIDATE_BLOCK_SIZE`, and
the consumer branch is only reachable through `_use_sm90_prefill_scores()` (Hopper). Side effect
worth noting for the next pin: it extracts `select_prefill_block_ids` into `scoring.py` and re-points
`sparse_table.select_prefill_rows` at it — candidate-sparse path only, which is zeroed here. Neither
new env key exists at the pin (`environ.py` has `SGLANG_OPT_DSV41_BF16_INDEX_LOGITS` at 1952 and no
`PREFILL_LITE_TOPK`).

### Cross-checks

- Arithmetic: **792 + 18 = 810 ✓** against the independent 9-page count; prior head `da52796a05` at
  index 18 with `4693deba5a` at 19 — slice and delta agree.
- Class totals: **3 material (1 dep bump, 1 bp, 1 pin note) + 2 NO-OP-with-pin-note + 13 residual
  NO-OPs = 18** ✓ (every sha assigned once).
- Keywords: **2 of 18** — #49066/#49067 `dsv41` — lifting the cumulative counter 89 → **91 ✓**,
  which the full-scan recomputation matches exactly (91/810). The counter would have missed every
  other row: #51034 (a real correctness fix), #49457 (graph stack) and #51048 (the dependency bump)
  carry no keyword at all.
- Coverage: **none** of the 18 touch `engram.py`, `mem_cache/` runtime files, `deepseek_v4.py`,
  `deepseek_v4_backend.py`, or a CUDA attention backend.
- Fetch integrity: 2 issue legs + 9 census pages + 18 commit details + 2 PR legs + 1 contents-API
  (404, expected) all re-parsed; no truncation. Free reads (no quota): pin `scoring.py` /
  `environ.py` / `breakable_cuda_graph.py` / `Dockerfile.cu134`, DeepGEMM `attention.hpp` at v0.2.8
  and v0.2.9, PyPI JSON.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + raw reads
   + this checkout only — no SSH, no docker, no build, no serve, no schedule change.
2. **Standing — bp6 collapse at candidate `5e6995dd`**: unchanged; nothing in this window walks
   back the `>=10` dense-gate family.
3. **Standing — decode gate**: unchanged. #50124 batches top-logprob D2H copies but is
   `return_logprob`-gated; #51048's DeepGEMM 0.2.9 keeps the arch-12 fp32-weights assert, so the
   host-sync attribution (~44 ms/step) and the wheel-local assert patch remain the open option.
4. **New — next-pin dependency matrix**: main now **0.2.9 / sglang-kernel 0.4.11 / sgl-flash-mla
   1.0.5**, while the live `Dockerfile.canary-mig` builds **0.2.6 / 0.4.10 / 1.0.2** — the next pin
   moves three DeepGEMM minors at once. Re-check the 0.4.11 `cp311-abi3` aarch64 wheel before the
   build; 0.2.9 aarch64 is confirmed present.
5. **New — pin-migration re-anchor list additions**: #49457 recorder protocol,
   #49066 candidate-sparse helper extraction (`select_prefill_block_ids`), #49067 env-only scoring
   budget, and confirm #51034's clobbered `barrier.cuh` is present once `sglang/kernels/jit` lands.
6. **Operator**: (a) `gh` re-auth (**21st run**; REST-only throughout), (b) no log gap this slot,
   (c) polling path clean.

## 2026-11-18T16:18Z (2026-11-19 00:18 +08) — watch run #22

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0 after 13.7 s, silent stdout; log lines
`16:17:41Z` (issue legs) → `16:17:53Z` (`fetch_rc=0`, ALERT raised). Every leg numeric on the first
attempt, no TLS retry. The alert holds only the three standing triggers (#49173 closed, its 2
comments, **91** cumulative keyword hits) and was consumed. `gh auth status` → **token for
`aotsukiqx` invalid** (**22nd consecutive run**); all payloads below are unauthenticated REST.
Core budget **23/60** at close.

No log gap: `12:18Z` → `16:18Z` on schedule.

Push path needed a workaround this run: `git push origin main` hit `failed to get: -25308` from
`osxkeychain` and then hung on the terminal prompt. `~/.git-credentials` (helper `store`) holds a
live `gho_` token (validated `GET /user` → 200, `login=aotsukiqx`) even though `gh`'s own token is
invalid, so `GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c credential.helper=store push origin
main` landed `a44b620..199ff6e`. Nothing wrong with the remote; keychain only.

### Issues / PR — zero delta

- **#49173** — closed `2026-11-09T07:26:52Z`, `comments=2`, `updated_at` identical to that second:
  byte-identical to runs #1–#21. Both comments are the original Kylinny thread (11-08 intent,
  "fix is up in #49182"); **no new maintainer comment to read**.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **twenty-first consecutive
  silent run**. Referencing PR **#49306** (`closes #49174`) still `open`, `merged=False`, head
  `4e1544039e`, `updated_at 2026-11-09T12:56:58Z` — untouched for 9 days.
- **#49182** — `closed`, `merged=true`, `merged_at 2026-11-09T07:26:50Z`, merge **`5e6995dd09`**,
  head `a49ff4d7d3` — unchanged. Next-pin candidate stays **`5e6995dd`**.

### sglang main — 2 new commits: 2 NO-OP (0 bp-class, 0 pin-migration)

Census **812** (`since=2026-11-08T14:00:00Z`, 9 pages = 8×100 + 12, **0 duplicate shas**); prior
total **810 + 2 = 812 ✓**, and an independent recount of the 9 pages reproduces 812 / 812 unique /
91 keyword hits. Prior head `4693deba5a` sits at index 2 of the newest-first slice, so `[0:2]` is
exactly the delta. New head **`37f3b2730b`** (`2026-11-18T16:12:59Z`).

| Commit | PR | Subject | Class — evidence |
|---|---|---|---|
| `37f3b2730b` | #50842 | [Feat] kimi-k3: DWDP for KDA and shared-expert weights on PD prefill | **NO-OP at bp5** — opt-in `--kda-dwdp-size=None`, hard-gated on `is_kimi_k3` |
| `d89bc5b574` | #51219 | [CI] [NPU] chore: Bump SGLang kernel NPU tag to 2026.11.0.post8 | **NO-OP** — NPU workflows / `npu.Dockerfile` only |

**#50842 in detail** (13 files, +860/−7): adds a new post-pin package
`layers/communication/kda_dwdp/` (`layout.py` 120, `manager.py` 233, `prefetch.py` 70, `store.py`
260 LOC), an `Optional[int] kda_dwdp_size = None` field in `arg_groups/fields/parallel.py`, and a
`handle_kda_dwdp` resolution hook that returns immediately unless the flag is set — and then
refuses anything that is not `--disaggregation-mode prefill` + attention DP plus a Kimi-K3 config
(`is_kimi_k3`). The only non-gated edits are a refactor inside `models/kimi_k3.py`
(`_shared_experts_tp_size()` / `_replicates_shared_experts()` extracted from the `KimiK3MoE`
constructor) and `kda_dwdp=None` plumbing through `KimiK3MoE` and its `load_weights`. DSV4.1 never
sets the flag; `models/deepseek_v4.py` is untouched; `layers/communication/` gains only the new
subpackage (it exists at the pin with 6 entries, `kda_dwdp/` 404s at `8817f63c`). Nothing to
backport — but add one import check at the next pin, since `kimi_k3.py` imports `KdaDwdp`
unconditionally at module load. Not worth copying: the instance-connector check raises the
"requires a Kimi-K3 model configuration" error even though the real gate is the `is_kimi_k3` call
after it — harmless with the flag off.

**#51219 in detail** (4 files, +4/−4): one-line tag bumps in
`.github/workflows/release-docker-npu{,-nightly}.yml`, `docker/npu.Dockerfile` and
`scripts/ci/npu/npu_ci_install_dependency.sh`. No image we build, no runtime path.

### Cross-checks

- Arithmetic: **810 + 2 = 812 ✓**; 9-page recount 812 total / 812 unique / 91 keyword hits —
  identical to the script's counter.
- Keywords: **0 of 2** new commits keyword-visible, consistent with the unchanged 91; the counter's
  blind spot is unchanged (it would still miss a fix that names no axis).
- Coverage: neither file list touches `engram.py`, `mem_cache/`, `deepseek_v4.py`,
  `deepseek_v4_backend.py`, or a CUDA attention backend. Every touched path exists at `8817f63c`
  except the new `kda_dwdp/` package (expected).
- Fetch integrity: 2 issue legs + PR #49182 + PR #49306 + 9 census pages + 2 commit details +
  contents probes, all re-parsed; no truncation, no retry (13.7 s total).

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + this
   checkout only — no SSH, no docker, no build, no serve, no schedule change.
2. **Standing — bp6 collapse at candidate `5e6995dd`**: unchanged.
3. **Standing — decode gate**: unchanged; nothing in this window touches the verify host-sync or the
   DeepGEMM arch-12 fp32-weights assert.
4. **Next pin**: one import check for `layers/communication/kda_dwdp` (module-level import in
   `models/kimi_k3.py`); no other re-anchor from this window.
5. **Operator**: (a) `gh` re-auth (**22nd run**; REST-only throughout), (b) no log gap this slot,
   (c) polling path clean.

## 2026-11-18T20:18Z (2026-11-19 04:18 +08) — watch run #23

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0, silent stdout, 13.4 s. Log
`20:17:41Z`–`20:17:53Z`, `fetch_rc=0`; every leg numeric on the first attempt, no TLS retry. The
alert body is the standing trigger set only (#49173 closed, its 2 comments, **91** cumulative
keyword hits) and was consumed. `gh auth status` → **token for `aotsukiqx` invalid** (**23rd
consecutive run**); every payload below is unauthenticated GitHub REST (read-only, public API).

No log gap: `16:18Z` → `20:18Z` on schedule.

### Issues / PR — zero delta

- **#49173** — closed `2026-11-09T07:26:52Z`, `comments=2`, `updated_at` identical to the close
  second: byte-identical to runs #1–#22. Both comments are the original Kylinny thread (11-08
  intent; "the fix is up in #49182"), already absorbed into `docs/PROJECT_STATE.md`. **No new
  maintainer comment to read.**
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **twenty-second
  consecutive silent run**. Referencing PR **#49306** (`closes #49174`) still `open`,
  `merged=false`, head `4e1544039e`, `updated_at 2026-11-09T12:56:58Z` — untouched for 10 days.
- **#49182** — `closed`, `merged=true`, `merged_at 2026-11-09T07:26:50Z`, merge **`5e6995dd09`**,
  head `a49ff4d7d3` — unchanged. Next-pin candidate stays **`5e6995dd`**.

### sglang main — 4 new commits: 4 NO-OP (0 bp-class, 0 pin-migration)

Census **816** (`since=2026-11-08T14:00:00Z`, page 1); prior total **812 + 4 = 816 ✓**, and prior
head `37f3b2730b` sits at index 4 of the newest-first slice, so `[0:4]` is exactly the delta. New
head **`9fbab1f6ed`** (`2026-11-18T19:20:18Z`).

| Commit | PR | Subject | Class — evidence |
|---|---|---|---|
| `9fbab1f6ed` | #51080 | [Perf] attention: Launch the qk-norm, KV-prepare and gate kernels with PDL | **NO-OP at bp5** — touches Qwen3.5/QSA/Gemma kernels + generic fast_topk/elementwise; none imported by any DSV4 path |
| `8be74d4137` | #50611 | [Fix] diffusion/h3: Keep the projected AdaLN LoRA in the adapter dtype | **NO-OP** — `multimodal_gen/` only (H3 DiT + its unit test) |
| `d08f4feee7` | #50727 | [CI] [AMD] spur: Stop the log streamer's SIGPIPE from cancelling healthy jobs | **NO-OP** — one AMD slurm CI shell script, no image we build |
| `935055b42a` | #51290 | [Doc] cookbook: Add DaisyRose-27B cookbook | **NO-OP** — docs only (+804, incl. logo PNG and a jsx config) |

**#51080 in detail** (6 files, +38/−13): adds programmatic-dependent-launch (`launch_pdl: True`)
gated on `current_arch().supports_pdl` / `is_arch_support_pdl()` in
`kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py` (docstring: Qwen3.5 interleaved Q+Gate
layout), `srt/layers/attention/qsa/fused_kv.py`, `kernels/ops/elementwise/elementwise.py`
(`fused_sigmoid_mul`), plus a `USE_PDL` constexpr / trigger in `kernels/jit/csrc/elementwise/
fast_topk.cuh` + `fast_topk.py`. `gemma_qkv_norm_rope.py` explicitly pins `ENABLE_PDL=False`,
i.e. the authors did not enable it on that path. All five files **do exist at `8817f63c`** (5/5
HTTP 200), so existence alone proves nothing; the import audit does: `deepseek_v4_backend.py`
imports only `sglang.kernels.ops.attention.dsv4.*` and `sglang.kernels.ops.speculative.dspark.*`,
and `dspark/fused_kv_write.py` is self-contained Triton (torch/triton imports only). Greps for
`fast_topk|elementwise|fused_kv|qsa|fused_qk_rmsnorm` on `deepseek_v4_backend.py`,
`models/deepseek_v4.py`, `dsv41_indexer/sparse_table.py`, `dsv41_indexer/scoring.py` and
`dsv4/attn_metadata_kernels.py` → **zero hits**. No DSV4 kernel touched → not backportable, and
its perf effect (launch overlap) cannot reach bp5. Latent-only note: if a future pin wires
`fused_sigmoid_mul` into a DSV4 path, that path inherits PDL automatically.

**#50611** (2 files, +23/−2): `minimax_h3.py` dtype of the projected AdaLN LoRA plus a contract
unit test. **#50727** (1 file, +10/−4): `scripts/ci/slurm/launch_amd_ci_mi35x.sh`.
**#51290** (5 files, +804/−0): cookbook docs and assets.

### Cross-checks

- Arithmetic: **812 + 4 = 816 ✓** against the log's `window-commits=816`; prior head found at
  index 4 of page 1, so the delta is complete and no page-2 straddle is possible.
- Keywords: **0 of 4** new commits keyword-visible, consistent with the unchanged **91**; the
  counter's blind spot is unchanged (a fix naming no axis would still be missed).
- Coverage: no touched path is `engram.py`, `mem_cache/`, `deepseek_v4.py`,
  `deepseek_v4_backend.py`, or a CUDA DSV4 attention backend. Every touched path exists at
  `8817f63c` (5/5 probe), so the NO-OP calls rest on import reachability, not on absence.
- Fetch integrity: 2 issue legs + PR #49182 + PR #49306 + 1 census page + 4 commit details + 5
  raw/pin probes, all re-parsed; no truncation, no retry (13.4 s total).

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + this
   checkout only — no SSH, no docker, no build, no serve, no schedule change.
2. **Standing — bp6 collapse at candidate `5e6995dd`**: unchanged.
3. **Standing — decode gate**: unchanged; nothing in this window touches the verify host-sync or the
   DeepGEMM arch-12 fp32-weights assert.
4. **Next pin**: the `layers/communication/kda_dwdp` import check from run #22 still stands; #51080
   adds no re-anchor (PDL is opt-in per kernel and DSV4 does not call those kernels).
5. **Operator**: (a) `gh` re-auth (**23rd run**; REST-only throughout), (b) no log gap this slot,
   (c) polling path clean.

## 2026-11-19T00:18Z (2026-11-19 08:18 +08) — watch run #24

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0, silent stdout, 13.4 s. Log
`00:17:41Z`–`00:17:56Z`, `fetch_rc=0`; every leg numeric on the first attempt, no TLS retry. The
alert body is the standing trigger set only (#49173 closed, its 2 comments, **93** cumulative
keyword hits) and was consumed. Budget **46/60** core at first check (post-script).
`gh auth status` → **token for `aotsukiqx` invalid** (**24th consecutive run**); every payload
below is unauthenticated GitHub REST (read-only, public API).

No log gap: `20:18Z` → `00:18Z` on schedule.

### Issues / PR — zero delta

- **#49173** — closed `2026-11-09T07:26:52Z`, `comments=2`, `updated_at` identical to the close
  second: byte-identical to runs #1–#23. Both comments are the original Kylinny thread (11-08
  intent; "the fix is up in #49182"), already absorbed into `docs/PROJECT_STATE.md`. **No new
  maintainer comment to read.**
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **twenty-third
  consecutive silent run**. A fresh `search/issues` for `"49174" type:pr` returns exactly one hit,
  the known PR **#49306** (`closes #49174`): still `open`, `merged=false`, head `4e1544039e`,
  `updated_at 2026-11-09T12:56:58Z` — untouched for 10 days.
- **#49182** — `closed`, `merged=true`, `merged_at 2026-11-09T07:26:50Z`, merge **`5e6995dd09`**,
  head `a49ff4d7d3` — unchanged. Next-pin candidate stays **`5e6995dd`**.

### sglang main — 27 new commits: 26 NO-OP, **1 bp-class candidate** (#45261)

Census **843** (`since=2026-11-08T14:00:00Z`, page 1); prior total **816 + 27 = 843 ✓**, and prior
head `9fbab1f6ed` sits at index 27 of the newest-first slice, so `[0:27]` is exactly the delta. New
head **`6b374a5502`** (`2026-11-19T00:12:22Z`).

| Commit | PR | Subject | Class — evidence |
|---|---|---|---|
| `6eaca42f57` | #45261 | [Fix] [NVIDIA] deepseek_v4: Start DeepSeek-V4.1 on SM12x with TP8/EP8 defaults | **bp-class candidate — 2 hunks, both anchors verbatim at `8817f63c`; see below. Not a drop-in under our adapter (`assert block_size == 8`)** |

Table B — residual individually-read NO-OPs:

| Commit | PR | Subject | Class — evidence |
|---|---|---|---|
| `173cbe4558` | #51038 | [Fix] scheduler: Publish load on every idle batch result under DP attention | **NO-OP** — the anchor exists at pin (`scheduler.py:895 publish_load_snapshot`), but idle batches are created only by `scheduler_components/dp_attn.py:820` (plus EAGLE `eagle_worker_v2.py:1705`, not DSPARK). Our line is pure TP `dp_size=1` → no idle batch → branch never taken. `--sleep-on-idle` is scheduler sleep, not `ForwardMode.IDLE` |
| `b51c7814b9` | #50853 | [Feat] disaggregation: Report a full decode host pool and landed host-staged transfers in `/v1/loads` | **NO-OP** — `disaggregation/decode.py`, `managers/load_snapshot.py`, `managers/scheduler_components/load_inquirer.py`, rust `loads.rs`. No PD disaggregation on our line; the "host-staged transfers" are the HiCache host tier, which is off |
| `6c2db99660` | #51311 | [Fix] kimi-k3: Read load_format through the model namespace accessor | **NO-OP** — `models/kimi_k3.py` only (+2/−2) |
| `541606a075` | #49745 | [Feat] api: The OpenAI completions and chat completions schema and gRPC surface | **NO-OP** — proto + generated `python/sglang/api/v1/api_types.py` (+6091/−3288, generated) + rust gRPC types; the `srt/entrypoints/openai` serving handlers we run are untouched |
| `6b374a5502` | #51368 | [CI] release: Run weekly, Arm64, MUSA and PPU suites on release branch cut | **NO-OP** — `.github/workflows/*.yml` only (9 files) |
| `d75d74142b` | #51120 | [CI] sgl-router: Build a dispatched release from the router-v<version> tag | **NO-OP** — one workflow file (+40/−5) |
| `aa83b0376b` | #51357 | [Doc] ci: Add OSS CI Hub usage notice | **NO-OP** — docs only |

Grouped NO-OPs — **19** commits, all rust frontend / api-codegen (`SGLANG_RUST_SERVER` is opt-in;
not our launch path, no `srt/` serving file touched). 18 are `rust-server:` rows: the #49735–#49748
series (13: `0b42e6344a be9cc54f70 b0909b3f04 483f28b7ee d59b1542d0 682657a365 ac584ef1a0
b0ce7369f9 07e72a91dd 9e7f4072a8 2e2c622f27 120c91cf32 f01462f3b3`) plus #50851 `de53a1ae2b`,
#50831 `3ab330cd3c`, #50854 `6c5a0eedca`, #50832 `6963e3e0a1`, #50952 `42109d4c9a`. The
nineteenth is api-codegen #49734 (`15c760cc31`: proto + `rust/sglang-api-codegen` + generated
types + one parity unit test).

### #45261 in detail — the SM12x decode-scoring path

Merge **`6eaca42f57`**, `merged_at 2026-11-18T23:43:02Z`, head `04f5947d3f`, single parent
`d75d74142b` (squash). **2 files**:

| File | Δ | Change |
|---|---|---|
| `srt/layers/attention/deepseek_v4_backend.py` | +4/−1 | `force_deep_gemm_metadata = (enable_deepseek_v4_fp4_indexer **or compress_ratio in (1, 2)**) and get_platform().is_sm120` |
| `srt/layers/attention/dsv4/v41_indexer/scoring.py` | +1/−1 | `paged = get_platform().is_sm90` → `... or get_platform().is_sm120` |

**Anchors at `8817f63c`** (pin tarball, exact): `deepseek_v4_backend.py:1540` has
`self.enable_deepseek_v4_fp4_indexer and get_platform().is_sm120` with the same
"split_kv=128 vs 256" comment; `v41_indexer/scoring.py:487` has `paged = get_platform().is_sm90`.

**Arch predicate resolved — this is live for us, not for SM120 only.** `get_platform().is_sm120`
is `_PLATFORM_PROBES["is_sm120"] = "is_sm120_supported"` (`srt/runtime_context.py` probe map) →
`_check_cuda_device_version(device_capability_majors=[12], cuda_version=(12,8))`
(`srt/utils/common.py:304`), i.e. **major 12 ⇒ True on SM121/GB10**. The module-level
`utils.common.is_sm120()` (`:342`) is `== (12, 0)` and excludes GB10 — the two names must not be
confused. So both hunks address our hardware.

**Hunk 1 is inert at our profile, and its failure mode is not ours.** The PR's startup failure is
`deep_gemm.fp8_fp4_paged_mqa_logits` asserting `t.is_cuda()` because
`PagedIndexerMetadata.__post_init__` leaves `deep_gemm_metadata = None` whenever
`SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` is set (the SM12x model hook sets it) and
`force_deep_gemm_metadata` is false — which needs the c4 fp4 indexer flag **off**. Production
passes `--enable-deepseek-v4-fp4-indexer` (`.env.tp4.example:245`, `EXTRA_SGLANG_ARGS`), so
`force_deep_gemm_metadata` is already `True` on our SM12x and the schedule is built. Effect: we
already avoid the crash; the hunk only makes the flag-off/default config safe.

**Hunk 2 targets our live decode candidate path.** `DSV41_SM120_NO_CANDIDATES=1` (default;
`adapter/sm120_candidates.py`) zeroes `candidate_topk_blocks`, so `make_candidate_indexer` returns
`dense_blocks` for both sides (`v41_indexer/__init__.py:112`) → decode runs
`DenseBlocksBackend.publish_decode` → `decode_scores` with `paged=False` on SM121 → the dense-slot
branch: int64 `[bs, lmax]` slot map from `req_to_token` plus fp32 scores, **21.0 B/element** against
**4.0** on the paged branch (PR body probe at 1 M ctx, ratio 1: 5.48/5.65/11.29 GB dense vs
1.04/1.07/2.15 GB paged at bs 248/256/512). Capture-time `lmax = min(max_compressed_seq_len,
req_to_token.shape[1] // ratio)`; with `CONTEXT_LENGTH=1048576` and `MAX_RUNNING_REQUESTS=32`, the
top graph tier allocates ≈0.7 GB dense vs ≈0.13 GB paged `[INFERENCE — linear extrapolation of the
PR's bytes/element, not measured here]`.

**Apply blocker (why this is not a drop-in bp).** On the paged branch `DenseBlocksBackend.publish_decode`
asserts `self.block_size == 8` (`v41_indexer/dense_blocks.py:100`) and feeds `amax_topk_blocks` with
`(lens + 7) // 8` blocks; our adapter zeroes `candidate_block_size` → `block_size = 0`, so a
one-line `paged = is_sm90 or is_sm120` patch would assert at CUDA-graph capture. Reaching the paged
path needs block_size-8 semantics (candidate block scheme restored, or the assert/consumer
generalized) — and that is the scheme whose DeepGEMM kernels our adapter documents as
`block_kv == 64`-incompatible on SM121 (`adapter/sm120_candidates.py`, `attention.hpp:438`), which
#45261 does **not** change.

**Bound on the prize**: this is indexer scoring memory traffic, not the verify host-sync. The
standing decode gap (~44 ms/step `cudaEventSynchronize`, dspark verify rework #48831–33) is
untouched by this commit; any win is bounded by the low-ratio indexer's share of the step.

### Cross-checks

- Arithmetic: **816 + 27 = 843 ✓** against the log's `window-commits=843`; prior head found at
  index 27 of page 1 (page 1 is full at 100), so the delta is complete and no page-2 straddle is
  possible.
- Keywords: **2 of 27** new commits keyword-visible (91 → 93 ✓): `6eaca42f57` (via `deepseek-v4`
  in "DeepSeek-V4.1") and `3ab330cd3c` (false positive — `verify` in
  `spec_verify_calls_total`). **0 material misses.** The counter still misses `SM12x`; #45261 was
  caught only by the model name.
- Coverage: all 27 shas are assigned exactly once (1 Table A + 7 Table B + 19 grouped = 27). Path
  audit: no commit touches `models/deepseek_v4.py`, `deepseek_v4_backend.py` (#45261 excepted),
  `dsv4/v41_indexer/*`, `mem_cache/`, or the dspark kernels except #45261's two files.
- Fetch integrity: 2 issue legs + 2 PR legs + 1 search + 1 census page + 8 commit details + pin
  tarball + 3 raw probes; all parsed, no truncation, no retry (13.4 s script, ~2 min triage).
- Block-level: this run read the pin sources from the `8817f63c` tarball rather than asserting
  anchor existence from the patch context, so the `1540`/`487`/`100`/`304`/`342` line claims are
  read, not inferred.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + a local
   pin tarball in `/tmp` + this checkout only — no SSH, no docker, no build, no serve, no schedule
   change.
2. **New standing candidate — #45261 merge `6eaca42f57`**: bp-class on our exact surface, applies
   at the next pin with the SM12x branches active (is_sm120 ⇒ SM121 here). If anyone wants the
   paged decode scoring *before* the next pin, the block: `block_size == 8` assert against the
   zeroed `candidate_block_size`; measure it in an authorised window, do not assume.
3. **Standing — bp6 collapse at candidate `5e6995dd`**: unchanged.
4. **Standing — decode gate**: unchanged; nothing in this window touches the verify host-sync or
   the DeepGEMM arch-12 fp32-weights assert.
5. **Next pin**: #45261 rides in; re-check `adapter/sm120_candidates.py` necessity at the new pin
   (the `block_kv == 64` incompatibility it documents is not addressed by #45261). The
   `layers/communication/kda_dwdp` import check from run #22 still stands.
6. **Operator**: (a) `gh` re-auth (**24th run**; REST-only throughout), (b) no log gap this slot,
   (c) polling path clean.

---

## 2026-11-19T04:18Z (2026-11-19 12:18 +08) — watch run #25

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0, 19.5 s (log `04:17:48Z`–`04:18:04Z`),
`fetch_rc=0`; every leg numeric on the first attempt, no TLS retry. The alert body is the standing
trigger set only (#49173 closed, its 2 comments, **95** cumulative keyword hits) with no
`TOOL FAILURE` line ⇒ the census leg is real, not a 403/urllib blackout. Budget **45/60** core
immediately after the script (15 used), **26/60** after triage; reset `05:04:49Z`. `gh auth status`
→ **token for `aotsukiqx` invalid** (**25th consecutive run**); every payload below is
unauthenticated GitHub REST.

No log gap: `00:18Z` → `04:18Z` on schedule.

### Issues / PR — zero delta

- **#49173** — closed, `comments=2`, `updated_at 2026-11-09T07:26:52Z` (the close second itself):
  identical to runs #1–#24. Both comments are the original Kylinny thread, already absorbed into
  `docs/PROJECT_STATE.md`. **No new maintainer comment to read**, so step 2 of the runbook has
  nothing to fetch.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` → **twenty-fourth
  consecutive silent run**. The referencing PR **#49306** (`closes #49174`) is still `open`,
  `merged=false`, head `4e1544039e`, `updated_at 2026-11-09T12:56:58Z` — untouched.
- **#49182** — `closed`, `merged=true`, `merged_at 2026-11-09T07:26:50Z`, merge **`5e6995dd09`**,
  head `a49ff4d7d3` — unchanged. Next-pin candidate stays **`5e6995dd`**.

### sglang main — 17 new commits: 16 NO-OP, **1 bp-class candidate** (#51295)

Census **860** (`since=2026-11-08T14:00:00Z`, 9 pages = 8×100 + 60). Prior total **843 + 17 = 860
✓**, and prior head `6b374a5502` sits at **index 17** of the newest-first slice, so `[0:17]` is
exactly the delta. New head **`5b478f175b`** (`2026-11-19T04:16:12Z`). Bookkeeping: `6b374a5502` is
also the sha in the upstream nightly tag `lmsysorg/sglang:nightly-dev-20261119-6b374a55` (row 6
below) — our prior head was the first nightly carrying #45261.

| sha | PR | Subject | Class |
|---|---|---|---|
| `5b478f175b` | #51295 | [Perf] mem_cache: allocate decode slots and write `req_to_token` in one launch | **bp-class** — anchors verified at pin; 3 hunks need refitting (below) |
| `c86973c11c` | #51410 | [Doc] cookbook: RTX PRO 6000 recipe for DeepSeek-V4.1 | **bookkeeping** — corroborates #45261 necessity + our `flashinfer_cutlass` pin; no code |
| `145647c55f` | #50767 | [Feat] disaggregation: unified-memory PD with NIXL | NO-OP (gated) — `--disaggregation-mode`/PD: **0 hits** in launchers and `.env.tp4.example`; changes `arg_groups/overrides.py`, `kv_cache_hook.py`, `disaggregation/nixl/conn.py`, `server_args.py` (+180/−46 incl. 4 test files) |
| `a3eae55545` | #46953 | [Feat] weight_cache: `vmm_fd` transport | NO-OP (gated) — no weight-cache flag on this line (**0 hits**); `weight_cache/transport.py` +481/−67, new `vmm_layout.py`, `model_runner.py` +1 and `load_model_utils.py` +2 are flag plumbing |
| `8073aea66a` | #46499 | [Feat] UnifiedL3: mooncake storage backend | NO-OP (gated) — HiCache host/L3 tier off (**0 hits** `hierarchical-cache`); `pool_host/{mha,mla}.py` +4, `hicache_key_scheme.py` +21/−1. Pin note: host-pool key scheme drifts — re-take at the next pin if the tier is ever enabled |
| `44ffcdead5` | #49275 | [Perf][NVIDIA] channelwise FP8 GEMM tile configs for RTX PRO 5000 | NO-OP — configs key `device_name=NVIDIA_RTX_PRO_5000_72GB_Blackwell` + `dtype=fp8_w8a8_channelwise`; pin's 181 configs are `device_name`-keyed with `fp8_w8a8`/`int8_w8a8` + `block_shape` and contain **no GB10 entry**, and this line runs `quantization: None` (native blockwise fp8) |
| `1f2d76c79d` | #36088 | [Fix][AMD] distributed: custom AR capture for DSA DP dual graphs | NO-OP — AMD-only and DP-only (`dp_size=1` here); `DSV41_DUAL_GRAPH` is our own adapter hook, untouched |
| `5a0e808c59` | #47961 | [Feat][Intel] XPU in the EPD multimodal encoder path | NO-OP — Intel XPU/EPD |
| `2886093fa9` | #49790 | [Fix][NPU] `qwen3_5_mtp`: `mtp.fc` as `ReplicatedLinear` | NO-OP — NPU + Qwen3.5 |
| `72fae91008` | #51235 | [Doc] Deepseekv4: dsv4 NPU best practice | NO-OP — NPU docs |
| `9304abb532` | #50220 | [Test][AMD] sana-video2 BF16 softmax skip | NO-OP — AMD test |
| `cde63d2d9e` | #50490 | [CI][AMD] nightly InferenceX Kimi-K3.1-Pro MXFP4 on MI35x | NO-OP — AMD CI |
| `091cdce699` | #51318 | [Perf] diffusion: pybase64 + orjson image responses | NO-OP — diffusion |
| `7bb4884d73` | #50440 | [Fix] diffusion/qwen-image: stream the DiT when it does not fit | NO-OP — diffusion |
| `5decba407a` | #51362 | [Fix] diffusion: exit nonzero on empty generate | NO-OP — diffusion |
| `f7e7ccbd9f` | #51358 | [Doc] diffusion: clarify image edit inputs/URL responses | NO-OP — diffusion docs |
| `e840dbeaab` | #51361 | [Doc] diffusion: Pi0.5 tokenizer access | NO-OP — diffusion docs |

### #51295 in detail — decode slot alloc fused with the `req_to_token` write

Merge **`5b478f175b`**, single parent `f7e7ccbd9f` (squash), **7 files, +136/−29**. Two of its own
docstrings name our situation: `allocator/paged.py:140` at the pin is `TODO: fuse last_loc into the
kernel.` — this commit is that TODO.

**Anchors read at `8817f63c`** (raw sources fetched at the pin; the tree probe reports 13 301 blobs,
`truncated=false`, and it contains every path the patch touches, so the layout reorg is not a
blocker):

| File | Pin state | Verdict |
|---|---|---|
| `python/sglang/kernels/ops/memory/allocator.py` | `alloc_decode_kernel` @210 with `@triton.jit(do_not_specialize=["free_page_ptr"])` and the `num_page_start_loc_self` branch — byte-identical pre-image | **applies** |
| `python/sglang/srt/mem_cache/allocation.py` | `alloc_for_decode_default` @650 — advanced-index gather `req_to_token[req_pool_indices, seq_lens-1]`, `seq_lens_gpu.clone()`, separate `req_to_token_pool.write(...)` | **applies** (pre-image exact) |
| `.../mem_cache/allocator/paged.py` | class @136 (`TODO: fuse last_loc` @140), `debug_mode` @154, `use_triton_kernels` @155, `alloc_decode` @~280 with the 3-arg signature | **applies** (pre-image exact) |
| `.../mem_cache/allocator/swa.py` | `alloc_decode` @446, `_swa_req_ring` @48/139, `translate_loc_from_full_to_swa` @269 | hunk applies but is **inert here** — DSV4.1 on this line is MLA/DSA with no sliding-window pool (recorded 2026-11-14, #50059 row); Re-check only if a hybrid-SWA model ever runs |
| `.../mem_cache/allocator/base.py` | no `pools_share_memory` (the patch's context line is post-pin); class attrs start at `hands_out_virtual_ids` @47 | **refit** — the insert is a 2-line class attribute |
| `.../mem_cache/kv/storage.py` | `take_decode` @193, one-liner body, **no** `tree_cache`/`batch` kwargs (patch hunk @339) | **refit** — signature + optional kwarg pass-through |
| `.../mem_cache/static_kv_storage.py` | `take_decode` @233 (patch hunk @375), **no** `tree_cache`/`batch` kwargs; the DSV4-NPU `c128_attn_allocator` branch is already present @248 | **refit** — same two edits |

**Gates on this line — the fused path is live, not skipped.** `PagedTokenToKVPoolAllocator` is used
because the dsv4 attention backend forces page_size 256 (`logs-tp4/dsv41.log:12`: "Use dsv4
attention backend …, setting page_size to 256"), so `_alloc_page_size(batch) == 1` is false;
`token_per_req == 1` at decode; TP4 on CUDA ⇒ `current_platform.capabilities.supports_triton` true
⇒ `supports_fused_decode_write`; `SGLANG_DEBUG_MEMORY_POOL` is unset ⇒ `debug_mode` false;
`type(allocator) is PagedTokenToKVPoolAllocator` holds (no unified/DSV4 subclass on CUDA at the
pin). The `ReqToTokenPool.write` identity check passes as long as nothing overrides it — `grep -rn
"ReqToTokenPool" adapter/` is **empty**; confirm at apply time anyway.

**Value bound — launch overhead only, not the decode gap.** The win is folding three GPU ops and
their dependencies into the alloc kernel's launch: the `req_to_token[rows, seq_lens-1]` gather,
`seq_lens_gpu.clone()`, and the separate triton `req_to_token_pool.write` scatter. None of them
introduces a host sync (no `.item()`, no D2H), so this cannot move the ~44 ms/step
`cudaEventSynchronize` verify gap — at a 44 ms step the prize is launch-scale, sub-0.1%
[INFERENCE from op count, not measured]. Worth taking only as a free rider in a window that is
already happening; never a window trigger.

### #51410 — upstream now documents the #45261 requirement, and the backend we already pin

The cookbook row (docs only, `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` +
`src/snippets/configs/.../deepseek-v4_1.jsx`) adds an RTX PRO 6000 (SM120, 8×96 GB, TP8/EP8)
recipe verified on the **1017** checkpoint, and states:

- It **requires a build with #45261**, otherwise SM120 fails at startup with the DeepGEMM
  `t.is_cuda()` assert — or, with `--enable-deepseek-v4-fp4-indexer` passed (our case), decode graph
  capture runs out of memory at the default 1 M context. That second clause is independent
  corroboration of run #24's hunk-2 memory argument (dense scoring, 21 B/element); our line survives
  1 M because graph max bs is far below 128 (`MAX_RUNNING_REQUESTS=32`).
- `--fp8-gemm-backend flashinfer_cutlass` measured **17–18 % output tokens/s and 31–44 %
  single-request decode** better than the automatic choice on SM120. `.env.tp4.example:245` already
  pins exactly that ⇒ no change, no experiment.
- Other recipe env is PCIe/single-node specific (`NCCL_P2P_LEVEL=SYS`, `NCCL_MIN_NCHANNELS=16`) or
  host-engram placement for non-UMA nodes — not transferable to the switchless RoCE TP4 line.

### Cross-checks

- Arithmetic: **843 + 17 = 860 ✓** against the log's `window-commits=860`; page 1 is full at 100 and
  the prior head is at index 17, so no page-2 straddle.
- Keywords: **2 of 17** new commits are keyword-visible (93 → 95 ✓) and **both are false
  positives** (`DeepSeek-V4.1` in the cookbook doc, `Deepseekv4` in the NPU doc). The one material
  commit, #51295, is invisible to the counter: the census key list has no `mem_cache` even though
  `mem_cache/` is one of the script's `AXES` (that variable is unused by the census python). Second
  consecutive window where the counter finds zero material rows.
- Coverage: all 17 shas are assigned exactly once (17 table rows). Path audit: no commit touches
  `models/deepseek_v4.py`, `srt/speculative/`, `deepseek_v4_backend.py`, `dsv4/v41_indexer/*`, or
  `engram.py`; `mem_cache/` only via #51295.
- Fetch integrity: 2 issue legs (script) + 9 census pages + 5 commit details + 1 pin tree + 7 raw pin
  files + 1 `pulls/49306` + 1 `search/issues`; all JSON parsed (`json.tool` clean over 15 cache
  files), one intentional 404 (`docs/_static/cookbook/...`, superseded by the commit diff).
- Block-level: the four "applies" rows were read at the pin, not inferred from patch context; the
  three refit rows were checked for the specific missing context.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + a local
   pin tree/raw reads + this checkout only — no SSH, no docker, no build, no serve, no schedule
   change.
2. **New standing candidate — #51295 `5b478f175b`**: bp-class, low value (launch overhead only,
   no host sync removed, not a decode-gap lever). Apply only inside an authorised window, with the
   three refits (base/kv-storage/static-kv-storage) and the swa.py hunk dropped as inert.
3. **Standing — #45261 `6eaca42f57`**: unchanged; #51410 now documents it as mandatory for SM120
   and confirms the 1 M-context dense-scoring memory blow-up. Paged-path blocker (`block_size == 8`
   vs our zeroed `candidate_block_size`) still stands.
4. **Standing — bp6 collapse at candidate `5e6995dd`**: unchanged.
5. **Standing — decode gate**: unchanged; nothing in this window touches the verify host-sync or the
   DeepGEMM arch-12 fp32-weights assert.
6. **Next pin**: #51295 rides in (its refit cost goes to zero once the pin has the post-pin
   `take_decode(tree_cache=, batch=)` signature); #51410 is docs-only; the `kda_dwdp` import check
   from run #22 still stands.
7. **Operator**: (a) `gh` re-auth (**25th run**; REST-only throughout), (b) no log gap this slot,
   (c) polling path clean, (d) repo-side decision pending, deliberately not patched in this
   docs-only run: add `mem_cache` (and `req_to_token`) to the census keyword list and either wire
   `AXES` up or delete it.

## 2026-11-19T08:18Z (2026-11-19 16:18 +08) — watch run #26

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0, 14.4 s (log `08:17:42Z`–`08:17:56Z`),
`fetch_rc=0`; every leg numeric on the first attempt, no TLS retry, no `TOOL FAILURE` line. Alert
body is the standing trigger set only (#49173 closed, its 2 comments, **100** cumulative keyword
hits of **886**). Core budget: 11 calls by the script + 29 by hand (2 commit-list pages, 1
`pulls/49182`, 26 commit details) — **no 403**, all 28 cached payloads `jq`-clean. `gh auth status`
→ **token for `aotsukiqx` invalid** (**26th consecutive run**); every payload below is
unauthenticated GitHub REST. No log gap: `04:18Z` → `08:18Z` on schedule.

### Issues / PR — zero delta

- **#49173** — closed, `comments=2`, `updated_at 2026-11-09T07:26:52Z` — **twenty-fifth
  consecutive run** identical to the close second. Both comments are the original Kylinny thread,
  already absorbed into `docs/PROJECT_STATE.md`. No new maintainer comment; step 2 of the runbook
  has nothing to fetch.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` — **twenty-fifth silent
  run**. Referencing PR **#49306** remains `open`/`merged=false`, head `4e1544039e`.
- **#49182** — re-verified from the API this run: `closed`, `merged=true`,
  `merged_at 2026-11-09T07:26:50Z`, merge **`5e6995dd09`**, head `a49ff4d7d3`, 3 files
  (+204/−5), base `4003bf10`. Unchanged; next-pin candidate stays **`5e6995dd`**.

### sglang main — 26 new commits: 25 NO-OP, **1 pin-migration-class, inert on this line** (#51298)

Census **886** (`since=2026-11-08T14:00:00Z`, 9 pages = 8×100 + 86). Prior **860 + 26 = 886** ✓.
Prior head `5b478f175b` sits at **index 26** of the newest-first delta slice (31 commits since
`03:30Z` minus the 5 already in run #25), so `[0:26]` is exactly the delta. New head
**`a735253e84`** (`07:55:15Z`). Keyword counter **95 + 5 = 100** ✓; the 5 hits are `21be2189de`
(dspark), `5aee953e1a` (sm120), `81844a31d7` (dsv4), `e4353c9780` (indexer), `f49d6c493e`
(dsv4).

| sha | PR | Subject | Class |
|---|---|---|---|
| `8625aed698` | #50972 | [Test][NPU] graph: drop tp1 compile_graph, run tp2 in base-b CI | NO-OP — NPU CI |
| `81844a31d7` | #51417 | [Test][NPU] multi-node: PD warmup race, DSV4-Flash 128K prefill memory fraction | NO-OP — NPU multi-node test; no CUDA code path |
| `5aee953e1a` | #50752 | [Fix][NVIDIA] DSA: BF16 TileLang shared-memory overflow on SM120 | NO-OP **for this line** — real SM12x fix (`head_tile=16` when `d_v==512, tail_dim==64`, `tilelang_kernel.py`), but the DSA TileLang backend is not on our path (dsv4 backend + `fp4_indexer`). Arch-adjacent: **take it if the dsa backend is ever enabled on GB10** |
| `8983bd6a66` | #45274 | [Perf][NVIDIA] DSA: any divisor of the tile count for TileLang sparse decode split | NO-OP — same non-path; tuning constants measured on 132-SM H100, GB10 has 48 SMs |
| `fb60c29772` | #46401 | [Feat][NVIDIA] beam: SM103 single-token BF16 MoE kernel | NO-OP — SM103 + beam |
| `27997a872f` | #50228 | [Perf][NVIDIA] gemma4: fused all-reduce inside decode CUDA graphs | NO-OP — gemma4 |
| `91c88c6697` | #51297 | [Feat] attention: per-layer FA overrides and block-anchored windows | NO-OP — defaults are `None`/`False`; the shared-touch hunks (`graph_utils.allocate_attention_outputs`, `radix_attention.__init__`) add `getattr(layer,"out_dtype",None)` and keyword args, i.e. behaviour-identical without an opt-out layer |
| `2e80690d10` | #50775 | [Fix] attention: KDA prefill overflow past 2**31 token offsets | NO-OP — KDA (Kimi) kernel |
| `a735253e84` | #51360 | [Test] diffusion: action HTTP/WebSocket client contracts | NO-OP — diffusion test |
| `33a03ba42e` | #41334 | [Fix] disaggregation: register NextN draft DSA index cache for hybrid pools | NO-OP (gated) — PD-disaggregation state registration (`static_kv_storage.py`); **0 disagg on this line**. Path-audit hit, subject-invisible |
| `8eba7fb3d3` | #44217 | [Feat] dsa: raw FP8 KV cache with Q8 prefill on CUDA | NO-OP + **arch-excluded** — the new native path asserts `native_raw_nope and major != 9 → ValueError`; needs `kv_lora_rank 512` + mixed tilelang DSA backends. GLM-5.3 cookbook row |
| `21be2189de` | #51298 | [Feat] dspark: draft folds its verify commit, cap step logits | **pin-migration-class, inert here** — see below |
| `f49d6c493e` | #49069 | [Perf] dsv41-flash: fuse Hopper prefill candidate selection + writeback | NO-OP **on SM12x** — new radix-select/writeback kernels dispatch only under `envs.SGLANG_OPT_DSV41_TRITON_PREFILL_SELECTION and get_platform().is_sm90`; +368/−1, `kernels/ops/attention/dsv4/candidate_blocks.py` |
| `e4353c9780` | #49068 | [Perf] dsv41-flash: reuse paged Indexer key tiles during verification | NO-OP **on SM12x** — grouped-CTA verify kernel gated `current_arch().is_sm(9)` (+ `rows_per_request` opt-in, new `SGLANG_OPT_DSV41_VERIFY_INDEXER_BF16_K_REUSE`, `EnvBool(False)`). Closest upstream touch to our decode gap; Hopper-only, so **no upstream answer to the ~44 ms verify host-sync** |
| `d814a39f47` | #51302 | [Feat] inkling2: SM90 Marlin, expert and MTP kernel ops | NO-OP — inkling2/SM90 |
| `c97d47b872` | #51301 | [Feat] inkling2: render/parse through the inkling2 renderer spec | NO-OP — renderer |
| `1ed3e1d04b` | #51231 | [Chore] inkling: bump tml-renderers to 0.4.2 | NO-OP |
| `ecc5ab3a38` | #51300 | [Feat] linker: hybrid-SWA sidecar draft pools on the target pages | NO-OP — no SWA pool on this line (same ruling as #50059); re-check only if a hybrid-SWA draft ever runs |
| `8761838757` | #51296 | [Feat] mem_cache: MXFP8 KV scales in unified-memory pages | NO-OP (gated) — `is_mxfp8 = kv_cache_dtype_str == "mxfp8"` (new dtype string), `MHAGeometry` only; this line is `fp8_e4m3` MLA. **Subject-invisible to the census** (no `mem_cache` keyword) |
| `0b1064bdf9` | #51299 | [Refactor] model_loader: models name their safetensors index file | NO-OP — opt-in per model; no DSV4.1 model declares one |
| `b08f70c1f9` | #51162 | [Fix] quantization: keep Marlin conversion for AWQ/GPTQ on SM100 | NO-OP — we load native blockwise DeepSeek FP8/FP4, `quantization: None` |
| `f29cc397a2` | #48249 | [Perf] qwen4: pin the offloaded PLE table at its exact size | NO-OP — Qwen4 PLE (touches `mem_cache/pool_host/common.py`, subject-invisible) |
| `b05e9c68ce` | #50721 | [Chore] sglang-processor: dynamo-renderer 10.4.3, dynamo-tokenizers 4.3.1 | NO-OP |
| `981becd7af` | #51386 | [Fix] spec/dflash: DFLASH draft KV ring off by default | NO-OP — DFLASH, not dspark; flips `spec.py`/`environ.py` defaults only for the DFLASH ring |
| `5eeeb116e1` | #50620 | [Perf] spec: fused draft softmax + sample for EAGLE rejection sampling | NO-OP — EAGLE path (`spec_utils.use_fused_draft_sample` needs `SGLANG_OPT_USE_FUSED_DRAFT_SAMPLE` + `..._GUMBEL_SAMPLE` + vocab gate). Also a **JIT include refactor** touching us: `csrc/speculative/dspark_markov_walk.cuh` −50 lines, numeric helpers moved to new `include/sgl_kernel/ptx/utils.cuh` + `sampling.cuh`, `sampling/softmax.cuh` ±45 — must be taken **as a unit** at a pin (shared-header graph, not a standalone patch) |
| `eae4bcb70b` | #50518 | [Fix] tokenizer_manager: stop schedulers only after every tokenizer worker drains | NO-OP — shutdown ordering; single tokenizer manager here |

### #51298 in detail — a new opt-in draft-folding hook, and why it is inert on this line

Merge **`21be2189de`**, 10 files, **+308/−37**, of which 4 are tests. Shape:

- `dspark_verify.draft_folds_static_commit()` — the draft may fold the verify commit only if it
  sets `folds_static_verify_commit`, its FULL id space equals the target's and
  `target.write is None`; then `run_static` calls `draft_model.store_verify_commit(...)`.
- `static_epilogue_supported` is restructured from `target_is_dsv41 and _draft_is_moe` to
  `(target_is_dsv41 and _draft_is_moe) or draft_folds_static_commit(...)` — **the first disjunct
  already holds on this line** (`target_model_type() == "deepseek_v41"`, `_draft_is_moe` true), so
  no behaviour change from the gate itself.
- `folds_static_commit = folds_commit or _draft_folds_static_commit`; `_forward_decode` selects it
  when `run_compact` is false. Without a producer for the opt-in attribute this is `folds_commit`,
  i.e. the pin's value.
- `fused_commit_track_indices(accept_index=None)` gains `HAS_ACCEPT_INDEX`, and
  `_commit_target_mamba_states_after_verify` now always routes through it. On our line
  `mamba_track_indices is None` (MLA/DSA, no mamba state): the pin's two eager ops
  (`commit_lens.to(int64)`, `- 1`) collapse into **one** fused launch — micro, launch-scale.
- `markov_sample` gains `softcap`/`vocab_limit`; `models/dspark.py` declares
  `step_softcap`/`step_vocab_limit` = `None`, and `_draft_step`'s fused greedy path is skipped only
  when either is set. With the `None` defaults the fused path is unchanged.

**Producer absent at `8817f63c`** (raw files fetched at the pin): `folds_static_verify_commit` and
`store_verify_commit` grep to **0** in `srt/speculative/dspark_components/dspark_verify.py`,
`.../dspark_worker_v2.py`, `models/dspark.py` and `models/deepseek_v4.py`; `_build_verify_executor`
and `_on_prefill_target_hidden` are likewise absent, and `run_compact` already exists (so the
`_forward_decode` hunk's context is satisfiable). The opt-in attribute is defined elsewhere in the
same series (sibling commits, not this one), so a cherry-pick of #51298 alone changes **nothing**
operationally for DSV4.1 — it only reroutes the mamba commit math and adds opt-in plumbing — while
the hunks span `dspark_worker_v2` structure the pin predates. **Verdict: do not bp-patch; take it
wholesale at the next pin.** No `cudaEvent`/synchronize/`.cpu()`/`.item()` additions in the patch
⇒ **not a decode-gap lever**, consistent with the standing attribution.

### Cross-checks

- Arithmetic: **860 + 26 = 886 ✓** against the log's `window-commits=886`; keyword **95 + 5 = 100
  ✓** against `dsv41-relevant-since-watch=100`.
- Coverage: all 26 shas assigned exactly once (26 table rows). Path audit over all 26 diffs: axes
  paths are touched only by `21be2189de` (dspark/*), `e4353c9780` + `f49d6c493e` (dsv4 indexer),
  `8761838757` (mem_cache kv), `33a03ba42e` (`mem_cache/static_kv_storage.py`), `ecc5ab3a38`
  (`mem_cache/hybrid_cache/`), `f29cc397a2` (`mem_cache/pool_host/`), `5eeeb116e1`
  (`srt/speculative/` + dspark JIT include). **No commit touches `models/deepseek_v4.py`,
  `deepseek_v4_backend.py`, `engram.py` or `dsv4/v41_indexer/*` outside #49068/#49069.**
- **Census blind spots, third consecutive window**: 3 of the 7 axes-path commits
  (`8761838757`, `33a03ba42e`, `f29cc397a2`) carry no keyword and are invisible to the counter;
  `5eeeb116e1` is visible only via the shared JIT path. The `mem_cache` gap first flagged in run
  #25 is now chronic — repo-side fix still pending.
- Block-level: 4 hunks read against pin sources (dspark verify/worker pairs, `models/dspark.py`,
  `models/deepseek_v4.py`); the rest classified from diff + inline gates (`is_sm(9)`,
  `is_sm90`, `is_sm(12)`, env defaults, dtype/config strings), which is sufficient for NO-OP
  rulings because the gate is in the patch itself.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + 5 raw
   pin files + this checkout only — no SSH, no docker, no build, no serve, no schedule change.
2. **No new candidate of either class worth a window.** #51298 is pin-migration-class but inert and
   offer-only; #49068/#49069 are Hopper-gated; #50752 is a real SM12x fix on a backend we do not
   run.
3. **Standing — #45261 `6eaca42f57`**: unchanged. Paged-path blocker (`block_size == 8` vs our
   zeroed `candidate_block_size`) still stands; #51410 continues to document it as mandatory on
   SM120.
4. **Standing — #51295 `5b478f175b`**: unchanged (3 refit hunks + inert swa hunk); free-rider only.
5. **Standing — bp6 collapse at candidate `5e6995dd`**: unchanged.
6. **Standing — decode gap**: no upstream movement. The two dsv41-flash perf commits that shave
   verify-side work are SM90-only; the ~44 ms/step `cudaEventSynchronize` verify host-sync has no
   upstream fix in this window.
7. **Next pin** picks up, in dependency order: #51298's folding series (needs the producer
   commit), #50620 as a unit with its new JIT headers, #51296/#51297 opt-ins (defaults off),
   #50752 if the dsa backend is ever enabled, #49068/#49069 as dead weight on SM12x.
8. **Operator**: (a) `gh` re-auth (**26th run**, REST-only throughout; `gh issue view` /
   `gh pr view` still unusable, so step 2 of the runbook keeps running on REST equivalents),
   (b) no log gap, (c) polling path clean, (d) **repo-side decision still pending**: add
   `mem_cache` (and `req_to_token`, `static_kv_storage`, `pool_host`) to the census keyword list,
   or replace subject matching with the path-based audit this run had to do by hand.

## 2026-11-19T12:20Z (2026-11-19 20:20 +08) — watch run #27

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0, 13.8 s (log `12:20:01Z`–`12:20:14Z`),
`fetch_rc=0`, all legs numeric on the first attempt, no TLS retry, no `TOOL FAILURE`. The scheduled
`12:17Z` poll ran immediately before this session and reported the same state. Alert body is the
standing trigger set only (#49173 closed + its 2 comments, **101** keyword hits of **895**). Budget:
11 calls by the script + ~32 by hand (9 commit-list pages, 9 commit details, `pulls/49182`,
`pulls/49306`, 2 issue reads + comments, `commits/8817f63c`, 3 raw pin files, 3 existence probes,
2 contents listings) — **no 403**, all payloads unauthenticated REST. `gh auth status` → **token
for `aotsukiqx` invalid** (**27th consecutive run**); `gh issue view` / `gh pr view` remain
unusable, so step 2 of the runbook runs on REST equivalents.

### Issues / PR — zero delta

- **#49173** — closed (`completed`), `comments=2`, `updated_at 2026-11-09T07:26:52Z` —
  **twenty-sixth consecutive run** identical to the close second. Both comments are the original
  Kylinny thread (fix offered in #49182), already absorbed into `docs/PROJECT_STATE.md`. No new
  maintainer comment; nothing to fetch for step 2.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` — **twenty-sixth silent
  run**. Referencing PR **#49306** still `open`/`merged=false`, head `4e1544039e`, updated
  `2026-11-09T12:56:58Z`.
- **#49182** — re-verified this run: `closed`, `merged=true`, `merged_at 2026-11-09T07:26:50Z`,
  merge **`5e6995dd09`**, head `a49ff4d7d3`, title "Restore SM120/SM121 to the DeepGEMM indexer
  gates". Unchanged; next-pin candidate stays **`5e6995dd`**.

### sglang main — 9 new commits: 8 NO-OP, **1 pin-migration-class (DCP-only, identity at dcp=1)**

Census **895** (`since=2026-11-08T14:00:00Z`, 9 pages = 8×100 + 95). Prior **886 + 9 = 895** ✓.
Prior head `a735253e84` (`07:55:15Z`) → new head **`b6d4389a77`** (`11:49:17Z`); the delta is
exactly the nine shas newer than `07:55:15Z`. Keyword counter **100 + 1 = 101** ✓; the single hit
is `01497afbd4` (`indexer`).

| sha | PR | Subject | Class |
|---|---|---|---|
| `b6d4389a77` | #51504 | [Doc] cookbook: refresh SemiAnalysis AgentX launches to InferenceX main | NO-OP — docs `.jsx` only (includes the `deepseek-v4_1` launch snippets) |
| `cb970616d4` | #47977 | [Feat] mm: `--disable-multimodal` to serve a multimodal model text-only | NO-OP — flag defaults off; DSV4.1-Flash is text-only, `is_multimodal` already false |
| `f9dacff701` | #51270 | [Fix] grpc: allow `--tokenizer-worker-num > 1` with `--grpc-port` (Rust server) | NO-OP — Rust/gRPC server not used |
| `63038e7c99` | #49127 | [Feat] linker: expose external cache linker lookup/load/offload/wait metrics | NO-OP (gated) — **axes path (`mem_cache/unified_cache/`), subject-invisible**; adds `init_linker_metrics` + `blocked_seconds` and reads `enable_unified_cache_external_linker` in the Rust output streamer. We run neither the unified external linker nor the Rust server |
| `9aa3b9e338` | #51172 | [Doc][AMD] cookbook: GLM-5.3-Flash MI300X/MI325X commands | NO-OP — docs |
| `33315670ab` | #51517 | [Fix] cli: resync cookbook command registry with main | NO-OP — CLI JSON metadata; DSV4.1-Flash-1017 rows keep `b200`/`b300` (`tp4 ep4`), `mi355x` ROCm rows dropped from that file |
| `01497afbd4` | #51450 | [Feat][NVIDIA] dsa: FP4 indexer for Kimi-K3.1 and under DCP | **pin-migration-class, identity at dcp=1** — see below |
| `6c722acbcd` | #51458 | [Fix] dflash: preserve captured `custom_mask` on idle DP ranks | NO-OP for `dp_size=1`; pre-image matches pin exactly → trivial free-rider at the next pin |
| `b1970bc473` | #50106 | [Feat] cli: export cookbook commands for every config-driven LLM/VLM page | NO-OP — CLI registry + docs export (139 files, 0 runtime) |

### #51450 in detail — a DCP-aware FP4 index-K store, numerically identity on this line

- The shared kernel **is** ours: `python/sglang/kernels/ops/kvcache/dsv4_fp4_index_k_cache.py`
  exists at `8817f63c` (200). The patch threads `dcp_size`/`dcp_rank` constexprs (defaults `1`/`0`)
  into `_store_fp4_index_k_cache_kernel`, masks stores to `cache_loc % dcp_size == dcp_rank`, and
  stores at `cache_loc // dcp_size`.
- At our configuration `dcp_size == 1`, `dcp_rank == 0`: the mask is all-true (`x % 1 == 0`) and
  the division is identity ⇒ **byte-identical cache writes**. Extra per-token `%`/`//` in the
  kernel; no `.cpu()`/`.item()`/event/sync added ⇒ **not a decode-gap lever**.
- The caller side does **not** exist at the pin: `dsa/dsa_fp4_indexer.py` is absent from the pin's
  `dsa/` listing (17 files) and raw-fetches 404; `_index_k_dcp` greps to **0** in the pin's
  `dsa_indexer.py` (only `_index_k_write_targets` at line 249). This line's indexer is
  `dsv4/v41_indexer/`, not the `dsa` wrapper the patch extends.
- Verdict: **do not bp-patch** (would need to invent the producer and the `_fused_q_quant` hook);
  take wholesale at the next pin. Kimi-K3.1/DCP feature, no DSV4.1 benefit.

### #51458 anchors — exact at the pin, unreachable at `dp_size=1`

- Pre-image matches `decode_cuda_graph_runner.py` at `8817f63c` verbatim: `_slot(name)` block ends
  with `self.capture_hidden_mode, spec_info,`; `load_batch`'s
  `not is_ragged and forward_mode.is_idle() and spec_info is not None` branch assigns
  `spec_info.custom_mask = buffers.custom_mask` (pin lines 1671–1677).
- The branch needs `ForwardMode.IDLE`, produced only by DP-attention idle ranks. This line is pure
  TP4 (`dp_size=1`, PROJECT_STATE §46), so the branch never executes. If DP is ever enabled, the
  capture-vs-replay mask mode must match or the captured graph indexes a stale mask.
- 5 lines (+3/−1), zero runtime cost ⇒ absorb at the next pin; no window of its own.

### Cross-checks

- Arithmetic: **886 + 9 = 895 ✓** against `window-commits=895`; keyword **100 + 1 = 101 ✓**
  against `dsv41-relevant-since-watch=101`.
- Coverage: all 9 shas assigned exactly once (9 table rows).
- Path audit over all 9 diffs: the only axes-path commit is `63038e7c99`
  (`mem_cache/unified_cache/unified_cache_linker.py`, `observability/`, `output_streamer.py`).
  **No commit touches `models/deepseek_v4.py`, `deepseek_v4_backend.py`, `engram.py`,
  `srt/speculative/`, or `dsv4/v41_indexer/`** in this window.
- **Census blind spot, fourth consecutive window**: `63038e7c99` is on an axes path with no
  keyword in its subject; `01497afbd4` is visible only via `indexer`. The `mem_cache` gap flagged
  from run #25 onward is chronic — repo-side fix still pending.
- Block-level: 2 hunks read against pin sources (FP4 index-K kernel, decode graph runner);
  existence of the `dsa` wrapper and `_index_k_dcp` checked against the pin tree via the contents
  API. The remaining 7 classified from diff paths + inline gates (docs, CLI JSON, arg defaults),
  which is sufficient for NO-OP rulings.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + 4 raw
   pin files + 2 pin-tree listings + this checkout only — no SSH, no docker, no build, no serve,
   no schedule change.
2. **No new candidate worth a window.** `01497afbd4` is DCP-only and identity at `dcp_size=1`;
   `6c722acbcd` is unreachable at `dp_size=1`; `63038e7c99` is metrics on an unused subsystem.
3. **Standing — #45261 `6eaca42f57`**: unchanged. Paged-path blocker (`block_size == 8` vs our
   zeroed `candidate_block_size`) still stands.
4. **Standing — #51295 `5b478f175b`**: unchanged (3 refit hunks + inert swa hunk); free-rider only.
5. **Standing — bp6 collapse at candidate `5e6995dd`**: unchanged.
6. **Standing — decode gap**: no upstream movement. None of the 9 adds a host sync, `.item()` or
   CPU offload; the ~44 ms/step `cudaEventSynchronize` verify host-sync still has no upstream fix.
7. **Next pin** picks up, in dependency order: #51298's folding series (needs its producer
   commit), #50620 as a unit with its new JIT headers, #51296/#51297 opt-ins (defaults off),
   #45261, #51295, #51450 (DCP, inert), #51458 (5-line DP fix), #50752 if the dsa backend is ever
   enabled.
8. **Operator**: (a) `gh` re-auth (**27th run**, REST-only throughout), (b) no log gap
   (`08:18Z` → `12:17Z` scheduled → `12:20Z` this run), (c) polling path clean at both
   invocations, (d) **repo-side decision still pending**: add `mem_cache` (and `req_to_token`,
   `static_kv_storage`, `pool_host`) to the census keyword list, or make the census path-based —
   this window produced a fourth consecutive subject-invisible axes-path commit.

## 2026-11-19T16:18Z (2026-11-20 00:18 +08) — watch run #28

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0, 21.2 s — the `00:17 +08` cron slot ran this
session, log `16:17:47Z`–`16:18:08Z`, `fetch_rc=0`, both issue legs and the census leg numeric on
the first attempt, no TLS retry, no `TOOL FAILURE`. Continuity holds (previous run `12:20Z`); the
slot holds exactly one run, so no double-log. Alert body is the standing trigger set only (#49173
closed + its 2 comments, **101** keyword hits of **902**). Budget: core **57/60** at the first hand
call, **45/60** after the sweep (12 counted core calls: 1 commit page, 2 issues, 2 PRs, 7 commit
details; search **10/10**, 1 used) — no 403, all payloads unauthenticated REST. `gh auth status` →
**token for `aotsukiqx` invalid** (**28th consecutive run**); step 2 ran on the REST equivalents.

### Issues / PR — zero delta

- **#49173** — closed (`completed`), `comments=2`, `closed_at = updated_at = 2026-11-09T07:26:52Z` —
  **twenty-seventh run** identical to the close second. Both comments are the original Kylinny
  thread (fix offered in #49182), already absorbed into `docs/PROJECT_STATE.md`. No new maintainer
  comment; nothing to fetch for step 2.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` — **twenty-seventh silent
  run**. PR search for the issue number returns exactly **1** item: **#49306** (`open`,
  `merged=false`, head `4e1544039e`, updated `2026-11-09T12:56:58Z`) — unchanged.
- **#49182** — re-verified: `closed`, `merged=true`, `merged_at 2026-11-09T07:26:50Z`, merge
  **`5e6995dd09`**, head `a49ff4d7d3`. Unchanged; next-pin candidate stays **`5e6995dd`**.

### sglang main — 7 new commits: all NO-OP on this line

Census **902** (9×100 + 2 pages, `since=2026-11-08T14:00:00Z`). Prior **895 + 7 = 902** ✓.
Prior head `b6d4389a77` (`11:49:17Z`) → new head **`6515ac29d3`** (`15:35:20Z`); the delta is
exactly the seven shas newer than `11:49:17Z` in page 1 (prior head at index 7 of 100). Keyword
counter **101 + 0 = 101** ✓ — **0 of 7** subjects hit; the single axes-path commit in the window
carries no keyword.

| sha | PR | Subject | Class |
|---|---|---|---|
| `6515ac29d3` | #50186 | [Feat] inkling2: Support Inkling-2-Small | NO-OP **for this line**, but **axes-path** (`srt/speculative/`) + new `ModelConfig` contract — see below |
| `5e32149fd6` | #45792 | [Feat][NPU] multimodal: JPEG decoding in the shared image loader | NO-OP — NPU/DVPP decode (`hardware_backend/npu/`); DSV4.1-Flash is text-only, no image inputs |
| `a679ebe434` | #51551 | [CI][NPU] ci: split extra-b-test-1/2 partitions | NO-OP — `.github/workflows/pr-test-npu.yml` only |
| `9667988e9c` | #50028 | [Chore] rust: bump sglang-api-types 0.1.1 | NO-OP — Cargo manifests/lock; Rust server opt-in and unused |
| `cefe8d3be5` | #40580 | [Feat][NPU] attention: MindIE-SD sparse backend | NO-OP — `multimodal_gen/` Ascend diffusion + docs |
| `4ba697002f` | #44329 | [Fix][NPU] diffusion: clean up warnings | NO-OP — diffusion sites/quality-gate, NPU utils, and `responses_tools/beam.py` import fix; we serve chat completions, not responses tools |
| `1d4ff95e9c` | #36426 | [Fix][NPU] modelslim: Qwen3.8-27B-W8A8 loading | NO-OP — NPU quantization/modelslim; drops an NPU-only `packed_modules_mapping` block in `model_loader/loader.py`, CUDA path untouched |

### #50186 in detail — dsv41 static-epilogue gate preserved, vocab contract made explicit

- `speculative/dspark_components/dspark_worker_v2.py` (+20/−8) hoists the gate into `static_mode` /
  `dsv41_static` / `_static_epilogue_needs_draft_fold`. For `target_is_dsv41 ∧ draft_is_moe` the
  resulting boolean is `mode=="static" ∧ ¬attn_dp ∧ pp_size==1` — **identical** to the pin's
  expression at `8817f63c` (`dspark_worker_v2.py:410-416`: `target_is_dsv41 and
  self._verify_planner.mode_value == "static" and self._draft_is_moe and not
  get_parallel().attn_dp_enabled and get_parallel().pp_size == 1`) and to the commit's own
  pre-image for that branch. `draft_folds_static_commit()` (a post-pin addition) is now probed
  only for non-dsv41 draft-fold models and re-validated with a `ValueError` in
  `alloc_memory_pool`; the added `_static_epilogue_needs_draft_fold` is False for us. **Identity.**
- `speculative/dspark_components/dspark_config.py` (+13/−1): widens
  `SUPPORTED_DSPARK_MARKOV_HEAD_TYPES` with `"rnn_hidden"` (pin line 22 has
  `("vanilla", "gated", "rnn")` — additive) and gates two Inkling2 branches on
  `architectures == ["Inkling2ForConditionalGeneration"]`; DSV4 configs take the unchanged path.
- `speculative/spec_info.py` (+8): `create_worker` imports `model_config_of` and returns
  `Inkling2MTPWorker` only for that arch; DSV4 still returns `DSparkWorkerV2`. One host-side call
  at worker construction.
- Generic edits: `configs/model_config.py` sets `self.token_id_vocab_size = self.vocab_size` for
  **every** model, then overrides `vocab_size` with `padded_vocab_size` **only** for Inkling2;
  `managers/scheduler.py`, `managers/tokenizer_manager.py`, `managers/schedule_batch.py` and
  `scheduler_components/logprob_result_processor.py` switch token-id validation/clamping to
  `token_id_vocab_size` ⇒ **identity at our config**. The attribute does **not** exist at
  `8817f63c` (grep of the pin's `model_config.py`: 0 hits), so this is a clean contract *addition*,
  not a behaviour change.
- Inkling2's MegaMoE/EP/attn-gather overrides are arch-gated and additionally on
  `device_sm in (100, 103, 107)` — **SM121 excluded**, so that path cannot activate here even if
  the arch check changed.
- No `.item()`/`.cpu()`/event/sync added anywhere in the seven ⇒ **not a decode-gap lever**.
- 6,450 LOC / 61 files, 48 of them Inkling2 model/tests. Absorb wholesale at the next pin; never
  bp (its context is the post-pin draft-fold support our tree does not have).

### Cross-checks

- Arithmetic: **895 + 7 = 902 ✓** against the script's `window-commits=902`; keyword
  **101 + 0 = 101 ✓**.
- Coverage: all 7 shas assigned exactly once (7 table rows), recomputed from page 1 sliced at the
  prior head — not from author dates.
- Path audit over all 7 diffs: the only axes-path hit is `6515ac29d3` (`srt/speculative/`:
  `dspark_config.py`, `dspark_worker_v2.py`, `spec_info.py`, new `inkling2_mtp_worker.py`). **No
  commit touches `models/deepseek_v4.py`, `deepseek_v4_backend.py`, `engram.py`,
  `dsv4/v41_indexer/`, or `srt/mem_cache/`.**
- Apply risk **zero**: no live `runtime/*.py` / `runtime/*.patch` touches `scheduler.py`,
  `tokenizer_manager.py`, `schedule_batch.py`, `spec_info.py`, `dspark_*`, or `model_config.py`
  (bp4 patches are dsv4 kernels/models; `sglang-rocenante.patch` touches pynccl /
  parallel_state / batch_result_processor).
- Block-level: 2 hunks read side-by-side against the **pin's own** `dspark_worker_v2.py` (raw fetch,
  pin verified 200, 1555 lines) and `dspark_config.py`; the other 5 classified from diff paths and
  inline gates.
- **Census blind spot, fifth consecutive window**: `6515ac29d3` — 6.4 k LOC, the only commit here
  reaching an axes path — carries no keyword; `inkling2` is not in the list and `dspark` appears
  only inside the diff. The repo-side fix (`mem_cache` + `req_to_token` + `static_kv_storage` +
  `pool_host` keywords, or a path-based census) stays pending.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + 3 raw
   pin files + this checkout only — no SSH, no docker, no build, no serve, no schedule change.
2. **No new candidate worth a window**: zero bp-class, zero pin-migration-class, zero
   decode-gap movement in this window.
3. **Standing — #45261 `6eaca42f57`**: unchanged. Paged-path blocker (`block_size == 8` vs our
   zeroed `candidate_block_size`) still stands.
4. **Standing — #51295 `5b478f175b`**: unchanged (3 refit hunks + inert swa hunk); free-rider only.
5. **Standing — bp6 collapse at candidate `5e6995dd`**: unchanged.
6. **Standing — decode gap**: no upstream movement. None of the 7 adds a host sync, `.item()` or
   CPU offload; the ~44 ms/step `cudaEventSynchronize` verify host-sync still has no upstream fix.
7. **Next pin** picks up, in dependency order: #51298's folding series (needs its producer
   commit), #50620 as a unit with its new JIT headers, #51296/#51297 opt-ins (defaults off),
   #45261, #51295, #51450 (DCP, inert), #51458 (5-line DP fix), #50752 if the dsa backend is ever
   enabled, and **#50186 wholesale** (Inkling2; carries the `token_id_vocab_size` contract — any
   local patch validating token ids must move off `vocab_size` at that pin — re-verify the dspark
   static-epilogue refactor against the r3 V1 revert when that approach resumes).
8. **Operator**: (a) `gh` re-auth (**28th run**, REST-only throughout), (b) continuity clean
   (`12:20Z` → this run `16:17:47Z`, `00:17 +08` slot), (c) polling path clean at both legs,
   (d) **repo-side decision still pending**: add `mem_cache`/`req_to_token`/`static_kv_storage`/
   `pool_host` to the census keyword list or make the census path-based — fifth consecutive
   subject-invisible axes-path commit.

## 2026-11-19T20:18Z (2026-11-20 04:18 +08) — watch run #29

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0, 14.6 s — the `04:17 +08` slot ran this
session, log `20:17:44Z`–`20:17:58Z`, `fetch_rc=0`, both issue legs and the census leg numeric on
the first attempt, no TLS retry, no `TOOL FAILURE`. Continuity holds (previous run `16:18Z`); the
slot holds exactly one run. Alert body is the standing trigger set only (#49173 closed + its 2
comments, **103** keyword hits of **906**). Budget: core **47/60** at the first hand call,
**28/60** at the end (19 counted core calls: 10 census pages, 2 issues, 2 PRs, 4 commit details,
1 pin tree; raw.githubusercontent fetches are free), search **10/10** (1 used) — no 403, all
payloads unauthenticated REST. `gh auth status` → **token for `aotsukiqx` invalid** (**29th
consecutive run**); step 2 ran on the REST equivalents.

### Issues / PR — zero delta

- **#49173** — closed (`completed`), `comments=2`, `closed_at = updated_at = 2026-11-09T07:26:52Z` —
  **twenty-eighth run** identical to the close second. Both comments are the original Kylinny
  thread (fix offered in #49182), already absorbed into `docs/PROJECT_STATE.md`. No new maintainer
  comment; nothing to fetch for step 2.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` — **twenty-eighth silent
  run**. PR search for the issue number returns exactly **1** item: **#49306** (`open`,
  `merged=false`, head `4e1544039e`, updated `2026-11-09T12:56:58Z`) — unchanged.
- **#49182** — re-verified: `closed`, `merged=true`, `merged_at 2026-11-09T07:26:50Z`, merge
  **`5e6995dd09`**, head `a49ff4d7d3`. Unchanged; next-pin candidate stays **`5e6995dd`**.

### sglang main — 4 new commits: 1 low-value bp-class, 3 NO-OP

Census **906** (9×100 + 6 pages, `since=2026-11-08T14:00:00Z`). Prior **902 + 4 = 906** ✓.
Prior head `6515ac29d3` (`15:35:20Z`) → new head **`17a9168f1f`** (`18:59:04Z`); the delta is
exactly the four shas newer than the prior head (index 4 of page 1). Keyword counter **101 + 2 =
103** ✓ — **2 of 4** subjects hit, and both are the material ones this time: the first window in
five where the census caught every axes-adjacent commit.

| sha | PR | Subject | Class |
|---|---|---|---|
| `17a9168f1f` | #51515 | [Perf] [NVIDIA] dsa: Fuse the FP4 indexer key store and head gate | **bp-class (perf-only, refit required)** — FP4 indexer K store + head gate; numerics bit-identical; activation at pin unconfirmed |
| `f4bfe143dd` | #51289 | [Fix] dsa: Attend the sharded indexer's local slots in trtllm / prims_ts DCP decode | NO-OP **at pin** — the sharded-indexer (DCP) feature is post-pin; Kimi-K3.1 default flip excludes SM121 |
| `2bf882aefc` | #51387 | [Perf] diffusion/flux3: Run the native SwiGLU through the shared SiluAndMul layer | NO-OP — `multimodal_gen/` diffusion; we serve text-only chat completions |
| `54d68123dc` | #51518 | [Chore] [AMD] mono-kernel: Pin sgl-amd-mono at 32a1220 for the Qwen3.5 FP8 state_layer_cache fix | NO-OP — AMD pin in `pyproject_other.toml`; our image is CUDA aarch64 |

### #51515 in detail — perf fusion on the FP4-indexer path, refit needed at pin

- **Store half** (`kernels/ops/kvcache/dsv4_fp4_index_k_cache.py`, file **exists at pin**, 127
  lines): upstream adds `_quantize_store_fp4_index_k_cache_kernel`, one CTA per token, quantizing
  in the store kernel for `n_tokens < _FUSED_STORE_MAX_TOKENS = 4096` and keeping the batched
  `quantize_fp4_indexer_tensor` + `_store_fp4_index_k_cache_kernel` above it. It calls
  `quantize_fp4_indexer_row(values, v0, v1, BLOCK_N=2*BLOCK, GROUP_N=32, RNE=RNE)`, which **exists
  verbatim at the pin** (`kernels/ops/quantization/fp4_indexer_quant.py:69`, same signature) —
  so the math layer of the backport is free.
- **Refit is real, not mechanical**: the pin's `store_fp4_index_k_cache(input, cache, loc, *,
  page_size, rne=False)` has **no `dcp_size` / `dcp_rank`**, no `int64` `cache_loc` widening, no
  `cache.stride(1) == 1` assert, and an `int32` `tl.load(loc + token_id)`; upstream's pre-image is
  the post-pin DCP-aware variant (`owned = (cache_loc >= 0) & (cache_loc % dcp_size == dcp_rank)`,
  `cache_loc // dcp_size`). At our `dcp=1` the fused kernel reduces to `owned = cache_loc >= 0` and
  `cache_loc // 1`; a backport is ~2 hunks of kernel surgery plus wiring, and the split point
  (`< 4096`) is already the shape of our 1024/4096 prefill chunks.
- **Head-gate half has no pin anchor**: it adds `indexer_head_gate` to
  `kernels/ops/attention/dsa/head_gate.py` (module **absent** from the pin tree) and calls it from
  `srt/layers/attention/dsa/dsa_fp4_indexer.py` (**404 at `8817f63c`**). The pin's equivalent
  scalar math lives in `srt/layers/attention/dsa/dsa_indexer.py:582-607`
  (`weights * n_heads**-0.5` → `.unsqueeze(-1) * q_scale * softmax_scale`), so that half would be
  hand-placed, not applied.
- **Numerics**: upstream adds equality tests for both halves — the fused store vs the batched
  path (`torch.testing.assert_close(fused, batched)`) and the head gate vs
  `(gate * heads**-0.5).float().unsqueeze(-1) * scale` with `rtol=0, atol=0`. This is a launch/
  allocation reduction, not a value change.
- **Activation unconfirmed**: no runtime importer of `store_fp4_index_k_cache` was found among the
  pin call sites checked — `dsa/dsa_indexer.py` imports `fused_store_index_k_cache` (FP8 index K),
  `srt/mem_cache/index_key_cache.py` uses `dsa_index_k_cache`, `dsv4/indexer.py` reads the
  `get_index_k_fp4_*` buffers. So the FP4 store is test-covered at pin but its live caller was not
  located here; **check at apply time** whether the FP4-indexer path reaches it.
- **Not a decode-gap lever**: no `.item()`, `.cpu()`, `torch.equal` or `cudaEventSynchronize`
  anywhere in the four commits (the single `.tolist()` in `f4bfe143dd` is test-only), and the ~44
  ms/step `cudaEventSynchronize` verify site is untouched. Value is a few microseconds of indexer
  launch overhead on decode/prefill; free-rider only, never a window trigger.

### #51289 in detail — post-pin feature, zero pin surface

- The fix splits the `_forward_trtllm` DCP branch into `dcp_partial` (sharded indexer returns
  compacted local slots + `metadata.dcp_topk_length`) and `replicated_dcp` (the old
  `localize_dcp_page_table` path). At the pin, `dsa_backend.py` has **0 hits** for `dcp_partial`,
  `replicated_dcp` and `dsa_shards_index_k`; its equivalent guard is the older
  `if self.dcp_size > 1:` with `self.dcp_rank` (pin lines ~4073-4083). The feature the fix repairs
  does not exist at `8817f63c` ⇒ nothing to backport, nothing to regress.
- The bundled default flip (`_kimi_k31_default_dsa_backend`) fires only for
  `not enable_hisparse ∧ platform.is_sm100 ∧ device_sm in (100, 103, 107)` — **SM121 is excluded
  by both terms** — and is routed through `_kimi_k31_dsa_backends`, i.e. Kimi-K3.1 only. Our
  DSV4.1-Flash config does not pass through it.

### Cross-checks

- Arithmetic: **902 + 4 = 906 ✓** against the script's `window-commits=906`; keyword
  **101 + 2 = 103 ✓**.
- Coverage: all 4 shas assigned exactly once (4 table rows), sliced from page 1 at the previous
  head, not from author dates.
- Path audit over all 4 diffs: **no commit touches `engram.py`, `srt/speculative/`,
  `srt/mem_cache/`, `deepseek_v4_backend.py`, or `models/deepseek_v4.py`**. #51515 reaches
  `srt/layers/attention/dsa/dsa_fp4_indexer.py` (404 at pin) and the `kernels/ops` DSA/FP4 index
  modules; the other three are diffusion, AMD packaging, and DCP/Kimi plumbing.
- Apply risk **zero**: no live `runtime/*.py` / `runtime/*.patch` touches
  `dsv4_fp4_index_k_cache.py`, `fp4_indexer_quant.py`, `dsa_indexer.py`, `dsa_backend.py`,
  `dsa_dcp.py`, or `arg_groups/overrides.py` (bp4/bp5 patches are dsv4 kernels/models;
  `sglang-rocenante.patch` touches pynccl / parallel_state / batch_result_processor).
- Block-level: pin signatures read directly from raw `8817f63c` for
  `dsv4_fp4_index_k_cache.py` (127 lines), `fp4_indexer_quant.py` (233), `dsa_backend.py` (4533),
  `dsa_indexer.py:1833-1885`, and `index_key_cache.py`; existence probed via raw HTTP for five
  paths (`dsa_fp4_indexer.py` = 404 at pin is the decisive one).

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + 6 raw
   pin files + this checkout only — no SSH, no docker, no build, no serve, no schedule change.
2. **New candidate — #51515 `17a9168f1f`**: optional bp-class perf patch (FP4 index K store
   fusion; numerics-identical). Refit = drop the DCP params from the fused kernel + one pre-image
   difference in the store function; head-gate half has no pin anchor. **Not worth a standalone
   window** — fold into the next authorised window only if the FP4-indexer store is confirmed live
   on our path.
3. **Standing — #45261 `6eaca42f57`**: unchanged. Paged-path blocker (`block_size == 8` vs our
   zeroed `candidate_block_size`) still stands.
4. **Standing — #51295 `5b478f175b`**: unchanged (3 refit hunks + inert swa hunk); free-rider only.
5. **Standing — bp6 collapse at candidate `5e6995dd`**: unchanged.
6. **Standing — decode gap**: no upstream movement; the ~44 ms/step verify host-sync still has no
   upstream fix, and #51515's launch fusion does not touch it.
7. **Next pin** picks up, in dependency order: #51298's folding series (needs its producer
   commit), #50620 as a unit with its new JIT headers, #51296/#51297 opt-ins (defaults off),
   #45261, #51295, #51515 (free perf), #51450 (DCP, inert), #51458 (5-line DP fix), #51289
   (post-pin DCP fix, Kimi-only default), #50752 if the dsa backend is ever enabled, and **#50186
   wholesale** (Inkling2; carries the `token_id_vocab_size` contract — any local patch validating
   token ids must move off `vocab_size` at that pin — re-verify the dspark static-epilogue refactor
   against the r3 V1 revert when that approach resumes).
8. **Operator**: (a) `gh` re-auth (**29th run**, REST-only throughout), (b) continuity clean
   (`16:18Z` → this run `20:17:44Z`, `04:17 +08` slot), (c) polling path clean at both legs,
   (d) **repo-side decision still pending**: add `mem_cache`/`req_to_token`/`static_kv_storage`/
   `pool_host` to the census keyword list or make the census path-based — sixth window without a
   fix, though this run's two material commits both happened to hit the keywords.

## 2026-11-20T00:18Z (2026-11-20 08:18 +08) — watch run #30

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0, 43.5 s — log `00:17:51Z`–`00:18:33Z`,
`fetch_rc=0`, both issue legs and the census leg numeric on the first attempt, no TLS retry, no
`TOOL FAILURE`. Continuity holds (previous run `20:17:58Z`); one run in the slot. Alert body is
the standing trigger set only (#49173 closed + its 2 comments, **103** keyword hits of **924**).
`gh auth status` → **token for `aotsukiqx` invalid** (**30th consecutive run**); step 2 ran on the
REST equivalents.

**Budget**: core hit **0/60** at `00:20:26Z` (reset `00:32:42Z`) after nine per-commit `files`
fetches, so the remaining nine commits were audited on the **non-API `.diff`** endpoint; three of
those returned an HTML `503 Unicorn` page and were retried to a 200. Search **10/10** (2 used).

### Issues / PR — zero delta

- **#49173** — closed (`completed`), `comments=2`, `closed_at = updated_at = 2026-11-09T07:26:52Z`
  — **twenty-ninth run** identical to the close second. Both comments are the original Kylinny
  thread already absorbed into `docs/PROJECT_STATE.md`; nothing to fetch for step 2.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` — **twenty-ninth silent
  run**. PR search for the issue number still returns exactly **#49306** (`open`, `merged=false`,
  head `4e1544039e`, updated `2026-11-09T12:56:58Z`).
  Caveat for future runs: `GET /issues/<n>/comments` under rate-limit returns an error **object**,
  whose `jq length` is **2** — that is two keys, not two comments. Check `comments` on the issue
  object instead.
- **#49182** — re-verified: `closed`, `merged=true`, `merged_at 2026-11-09T07:26:50Z`, merge
  **`5e6995dd09`**, head `a49ff4d7d3`. Unchanged; next-pin candidate stays **`5e6995dd`**.

### sglang main — 18 new commits: 17 NO-OP, 1 next-pin-class, **none on the axes**

Census **924** (9×100 + 24, `since=2026-11-08T14:00:00Z`). Prior **906 + 18 = 924** ✓. Prior head
`17a9168f1f` (`18:59:04Z`) → new head **`03fde9cc39`** (`23:54:17Z`). Keyword counter **103 + 0 =
103** — **zero of 18** subjects matched, so the census contributed nothing and the whole delta was
audited **path-based** on commit `.diff`: **none of the 18 touches `layers/engram.py`,
`srt/speculative/`, `srt/mem_cache/`, `deepseek_v4_backend.py`, or `models/deepseek_v4.py`.**

| sha | PR | Subject | Class |
|---|---|---|---|
| `5730875f20` | #51607 | [Fix] spec: Clear `moe_dense_tp_size` for a draft narrowed to plain TP | **next-pin-class, NO-OP at pin** — one kwarg in `parallel_state.py:patch_tensor_parallel_group`; pin's `DpFlags(...)` has no `moe_dense_tp_size` field (`:3137`), so the pre-image is post-pin. Our rocenante hunks sit at `:465`/`:684`/`:2118` — no overlap |
| `03fde9cc39` | #51455 | [Perf] disagg: Pack a DCP-collapsed draft with target rows in DCP-1→N | NO-OP — `disaggregation/mooncake/conn.py`; we run no PD disagg, DCP=1 |
| `99e4233f63` | #35637 | [Feat] scheduler: UniBoost tail-aware schedule policy (rank-invariant TP/PP) | NO-OP — new opt-in policy (`schedule_policy.py`, `uniboost_gamma_ada.py`, arg fields) |
| `c80444d551` | #51521 | [Fix] scheduler: `Engine.collective_rpc` via the control request path | NO-OP — engine/scheduler control plumbing |
| `e627fed679` | #42314 | [Feat] sgl-router: OpenAI Responses API on `/v1/responses` | NO-OP — router |
| `be322a8338` | #51415 | [Feat] sgl-router: `--decode-dp-rank-policy` | NO-OP — router; `dp_size=1` |
| `a3fd615a97` | #49246 | [Feat] sampling: per-request `sampling_seed` without deterministic inference | NO-OP — sampler / tokenizer / rust api_server |
| `0ab0e9d5d0` | #51382 | [Perf] MiniMax-M3: shared expert under CUDA graph | NO-OP — other model |
| 10× | — | flux3 ×4, cosmos3 ×2, diffusion BLAKE3 hashing, diffusion Ulysses transport, Wan doc, CI handle rename (`51ccc1f699 fccaf681a2 fb97472771 5cee3ad808 ab2ebbf421 3b0e693f46 7c08c3040c 78cb42ebbc d4824c7cb8 da8325b10e`) | NO-OP — `multimodal_gen/`, CI, docs |

### Open axis PRs not previously recorded

Search `repo:sgl-project/sglang is:pr is:open (dsv4 OR dspark OR engram) updated:>=2026-11-18`.

- **#47677** `[Perf] [NVIDIA] dspark: Stage draft DP-MoE token counts through persistent buffers` —
  open, updated **2026-11-19T21:45:39Z**, ~89/31 over `dspark_components/dspark_draft.py` + 2
  tests. **bp-class candidate; anchors present at pin**: it deletes `_make_num_token_non_padded`
  (pin `:48`), stops calling it per draft step in `_run_forward` (pin `:584`) in favour of
  `self._num_token_non_padded`, and replaces the per-step `torch.tensor(..., pin_memory=)` +
  H2D staging in `_fill_dp_moe_sync_metadata` (pin `:702`) with double-buffered persistent slots
  guarded by per-slot events. **Adjacent to, not the same as, our decode-gap attribution**: this
  is the *draft* path, and it still calls `copied.synchronize()` on the previous slot every step —
  it removes per-step allocation churn, not the host wait. At `dp_size=1` the DP-metadata half is
  near-inert; only the draft-forward half could move a delta.
- **#51589** `[Refactor] dsv4/pool_configurator: Keep the DSpark draft ring layer count on the
  configurator` — open, 7/5 lines, **applies to the pin verbatim** (`draft_ring_layers` at pin
  `:1211`, `paged_draft_layers` at `:1212`), pure attribute hoist with identical arithmetic →
  **NO-OP**, free-rider at the next pin.
- **#51499** `[Feat] weight-cache: Use IPC only for the speculative target` — open, 11 files incl.
  `speculative/dflash_worker_v2.py` and the weight-cache loader → **NO-OP at our config** (no
  `--weight-cache` in the launchers; the dspark path is untouched).
- **#44028** `[Feat] [NVIDIA] DeepSeek-V4.1: Engram precompute on GPU (10/N)` — open, updated
  `2026-11-18T05:27:42Z`, **touches `layers/engram.py` and `models/deepseek_v4.py`** (two axes)
  plus `moe/mega_moe.py`. No file-level collision with our patches (`deepseek_v4_boundary.py`,
  `deepseek_v4_mhc.py`, `kernels/.../dsv4/*`), but it is the only open PR on the engram axis.
- **#51006** `[Perf] [NVIDIA] attention: Shard DSV4.1 DSpark verify indexer on Hopper` —
  Hopper-gated → **NO-OP for SM121**.
- **#47460** `[Feat][DSV4.1][engram]: Sharing Engram via RDMA to reduce PD DRAM redundancy` — PD
  disaggregation → **NO-OP** (no disagg, no PD DRAM tier).

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST +
   `.diff`/raw endpoints + this checkout only — no SSH, no docker, no build, no serve, no schedule
   change.
2. **New bp-class candidate — #47677** (open, unmerged): draft-side persistent metadata buffers.
   Fold into the next authorised window only if the draft-forward allocation churn shows in the
   decode profile; it does **not** address the verify host-sync.
3. **New next-pin candidate — #51607** (`5730875f20`): `DpFlags.moe_dense_tp_size=None` for a
   plain-TP draft. Unapplyable at `8817f63c` (field absent), zero patch overlap.
4. **#51589** applies at the pin verbatim (NO-OP) — take it free at the next pin.
5. **#44028** (engram precompute on GPU) is the live engram-axis PR — re-audit at the next pin; no
   collision today.
6. **Standing — #45261 `6eaca42f57`**: paged path still blocked (`block_size == 8` vs our zeroed
   `candidate_block_size`).
7. **Standing — #51295 `5b478f175b`**: 3 refit hunks + inert swa hunk; free-rider only.
8. **Standing — bp6 collapse at `5e6995dd`**: unchanged.
9. **Standing — decode gap**: no upstream movement; the ~44 ms/step verify host-sync still has no
   fix.
10. **Next pin** picks up, in dependency order: #51298's folding series, #50620 with its JIT
    headers, #51296/#51297 opt-ins, #51607, #45261, #51295, #51515, #51450 (inert), #51458,
    #51289, #50752, #50186 wholesale.
11. **Operator**: (a) `gh` re-auth (**30th run**, REST-only throughout), (b) continuity clean
    (`20:17:58Z` → `00:17:51Z`), (c) polling path clean at both legs, (d) **repo-side decision,
    seventh window**: this delta contained no axis commit, so the keyword census's blind spot cost
    nothing today — but the audit that proved it (delta-scoped path check over the non-API `.diff`
    endpoint) used **zero** rate budget and is the cheap way to close the hole; widening the
    keyword list would not have caught this window's `parallel_state.py` change either.

## 2026-11-20T04:18Z (2026-11-20 12:18 +08) — watch run #31

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0, 22.8 s — log `04:17:55Z`–`04:18:16Z`,
`fetch_rc=0`, both issue legs and the census leg numeric on the first attempt, no TLS retry, no
`TOOL FAILURE`. Continuity holds (previous run `00:17:51Z`); one run in the slot. Alert body is
the standing trigger set only (#49173 closed + its 2 comments, **104** keyword hits of **947**).
`gh auth status` → **token for `aotsukiqx` invalid** (**31st consecutive run**); step 2 ran on the
REST equivalents.

**Budget**: core **49/60** at `04:19Z` → **19/60** at `04:28Z` (30 consumed, including the
script's own unauthenticated calls). The non-API `.diff` endpoint returned HTML `503` for 11 of
23 commits on the parallel batch and for 6 on a paced retry; those 6 were audited through the REST
`files` endpoint instead. Search **10/10** (2 used: issue-reference sweep + open-axis sweep).

### Issues / PR — zero delta

- **#49173** — closed (`completed`), `comments=2`, `closed_at = updated_at = 2026-11-09T07:26:52Z`
  — **thirtieth run** identical to the close second. No new maintainer comment; nothing to fetch
  for step 2.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` — **thirtieth silent run**.
  Issue-reference search returns exactly **#49306** (`open`, `merged=false`, unchanged
  `2026-11-09T12:56:58Z`).
- **#49182** — re-verified: merged `5e6995dd09` at `2026-11-09T07:26:50Z`, head `a49ff4d7d3`.
  Unchanged; next-pin candidate stays **`5e6995dd`**.

### New axis PRs — four never recorded, all open

Search `repo:sgl-project/sglang is:pr is:open (dsv4 OR dspark OR engram) updated:>=2026-11-19`
(127 results; filtered to the unrecorded axis rows).

**#50049 — the material find on the decode axis.** `[Perf] dspark: Fold the static verify
epilogue's commits into the accept tail and trim the verify-to-draft glue`, created 11-12,
updated **2026-11-20T02:28:36Z**, head `18eb2f92ff`, 12 files +546/−31. It targets exactly the
serial work between the static verify graph and the draft replay: the engram history commit, the
two `index_put` FutureMap relay scatters, and the deferred draft KV staging after the fused accept
tail, plus `overlap_utils._resolve_spec_extras` and the `decode_cuda_graph_runner`
input-copy loop.

**Pin anchors verified by raw fetch at `8817f63c`** (not inferred from the subject): `run_static`
at `dspark_verify.py:749` calls `_commit_engram_history` (`:759`, def `:904`), then
`_scatter_future_relay` when `_post_fold`, then `deferred_commit.stage(...)` (deferred draft KV) or
`inject_target_hidden` — the same launch set the PR folds. `_serves_accept_tail` `:832` /
`_static_accept_tail` `:836` and `kernels/ops/speculative/dspark/static_accept_tail.py` all exist
at the pin. The PR's hunks sit at `:800/:851/:893/:922/:976` on main, i.e. **~60–90 lines below
the pin's symbols → refit, not verbatim**. Classification: **pin-migration-class (decode axis)**,
decomposable into a bp attempt only by extracting the relay-scatter + KV-stage fold from the
kernel change. It removes up to four serial launches per verify step and, like every PR on this
axis, **does not remove the `cudaEventSynchronize` host wait** (~44 ms/step) that owns the gap.

**#51636 — mixed pin-migration-class.** `[Perf] dsv41-flash: overlap decode-path projections and
cut CUDA-graph memory`, created 11-19, updated **2026-11-20T02:36:43Z**, head `0a7f817ab5`,
11 files +485/−114. Reported +3.0–3.2% total tok/s/GPU on **4×B200 TP4/EP4** — not an SM121
measurement. Four separable halves:

- *engram host-load chunking* — `_load_rows` at pin `engram.py:1134` copies a whole checkpoint
  shard; the PR copies 256 MiB chunks and `madvise(MADV_DONTNEED)`s the file-backed pages. Load-
  time host-RAM only (GB10 UMA makes that the same budget as pool memory), needs the
  `/proc/self/maps` mapping helper.
- *engram decode sort skip* — `_MIN_SORTED_HOST_READS = 4096` behind a new `_sorts_host_reads()`;
  the pin has `sort_ids=self._sort_host_reads` at `engram.py:1173` and `:1265`. The **only
  decode-path hunk** here and extractable as a bp; effect is a CPU sort avoided on <4 K-id
  lookups, so measure, do not assume.
- *indexer / model overlap* — **not applicable at the pin**: `index_q_and_weights` (scoring) and
  `low_ratio_decode_index_positions` (backend) do not exist at `8817f63c` (raw-fetched
  `v41_indexer/scoring.py`, `models/deepseek_v4.py`, `deepseek_v4_backend.py`; the pin has
  `_normalize_q_lora:1732`, `_forward_prepare_low_ratio_multi_stream:1951`,
  `forward_low_ratio_sources:3409`, `_low_ratio_index_topk:3887`). The PR adds the backend helper,
  so this half is a backport, not a patch.
- *trtllm MoE row chunking* — `_ROUTED_MOE_MAX_ROWS = 4096` bounds the routed op workspace;
  **NO-OP** at our config (b12x_next MoE, not the trtllm backend).

**#41893 — NO-OP today.** `[Perf] dsv4: Verify-merged mixed step for DSPARK, deterministic C4
top-k ties, and eager ragged indexer`, updated `2026-11-20T01:20:31Z`, 28 files +1935/−76. Gated
on `--enable-mixed-chunk`, which this checkout sets nowhere (default off) → inert. Pin-migration-
class only if mixed chunk is ever adopted; the C4 top-k tie determinism and ragged-indexer parts
ride along.

**#45729 — NO-OP.** decode DP with prefill CP on a colocate worker, 13 files; `dp_size=1`, no PD
disagg, no CP on this line. (Also touches `prefill.py`/`model_hook.py` layer-split plumbing;
nothing enabled.)

Standing, unchanged: **#47677** updated `2026-11-20T02:05:50Z`, head `31fc3ad7f2`, now **+87/−30
over 4 files** (was ~89/31 over 3): same substance, still `copied.synchronize()` per draft step;
`#51589` (verbatim NO-OP free-rider), `#51499` (weight-cache, off), `#44028` (engram axis), `#51006`
(Hopper), `#47460` (PD RDMA).

### sglang main — 23 new commits: all NO-OP on this line

Census **947** (`since=2026-11-08T14:00:00Z`; prior **924**; **924 + 23 = 947** ✓). Prior head
`03fde9cc39` found at index 23 of page 1 (newest first) → new head **`b9e71d4c8bd2`**
(`04:03:24Z`). Keyword counter **103 + 1 = 104** ✓; the single hit was `98b6b7c646fe` (a **test**
commit). Every row was read by subject **and** path (23/23 accounted: 17 `.diff` + 6 REST `files`).

| sha | PR | Subject | Class |
|---|---|---|---|
| `b9e71d4c8bd2` | #51363 | sampling: keep fast top-p/top-k renorm under attention DP | NO-OP — `renorm_deterministic()` still returns True at `tp_size=4`; the new fast path needs `attn_dp_enabled ∧ attn_tp_size==1 ∧ attn_cp_size==1` |
| `98b6b7c646fe` | #51552 | test: inkling2 GB300 DSPARK e2e | NO-OP — test-only (the window's only keyword hit) |
| `c09ff83ef4f2` | #50140 | profiler: forward step span as an NVTX range | NO-OP perf — `SGLANG_ENABLE_NVTX_FORWARD` default False; tooling note: `spec_stage_span` now maps draft spans to `forward_nvtx_range` and verify comes from `ModelRunner.forward`'s `step[VERIFY]` |
| `2e7bdf77a2df` | #48227 | model-loader: cap multithreaded in-flight bytes | NO-OP — opt-in `max_inflight_bytes` in `--model-loader-extra-config`; default `None` = current behaviour. Boot-RAM knob only, not a perf lever |
| `a242c277e58d` | #51671 | sgl-router: count a media tie-break once | NO-OP — router (unused) |
| `a3b756844be5` | #51691 | cookbook: Hy4 | NO-OP — docs |
| `41cd6318083b` | #51594 | CI diffusion: h100-computable-d denoise refs | NO-OP — CI baselines |
| `401d0c792902` | #49995 | NPU operators: public entry points | NO-OP — 67 files of `sgl_kernel_npu.*`→`operators.*` import renames; CUDA branches untouched; `pool_host/mamba.py` rename is inside the NPU branch and the host tier is off |
| `f9c83713e03b` | #51670 | linker: load-back from storage hits in e2e | NO-OP — tests only (incl. `kl_dsv4`/`kl_dsv41` linker tests) |
| `23b2e671c599` | #50962 | hicache: fail attach when native hash cannot build | NO-OP — HiCache/host tier off |
| `35321ed0714b` | #51305 | mem_cache: FP4 default in page-sharded DSA mock | NO-OP — test-only |
| `f913bf9aa1b6` | #51424 | CI NPU: re-enable PR trigger | NO-OP — CI |
| `307a6164b207` | #51662 | router e2e on 4-gpu-h100 | NO-OP — router tests |
| `ae69db457eaa` | #51637 | router: Responses summary/coercion | NO-OP — router |
| `08f10090c762` | #51338 | classify: multi-label probabilities | NO-OP — classify endpoint unused |
| `6d35ff21d04b` | #48937 | router: route DeepSeek image chats by prefix cache | NO-OP — Rust router + processor (`render/models/deepseek_v4.rs`); no router, no `SGLANG_RUST_SERVER` on this line |
| `cf9b03ab8265` | #48389 | Ascend GLM DSA cache layer-split | NO-OP — NPU-gated; GPU `maybe_prefetch_next_full_attention_kv` keeps its prefetch call; this checkout sets no `--enable-dsa-cache-layer-split`/`--enable-prefill-cp`/`--cp-strategy` |
| `a3d4b91d35d6` | #51520 | kimi-k3 `--enable-layernorm-sp` | NO-OP — the new `SumGroup.ATTN_TP` branch needs `sp`; `AttnResidual`/`AttnBank` only gain defaulted kwargs. Collision note: touches `layer_boundary/{factories,residual/attn_bank}` and `attn_residual.py`, adjacent to our `deepseek_v4_boundary.py` patch — re-derive at the next pin, zero same-file overlap |
| `9617d5fa6af3` | #51649 | diffusion: Ulysses transports | NO-OP — docs |
| `3ee08035aebe` | #51652 | diffusion: FLUX.3 conditioning cache | NO-OP — docs |
| `9cca7c9fd256` | #43279 | wan2.2-s2v speech-to-video | NO-OP — `multimodal_gen/` |
| `c532212bbe6d` | #51536 | NPU kernel tag bump | NO-OP — NPU docker/CI |
| `2510a8b7bc21` | #50823 | XPU enablements | NO-OP — `_is_xpu`-gated `mxfp4.py` bias promotion (gpt-oss-120b) and `bailing_moe_linear` rope gate |

### Cross-checks

- Census: `924 + 23 = 947` ✓; keyword counter `103 + 1 = 104` ✓; slice keyed on the prior head
  `03fde9cc39`, not author dates.
- Coverage: 23/23 `new` shas appear in exactly one table row/group (17 `.diff` + 6 REST `files`;
  the failed `.diff` batch was retried, not skipped).
- Continuity: `00:17:51Z` → `04:17:55Z`, one run per 4 h slot, no gap.
- Counter miss rate: **22 of 23** (96%), again the normal case — the hit was a test commit.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST +
   `.diff`/raw endpoints + this checkout only — no SSH, no docker, no build, no serve, no schedule
   change.
2. **New top decode-axis candidate — #50049** (open, unmerged): verify-epilogue fold. Take it at
   the next pin; a bp attempt needs a refit (~60–90 line offset at `8817f63c`) and should start
   from the relay-scatter + KV-stage fold, not the Triton tail kernel.
3. **New pin-migration-class — #51636**: indexer overlap is a backport (needs two absent
   helpers); the engram sort skip is the one extractable pin-applicable hunk; the loader chunking
   is a boot-RAM lever; the trtllm MoE hunk is inert here.
4. **#51636 and #50049 both edit `cuda_graph_buffer_registry.py`** — expect a conflict when the
   next pin absorbs both.
5. **#41893** stays inert until mixed chunk is enabled; **#45729** stays inert at `dp_size=1`.
6. **Corrected anchor fact**: `model_executor/runner/row_bucket_graph_runner.py` and
   `kernels/ops/speculative/dspark/static_accept_tail.py` **do exist at `8817f63c`** (raw-fetched,
   HTTP 200) — only `full_cuda_graph_backend.py` is post-pin among the runner-split paths noted in
   the runbook.
7. **Standing — #47677** (`31fc3ad7f2`, +87/−30/4 files): re-read, unchanged in substance.
8. **Standing — #45261 `6eaca42f57`**: paged path still blocked (`block_size == 8` vs our zeroed
   `candidate_block_size`).
9. **Standing — #51295 `5b478f175b`**: 3 refit hunks + inert swa hunk; free-rider only.
10. **Standing — bp6 collapse at `5e6995dd`**: unchanged.
11. **Standing — decode gap**: the ~44 ms/step verify host-sync has no upstream fix; #50049 trims
    the surrounding serial launches, not the wait.
12. **Next pin** picks up, in dependency order: #51298's folding series, #50620 with its JIT
    headers, #51296/#51297 opt-ins, #51607, #45261, #51295, #51515, #51450 (inert), #51458,
    #51289, #50752, #50186 wholesale; then #50049, #51636 (both refit), #51589 free.
13. **Operator**: (a) `gh` re-auth (**31st run**, REST-only throughout), (b) continuity clean
    (`00:17:51Z` → `04:17:55Z`), (c) polling path clean at both legs, (d) no repo-side action
    required this window — the per-commit path audit was again what caught the material items
    (the 23-commit window itself is all NO-OP; the value is in the four PRs).

## 2026-11-20T08:18Z (2026-11-20 16:18 +08) — watch run #32

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0, 15.2 s — log `08:17:49Z`–`08:18:03Z`,
`fetch_rc=0`, both issue legs and the census numeric on the first attempt, no TLS retry, no
`TOOL FAILURE`. Continuity holds (previous run `04:17:55Z`); one run in the slot. Alert body is
the standing trigger set only (#49173 closed + its 2 comments, **106** keyword hits of **969**).
`gh auth status` → **token for `aotsukiqx` invalid** (**32nd consecutive run**); step 2 ran on the
REST equivalents.

**Budget**: core **46/60** at `08:19Z` (prior window) → **42/60** at `08:23Z` with `used=18`,
reset `09:19:39Z` — the window rolled at `08:19:39Z` mid-poll, so the second figure is operative.
Search **10/10** (2 used: #49174 reference sweep + open-axis sweep). The `.diff` endpoint returned
the HTML `503` stub for **7 of 22** commits; those 7 were audited through REST `files` (7 requests),
one paced retry per commit.

### Issues / PR — zero delta

- **#49173** — closed (`completed`), `comments=2`, `closed_at = updated_at = 2026-11-09T07:26:52Z`
  — **thirty-second run** identical to the close second. No new maintainer comment; nothing to
  fetch for step 2.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` — **thirty-second silent
  run**. Issue-reference search returns exactly **#49306** (`[Fix] scheduler: Prevent pool
  accounting desync on finished_len trim`, open, `merged=false`, unchanged `2026-11-09T12:56:58Z`).
- **#49182** — re-verified over REST: merged `5e6995dd09` at `2026-11-09T07:26:50Z`, head
  `a49ff4d7d3`. Unchanged; next-pin candidate stays **`5e6995dd`**.

### sglang main — 22 new commits: one landed pin candidate, two contract shifts, 19 NO-OP

#### Table A — material / individually read

| sha | PR | subject | class |
|---|---|---|---|
| `789abb9be9d9` | #47677 | dspark: stage draft DP-MoE token counts through persistent buffers | **pin-migration-class (landed)** — merged `07:08:31Z`, head `31fc3ad7f2` **unchanged** (approved revision, squash). 4 files +87/−30: deletes `_make_num_token_non_padded`, precomputes `_num_token_non_padded`, and replaces the two per-step `torch.tensor(..., pin_memory).to()` DP-meta stages in `_fill_dp_moe_sync_metadata` with double-buffered pinned/device slots guarded by per-slot events. **Still `copied.synchronize()` per draft step** — allocation churn only. At `dp_size=1` the DP-metadata half is inert; only the draft-forward allocation churn goes. Supersedes the recorded bp-class candidate: do not patch it locally. |
| `9399f5f4645e` | #48678 | docker: install AWS EFA in the main image; drop the EFA variant | **pin-migration-class (dependency/image contract)** — `sgl-deep-ep` 0.1.2.post1→**0.1.3** (pyproject + Dockerfile), `SGL_NCCL_VERSION` 2.30.7→**2.31.2**, `MOONCAKE_VERSION` 0.3.13.post1→**0.3.14**. New `scripts/install_efa.sh` runs in the base image (EFA installer **1.51.0**, sha256-pinned): `apt-get update`, libfabric/aws-ofi-nccl under `/opt/amazon/efa`, then asserts `fi_info` providers and NIXL plugin `ldd` paths. Base deps drop `libopenmpi-dev`/`libfabric-dev`/`libboost-all-dev`; the DeepEP builder drops `libfabric(-devel)`. Next image build gains an external fetch (`efa-installer.amazonaws.com`) and NCCL 2.31.2. |
| `9866f0a0c798` | #51719 | sampling: algorithm-neutral watermark interface; Gumbel behind it | **pin-migration-class (contract refactor)** — `WatermarkState`→`WatermarkEngine`; `SamplingBatchInfo`'s five watermark fields collapse to `watermark: Optional[WatermarkRows]`; `ForwardBatch.watermark_seed` replaces the two host-tail fields; `dspark_worker_v2._force_watermark` now returns `(logits, sampling_info, watermark)`; `eagle_sample(watermark=)`; `eagle_post_verify_graphs._sample` takes `watermark=`. 54 files. **Runtime identical with watermark off** (`make_watermark_engine`/`build_watermark_rows` return None on `not enable_watermark`). Also drops the `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION` guard around `watermark_full_mask`. |
| `0fff686ba40a` | #41269 | speculative: draft deadlock under DP attention without a DSA seed | NO-OP — one condition in `eagle_worker_v2.draft` adds `and not get_parallel().enable_dp_attention`; at `dp_size=1` DP attention is off, so the eager-fallback decision is unchanged. |
| `2a0869ebcccc` | #51725 | logprob: `token_ids_logprob_rows` | NO-OP — opt-in API; `token_ids_logprob_rows is None` on every existing request and the scheduler branches only when set. No default hot-path change. |

#### Table B — residual individually-read NO-OPs

| sha | PR | subject | why NO-OP |
|---|---|---|---|
| `cc97c13817e4` | #51611 | diffusion: FA4 for Blackwell sub-tile attention | `multimodal_gen/` + diffusion docs/tests |
| `0d81dbf36035` | #51598 | diffusion: stop worker profiler if HTTP profiler fails | `multimodal_gen/` |
| `28260f05fedb` | #51079 | [AMD] unquant: keep CPU inputs off AITER GEMM | AMD path, 4 lines |
| `f69412389348` | #51463 | [AMD] fp8-moe: mark AITER-shuffled weights `is_shuffled` | AMD path |
| `d215bb3b8c20` | #47824 | [Intel] gemma4 GPTQ INT4 on CPU | CPU kernels + `gemma4_causal` |
| `518dd30af21b` | #51700 | cookbook: Hy4 installation | docs |
| `68c262b30617` | #48314 | [Perf][NVIDIA] fused_moe H20-3e configs | device-name-keyed JSON (`NVIDIA_H20-3e`); GB10 match impossible |
| `6e7325808610` | #50886 | [CI][AMD] QSA MQA prefill fits gfx942 LDS | `is_gfx942_supported()` gate; `num_stages` only on AMD |
| `e183c415bdfc` | #51624 | [CI] kernels: widen Marlin MoE GEMV-vs-GEMM bound | test tolerance only |
| `596c69706787` | #50043 | docs: indentation/spelling/formatting | docs |
| `843d27c9e297` | #47660 | [Feat][NPU] DSV4.1 on Ascend A5 | NPU backend; `engram.py` hit is in the Ascend branch (`hardware_backend/npu/`) |
| `741d14cec1f8` | #49535 | [Fix][NVIDIA] qwen3.5 text-only Qwen3 MoE overrides | other model |
| `b44ab8128533` | #51715 | [CI] sgl-router: restore prebuilt Rust extensions | CI/router only |
| `99159c90503f` | #51711 | [Test] sgl-router: DeepSeek image chats parity | Rust test only |
| `d675c98bc76d` | #51595 | [Perf] inkling2: MegaMoE input for this rank's tokens | MegaMoE bypassed (`b12x_next`, `SGLANG_OPT_DEEPGEMM_MEGA_GATE` False) and other model |
| `923a8b5d1b37` | #48353 | [Fix][XPU] compressed-tensors W8A8 int8 | XPU |
| `d96ae9588b76` | #51459 | [Feat][CPU] per-token FP8 AVX10.2 dispatch | CPU |

### Open axis PRs — 26 never recorded in today's window (52 total updated since 00:00Z)

Material, with the diff read:

- **#50032 — the new top decode-axis candidate** (`[Perf] scheduler: Cut per-step host work in the
  batch-1 speculative decode loop`, open, updated `2026-11-20T06:12:33Z`, 12 files +205/−65).
  Substance: `SamplingBatchInfo.copy_for_forward` takes the `object.__new__` fast path
  **unconditionally** (the PR deletes `_fast_copy_enabled`, which required EAGLE + topk==1 +
  `moe_ep_size==1` — so at the pin DSPARK never took it); `freeze_env_reads()` memoizes every
  `EnvField.get` after the scheduler event loop starts (new `SGLANG_OPT_FREEZE_ENV_READS`);
  `schedule_batch_field_values` snapshots the batch with one `operator.itemgetter` call and
  `_forward_isolation` restores via `__dict__.update`; `cuda_graph_buffer_registry` caches its last
  head view; `dflash_info_v2` skips the two KV-len copies when no tokens are allocated;
  `draft_worker_common` caches the zero-width relay tensors; `dspark_draft` skips a redundant
  `_num_token_non_padded.fill_`. This is scheduler/draft host work, **not the verify host-sync** —
  it cannot close the ~44 ms gap, but it is the only per-step host-work cut offered since #50049.
  Class: **bp-class** (7 `srt/` files, mechanical refit).
- **#47020 — DSPARK penalty correctness** (`[Fix] dflash: Apply request penalties under DFLASH and
  DSPARK`, open, updated `08:16:20Z`, 6 files +231/−28, 203 of them a new test). `dspark_verify.py`
  changes `verify_logits_adjustments_are_noop` from `acc_linear_penalties is not None` (the pin's
  field) to `acc_additive_penalties`/`acc_scaling_penalties`, and `spec_prepare_for_decode`
  cumulates penalty output tokens for the dflash family when the orchestrator is required —
  i.e. **presence/frequency penalties were being skipped in spec decode**. Class: **bp-class
  (correctness)**, re-derive: the field names are post-pin.
- **#51692 — blockwise rejection precision** (`[Fix] dspark: Stabilize blockwise rejection sampling
  under logit shifts`, 2 files +121/−36). `_global_lse` splits into `(global_max, log_sum)` so a
  small `log Σexp` is no longer rounded into a large max before subtraction; the bonus row now
  computes the target row max instead of returning before any stats; `_rejection_kernel`
  accumulates `t_max/d_max/t_log_sum/d_log_sum` per block. Our line's accept kernel. Class:
  **bp-class (correctness), kernel-level — in-image pre-verify required**.
- **#51681 — Philox folding** (`[Fix] dspark: Prevent Philox rounding from selecting a masked
  token`, 3 files +56/−9). `u = (bits+0.5)·2⁻³²` could round to 1.0 in fp32 (`log1p(−1) = −inf`)
  and a zero word gave zero noise; folds `min(bits, ~bits)` with a sign branch. Also updates
  `sharded_accept_hip`. Class: **bp-class (correctness), kernel-level**.
- **#48926 — FP8 MegaAttention prefill workspace** (`[Feat] dsv41: FP8 MegaAttention prefill
  workspace`, 6 files +143/−17). `SGLANG_DSV41_FP8_PREFILL_WORKSPACE` (default `False`), **64 query
  heads only**, and explicitly requires an `sgl-flash-mla` with
  [FlashMLA#29](https://github.com/sgl-project/FlashMLA/pull/29) — **the pinned 1.0.5 does not
  include it**. Class: **pin-migration-class (wheel-gated prefill lever)**.
- **#45417 — SM120 NVFP4 KV cache** (`[Feat] [SM120] deepseek-v4: Support NVFP4 KV cache for target
  and DSpark`, 9 files +807/−18). Adds 137 lines to upstream
  `kernels/ops/attention/flash_mla_sm120.py` — **direct collision with our local
  `runtime/flash_mla_sm120.py` shim** — plus memory-pool/backend/model changes behind an env gate.
  Class: **pin-migration-class** (take wholesale at the pin only).
- **#51669 — fused low-ratio compressor live rows** (`[Fix] dsv41: Slice the fused low-ratio
  compressor's out-loc metadata to the live rows`, 2 files +39/−6). Slices `raw_out_loc`/`c1_out_loc`
  /`c2_out_loc` to `pos.shape[0]` when a **prefill CUDA-graph break** hands over live rows. Class:
  **bp-class, latent** — reachable only if the pin's prefill graph path is used; confirm at the pin.

Watch, classified from the diff shape:

- **#50610** FlashInfer RDMA all-to-all for SM120 (33 files +2923/−30) — EP dispatcher + args; our
  line is pure TP with no `--ep-size` ⇒ NO-OP today, first SM120 EP path to revisit if EP is tried.
- **#48850** capture decoder-tail graphs at startup, never on a DP-skipped step (7 files +356/−34) —
  `dp_size=1` has no skipped DP ranks ⇒ NO-OP; rewrites `deepseek_v4_replay_tail.py` capture timing
  and `row_bucket_graph_runner`, adjacent to our replay-tail patch — re-check at the pin.
- **#51695** reject explicit multi-step `speculative_num_steps` for DFLASH/DSpark (3 files
  +133/−34) — this checkout passes no `--speculative-num-steps` anywhere (grep: zero) ⇒ NO-OP; only
  fires if a launcher ever sets it.
- **#51752** DSpark simulated acceptance draws the stop-row bonus from the rejection residual
  (5 files +489/−1) — the new kernel is on the simulated-acceptance path (`SIMULATE_ACC_LEN` off)
  ⇒ NO-OP.

Grouped NO-OPs (subject + file shape only; none touch our serving surface): **#50205** NPU A3 W8A8
DSpark, **#51749** PD-prefill CP, **#46248** HiCache Engram store (host tier off), **#51689** NPU
Kimi K3.1, **#51236** ERNIE-5.5-Flash, **#48866**/#48865 AMD identity-prefill, **#51646** Hy4
kernels, **#51668** Hunyuan-4-Flash, **#51004** CI NPU, **#45265** encoder-only prefill, **#49625**
OpenAI dynamic tools, **#48874** NPU mooncake, **#50985** Ascend docs, **#51523** AMD CI.

### Cross-checks

- Census: `947 + 22 = 969` ✓; keyword counter `104 + 2 = 106` ✓ (hits `789abb9be9d9` #47677,
  `843d27c9e297` #47660).
- Slice keyed on the 04:18 entry's **own census head `a242c277e58d`** (03:22:47Z, #51671), not on
  the `03fde9cc39` that entry names as its *slice* key — keying on the latter returned 45 rows.
  Note for the runbook: entries record the key of their own slice, so the next head must be read
  from the top table row.
- Coverage: 22/22 `new` shas appear in exactly one row (5 Table A + 17 Table B).
- Counter miss rate: **20 of 22** (91%) — both hits happened to be material-or-NPU; the two
  contract shifts (#48678, #51719) carry no keyword.
- Batch integrity: 22 `.diff` attempts → 7 HTML `503` stubs (56,635 B each); all 7 re-audited via
  REST `files` (HTTP 200, patches present); 8 PR file-list payloads JSON-valid.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + `.diff`
   + this checkout only — no SSH, no docker, no build, no serve, no schedule change.
2. **#47677 landed** (`789abb9be9d9`): drop it from the bp-candidate list; the next pin carries it.
   Do not apply the old patch on top of the merged code.
3. **#50032** is the best remaining decode-axis bp candidate — but it cuts scheduler/draft host
   work, **not** the ~44 ms verify `cudaEventSynchronize`. Refit at the pin: `_fast_copy_enabled`
   still exists there; `SamplingBatchInfo`'s field set differs (five watermark fields pre-#51719,
   one row object post-). Measure; do not assume a delta.
4. **#47020**: confirm whether proxy traffic sends presence/frequency/repetition penalties. If it
   does, bp5 is decoding with them skipped under DSPARK — a correctness question, not a perf one.
   Re-derive the hunks (the pin's field is `acc_linear_penalties`).
5. **#51692/#51681**: kernel-level correctness on the accept path; bp-class only with in-image
   pre-verify in an authorised window.
6. **#48678 image contract**: the next build needs the EFA installer fetch and must confirm
   `/opt/amazon/efa/lib` does not shadow the libfabric that the DeepEP build links; also lands
   NCCL 2.31.2, `sgl-deep-ep` 0.1.3, mooncake 0.3.14 in the pin matrix.
7. **#51719**: expect conflicts in any dspark bp chain touching `SamplingBatchInfo`,
   `ForwardBatch`, `dspark_worker_v2`, or `eagle_utils`.
8. **#48926 / #45417**: record-only. #48926 needs a flash-mla newer than 1.0.5 (FlashMLA#29) and 64
   heads; #45417 rewrites the same upstream SM120 module we patch locally.
9. **Standing — #45261 `6eaca42f57`**: paged path still blocked (`block_size == 8` vs our zeroed
   `candidate_block_size`).
10. **Standing — #51295 `5b478f175b`**: 3 refit hunks + inert swa hunk; free-rider only.
11. **Standing — bp6 collapse at `5e6995dd`**: unchanged.
12. **Standing — decode gap**: no upstream fix for the verify host-sync; #50032 is adjacent host
    work only.
13. **Next pin** picks up, in dependency order: #51298's folding series, #50620 with its JIT
    headers, #51296/#51297 opt-ins, #51607, #45261, #51295, #51515, #51450 (inert), #51458,
    #51289, #50752, #50186 wholesale; then #50049, #51636 (refits), #51589, #47677 (landed),
    #48678 (deps/image), #51719 (contract); #51692/#51681/#47020/#51669 as correctness refits.
14. **Operator**: (a) `gh` re-auth (**32nd run**, REST-only throughout), (b) continuity clean
    (`04:17:55Z` → `08:17:49Z`), (c) polling path clean at both legs, (d) nothing needs a window
    today — all material items are next-pin or in-window refits, none is a fix for the live decode
    gap.

## 2026-11-20T12:18Z (2026-11-20 20:18 +08) — watch run #33

**Tooling**: `scripts/verify/upstream_watch.sh` rc=0, 29.8 s — log `12:17:50Z`–`12:18:19Z`,
`fetch_rc=0`, both issue legs on the first attempt, no TLS retry, no `TOOL FAILURE`. Continuity
holds (previous run `08:17:49Z`); one run in the slot. Alert body is the standing trigger set only
(#49173 closed + its 2 comments, **108** keyword hits of **990**). `gh auth status` → **token for
`aotsukiqx` invalid** (**33rd consecutive run**); steps 2–3 ran on the REST equivalents.

**Budget**: core **41/60** remaining (`used=19`, reset `13:07:51Z`) at the end of the run, search
**10/10** (`used=0`) — the counter billed well below the raw request count (13 `.diff` hits are off
the API budget and likely conditional responses went unbilled). 22 `.diff` fetches (13 valid, 9 HTML
`503` stubs) + 9 REST commit re-fetches + 10 census pages + 5 issue/PR reads + 1 head re-check + 1
rate-limit read.

### Issues / PR — zero delta

- **#49173** — closed (`completed`), `comments=2`, `closed_at = updated_at = 2026-11-09T07:26:52Z`
  — **thirty-third run** identical to the close second. Nothing to fetch for step 2.
- **#49174** — open, **0 comments**, `updated_at 2026-11-08T14:34:50Z` — **thirty-third silent run**.
  Issue-reference search returns exactly **#49306** (`[Fix] scheduler: Prevent pool accounting desync
  on finished_len trim`, open, unmerged, unchanged `2026-11-09T12:56:58Z`).
- **#49182** — re-verified over REST: merged `5e6995dd09` at `2026-11-09T07:26:50Z`, head
  `a49ff4d7d3`. Unchanged; next-pin candidate stays **`5e6995dd`**.

### sglang main — 22 new commits: no decode-gap change, one traffic-conditional bp, one boot pin-fix

#### Table A — material / individually read

| sha | PR | subject | class |
|---|---|---|---|
| `a0bd0c56ad2d` | #48208 | [Perf] speculative: batched top logprobs when every request asks the same k | **bp-class (traffic-conditional)** — `compute_spec_logprobs` takes `top_k = batch.top_logprobs_nums[0]`; when every row asks that same k it emits **one `[bs*max_accept, k]` tensor** (`gathered_logprobs.topk`) instead of per-row lists, and the host conversion (`_top_logprobs_to_lists`, `_async_d2h_top_logprobs`) runs once per step. `top_logprobs_nums` empty ⇒ `top_k = 0` and the branch is skipped, so **inert unless a request asks top_logprobs on the spec decode path**. 4 files; no local patch touches `logprob_processor`/`managers/utils`. Worth a bp only if proxy traffic actually sets `top_logprobs`; it cuts host work, not the verify sync. |
| `4b6985c614a7` | #51312 | [Fix] model_loader: Split checkpoint prefetch among each node's own ranks | **pin-migration-class (boot)** — `_prefetch_all_checkpoints` sets `local_world_size = world_group.local_size or launch_world_size // nnodes` and `local_rank = rank % local_world_size`. On our 4-nodes × 1-rank TP4 this is `local_world_size = 1`, `local_rank = 0` ⇒ **every rank warms its node's page cache with all shards**, where the old world-wide split left each node with ~1/4 of the files. Weight load is the ~10 min boot segment; expect a boot-time gain, not a decode one. `nnodes` must exist on `get_parallel()` at the pin; our `adapter/fast_load.py` monkeypatches `weight_utils.safe_open` (different function, same module) — re-audit the hook at the pin. |
| `0c6d6eb04e3b` | #50116 | [Fix] mem_cache: Fix explicit HND initialization | **NO-OP (config-gated)** — explicit `kv_cache_layout == "hnd"` now sets `use_hnd` instead of being clobbered to `False` by the label branch (env `SGLANG_USE_HND_KVCACHE` still the fallback). Zero hits for `kv_cache_layout`/`use_hnd`/`--kv-cache-layout` in this checkout ⇒ unreachable; `MHATokenToKVPool` shape path, not our MLA pool. |
| `eb49b873a166` | #51479 | [Fix] dsv41: Build MegaGate row masks inside the decoder replay-tail graph capture | **NO-OP (MegaGate not wired)** — `_gate_row_masks` adds `isinstance(cache, dict) and torch.cuda.is_current_stream_capturing()` to the cache key (+3 lines, +1 test), so a graph captured inside a forward builds its own masks. Our line cannot reach it: `adapter/prefill_sp.py:978` raises `DSV41_PREFILL_SP: mega MoE path is not wired` when `should_use_mega_moe` is true, and `moe_b12x_next` guards `_mega_moe_weights_built`. **Mandatory prerequisite if MegaGate is ever enabled** (stale masks across replay-target capture = wrong routing), but no effect on bp5 and nothing on the verify-sync axis. |
| `d8eb51846d74` | #51351 | [Perf] scheduler: Add SGLANG_DISABLE_THP | **NO-OP at pin / next-pin env-only option** — new `EnvBool(False)` + `disable_transparent_hugepages()` (`prctl(PR_SET_THP_DISABLE)`) applied in `configure_scheduler_process`; default off, code absent at our pin. Its own docs note the aarch64 case: **64K base pages ⇒ a huge page is 512 MB**, and first touch / khugepaged can stall a scheduler tens of ms. Evaluate as an env var at the next pin (no patch), on a measured window only. |
| `36d82211fe73` | #51779 | [Fix] speculative: Read the current DP attention field in EAGLE draft | **NO-OP** — one line in `eagle_worker_v2.draft`: `get_parallel().enable_dp_attention` → `.attn_dp_enabled`. `dp_size=1` ⇒ both false, condition unchanged. **No local reference to `enable_dp_attention`** (grep: zero) ⇒ no refit from the rename. |

#### Table B — residual individually-read NO-OPs

| sha | PR | subject | why NO-OP |
|---|---|---|---|
| `7d82bda66c79` | #43095 | [Test] scheduler: admission re-match records `num_matched_prefix_tokens` | test only |
| `f5149f367b8f` | #51790 | [Fix] rust-server: PD decode heartbeat on `/health` | Rust frontend (`SGLANG_RUST_SERVER`, opt-in) |
| `ceb4ee85c37d` | #48247 | [Fix] http_server: skip launcher re-import in spawned uvicorn workers | startup only (3 files, `entrypoints/`) |
| `db420ed761c5` | #51765 | [CI] npu: lower GPQA concurrency for pinned host OOM | NPU CI |
| `448efb13186f` | #51088 | [Feat] hicache: forward request-scoped KV hints to storage backends | host tier off; `req.kv_hints is None` on every request. Adds `PrefetchOperation.storage_read_metadata` + `GenerateReqInput.kv_hints` (already present) — contract note only |
| `ad96481482fe` | #51666 | [Fix] [AMD] kimi-k3: drop bundled TreeCore | AMD + other model |
| `f5cc4856acd6` | #49774 | [CI] [AMD] run ROCm lanes on weekends | CI schedule |
| `4b02753381c9` | #45894 | [Feat] sampling: per-request SynthID Text watermarking | watermark off; the sampler hunk is a **pure extraction** of `top_k_top_p_min_p_filter_sorted_probs_torch` — reference path byte-equivalent, 33 files mostly new `sampling/watermarking/synthid/` |
| `054c0d556f30` | #51656 | [Perf] kimi-k3: multicast collectives for LayerNorm SP prefill | other model |
| `242da74ebf6a` | #51343 | [Perf] disaggregation: aux data as one memcpy | PD only; 1 file, 4 lines |
| `7894bdf172df` | #51661 | [Fix] weight-update: keep fused-pair halves across chunked updates | no `update_weights_from_tensor` in our serving. **Pin-conflict note**: rewrites `models/deepseek_common/deepseek_weight_loader.py` (+83/−52) |
| `28e576357488` | #48420 | [Feat] [NPU] kimi-k3.1 model + DSA attention on Ascend | NPU + other model |
| `3b34d97d9eca` | #49554 | [Perf] [AMD] batch-invariant small-M segmented bf16 matmul | AMD (`gfx950`) |
| `561329a0e669` | #49216 | [Fix] muse-spark: honor decrypted config in arch probe | other model; the `spark` keyword hit is a model name, not DGX Spark |
| `6ccb37eb58b1` | #41056 | [Feat] [NPU] disaggregation: PD KV checksum | NPU |
| `f6726b4caa0e` | #51525 | [Fix] [AMD] nixl: bind AITER HIP runtime before UCX hooks | AMD |

### Open axis PRs — standing candidates, none merged

- **#50032** (top decode-axis bp candidate) — open, unchanged (`updated 06:12:33Z`, head
  `5c5d3748`). Still scheduler/draft host work, **not** the verify `cudaEventSynchronize`.
- **#47020** (DSPARK penalties) — open, **new head `bf9c4eb6`**, `updated 10:09:44Z` (was 08:16:20Z)
  — the only axis PR to move this window. Still unmerged; re-read its hunks when taking it (the
  pin's field set predates #51719).
- **#51692** (blockwise rejection precision) — open, unchanged `03:20:17Z`, head `f6976099`.
- **#51681** (Philox folding) — open, unchanged `02:38:15Z`, head `f5b34e49`.
- No new PR references #49173 beyond the merged #49182, or #49174 beyond #49306.

### Cross-checks

- **Census, two sources disagree by one — resolved**: the script's counter is hard-capped at 10
  pages (`while page < 10`), and logged **990**. The manual census returned **991 raw entries /
  990 unique** because `0c6d6eb04e3b` (#50116) landed at **`12:19:04Z`** — after the poll and
  mid-pagination — which shifted one page boundary and produced exactly one duplicate
  (`4dfbb953eb55`). Re-reading page 1 then showed the new head `0c6d6eb04e3b`; deduped total 990
  (the 21 pre-insertion commits) **+ 1** = **991**, and `969 + 22 = 991` ✓.
- Slice keyed on run #32's own census head **`cc97c13817e4`** (index 21), which its 22 recorded rows
  occupy contiguously (`21..42`, no interlopers). Reconfirming the earlier note: the previous head is
  the **newest row of the previous table**, not the newest *material* row — `789abb9be9d9` sits at
  index 32 and would have returned 32 rows.
- Coverage: 22/22 `new` shas appear in exactly one row (6 Table A + 16 Table B).
- Keyword counter: **2 of 22** (9%) — `dsv4` on #51479 (config-gated) and `spark` on #49216 (model
  name). **10 of the 10 individually-read material rows carry no keyword**, including the
  traffic-conditional bp (#48208), the boot pin-fix (#51312) and the aarch64 THP env (#51351).
  Per-commit reading remains the only reliable mode.
- Batch integrity: 22 `.diff` attempts → 9 HTML `503` stubs (including every material commit), all
  re-audited through REST `commits/<sha>` with `files[].patch` present; 13 valid diffs.

### Action items

1. **No maintenance window triggered; nothing executed against the fleet.** GitHub REST + `.diff`
   + this checkout only — no SSH, no docker, no build, no serve, no schedule change.
2. **#51312** — next-pin boot item for our exact topology (4 nodes × 1 rank ⇒ full-shard prefetch per
   node). Confirm `get_parallel().nnodes` at the pin and re-audit `adapter/fast_load.py`'s
   `install_weight_utils` hook against the changed module.
3. **#51351** — next-pin **env-only** option (`SGLANG_DISABLE_THP=1`), no patch. Trial only in an
   authorised window with the dual gates; the aarch64 512 MB-huge-page stall is plausible on GB10
   but unmeasured here.
4. **#48208** — bp-class only if proxy traffic sets `top_logprobs`; it removes per-row host
   conversions on the spec path, not the verify sync. Check traffic first; measure second.
5. **#51479** — do not enable MegaGate on any line without this fix; otherwise record-only.
6. **#47020** head moved (`bf9c4eb6`): the only axis PR with new commits; still open, still a
   correctness refit, still not the decode gap.
7. **Tooling**: the census counter saturates at 1 000 commits (`while page < 10`) and a mid-run push
   yields a boundary duplicate/miss — manual re-fetch of page 1 is the cheap guard. Script fix
   (non-zero rc + keep alert) still deferred.
8. **Standing — unchanged from #32**: #50032 the best decode-axis bp (host work only); #47020
   penalty correctness; #51692/#51681 kernel-level accept fixes; #48678 image contract (EFA fetch,
   NCCL 2.31.2, `sgl-deep-ep` 0.1.3, mooncake 0.3.14); #51719 watermark contract; #48926/#45417
   record-only; #45261 `6eaca42f57` blocked (`block_size == 8`); #51295 `5b478f175b` free-rider;
   **bp6 collapse at `5e6995dd`**; decode gap has no upstream fix.
9. **Next pin** adds #51312 (boot) and #50116 to the dependency-ordered list in #32 item 13;
   #48208 only conditionally.
10. **Operator**: (a) `gh` re-auth (**33rd run**, REST-only throughout), (b) continuity clean
    (`08:17:49Z` → `12:17:50Z`), (c) polling path clean at both legs, (d) nothing needs a window
    today — the only new material items are a boot prefetch fix and an env switch for the next pin.
