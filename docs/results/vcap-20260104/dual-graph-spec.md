# Dual verify-graph families (gamma=8 universal): implementation spec

Goal: at a `DSPARK_BLOCK_SIZE=8` boot, every bs captures TWO verify graphs (rows 6 and
9); a step whose batch max live length is <= 6 replays the 6-row graph (prose pays
nothing for the widened window), an extending batch replays the 9-row graph (code keeps
the measured +12.3 %). Expected end state: mixed-traffic default-on, qeval-clean.

All anchors verified on the serving image (sglang dsv4.1 branch f80c91a4b) on
2026-10-04. Mechanism discovery trail in `gamma8-probe.md`.

## Verified engine facts (the load-bearing anchors)

1. Graphs are keyed by `ShapeKey(size=bs, ..., attention_variant=...)` —
   `decode_cuda_graph_runner._make_graph_key` (line ~543). Width is NOT in the key; it
   is baked into each captured graph via `capture_one_shape`'s
   `num_tokens = size * self.captured_req_width` (line ~1116, the single-width
   hardcode).
2. A per-bs multi-graph mechanism already exists and DSV41 already uses it:
   `self.attention_graph_variants = create_attention_graph_variants(hf_config) or
   create_dsv41_candidate_graph_variants(model_runner, self.capture_forward_mode,
   self.captured_req_width)` (line ~302). Capture loops
   `for bs in capture_range: for label in variants.capture_labels:
   capture_one_shape(bs, ..., attention_variant=label)` (line ~1078+); replay selects
   with `variants.select(forward_batch)` (line ~557, inside `can_run`'s key build).
   Follow this pattern for a width variant.
3. The per-step Python seam where the live length is known before the verify forward:
   the DSpark planner computes the confidence during propose
   (`dspark_draft_sampler.__call__` -> `confidence_fn` -> planner
   `compute_confidence_tensor`); verify_cap's `_state["conf"]` holds it from there.
   The verify forward batch is built by the dspark worker ABOVE
   `TargetVerifyExecutor.run_non_compact` (verify_cap already wraps that entry and
   receives the built `batch`/`verify_ids_2d`), so width selection must hook the
   worker's batch-construction site, not run_non_compact.
4. The confidence path at gamma=8 already works via the adapter shim (commits
   e1fb298..90d7ad4): `[bs, 8]` zero-padded past position 5; `set_live_from_confidence`
   returns live in {2..6} or 9 (EXT gate).

## Work items

W1 — runner: capture a second family. Extend `capture_one_shape` with a width override
(num_tokens = size x width) and register a "narrow" attention-variant label per bs
(either grow `create_dsv41_candidate_graph_variants` or add a parallel variants object).
`variants.select` must read the step's family: thread the decision via a module-level
hook the adapter sets (verify_cap exposes max live / family from `_state`), because
`select` only receives `forward_batch`.

W2 — worker: build the verify batch at the chosen width. In the dspark worker's verify
step (the method containing grammar-mask build + `accept_and_finalize`, around
dspark_worker_v2.py:700-850): before constructing the ForwardBatch, read the family
(max live <= 6 -> width 6, else 9), truncate `verify_ids_2d[:, :width]`, the draft
block rows, positions/seq_lens offsets, and the KV-injector writes to that width. The
accept epilogue buffers are per-graph (each captured graph owns its AcceptOuts), so a
6-row replay caps acceptance at 5 drafts naturally. The draft graph stays 8-row (3
layers; the parallel head emits all 8, truncation is free).

W3 — adapter: b12x capacities for both families: rows {5, 6, 8, 9} x GRAPH_BS (the
capacity builder currently derives (BLOCK, BLOCK+1); extend to a set). verify_cap: the
family decision helper (max live over the batch) exported for W1/W2.

## Risks / open checks

- R1: the folded propose->verify sequencing must leave the Python seam where W2 hooks;
  verified_cap's per-step hooks running there are the evidence, confirm once in code.
- R2: KV injector semantics for truncated rows (draft KV written for 8 positions,
  verify commits <= 5): the engine already rolls back unaccepted positions at gamma=5
  (accepted < gamma is the common case); same machinery applies.
- R3: memory: two families x bs list. bs <= 8 both families measured to fit (16 GB free
  at one family bs <= 8); bs 9-16 second family needs one fitting boot (KV pin -~1M).
- R4: math_m9 flipped under gamma=8 numerics (qeval BROKE 1). Re-run qeval paired on
  the dual-graph arm and inspect math_m9 against the serial reference specifically;
  if it is a rounding near-tie, document; if systematic, stop.
- R5: within-boot greedy repeatability must hold on the dual-graph line (it did on
  every gamma=8 boot so far).

## Gates (all must pass for default-on)

Boot gate (both families armed lines) -> ab_bench three-arm (prod gamma=5, gamma=8
EXT, gamma=8 dual) with accept-len join -> qeval paired BROKE=0 -> tier_bench c16 warm
(no aggregate regression) -> 190k needle + 4-concurrent long context -> within-boot
greedy repeat. Acceptance bar: code >= +10 %, prose >= -2 % vs production.

## Sequencing

W3 + W1 capture-only first (second family captured, never selected - validates memory
and capture), then W2 selection on, then gates. Est. 1-2 focused days.

## W1 first attempt (2026-10-05 night): capture crashes in the DSA indexer

adapter/dual_graph.py (commit 274e3fe) swaps captured_req_width during a second
capture pass inside _capture_one_stream. The narrow (6-row) capture dies in the
TARGET verify forward at deep_gemm's fp8_fp4_paged_mqa_logits with
`_batch_size == batch_size` (attention.hpp:530) - the DSA indexer's paged-MQA row
count disagrees with its request/schedule metadata. The width couples beyond the
runner attribute: the indexer's spec-row layout (context_lens/block_table/
schedule_meta per verify row) is built for the wide family. Next diagnostic (cheap,
hotpatch-able on a crashed-env container): per-bs try/except in the capture loop +
print the dims capture_prepare produced and which bs fail (all vs subset); the fix
will be to parameterize the indexer's per-req row layout by the width override too
(deepseek_v4_backend._low_ratio_index_topk_decode path). Production was restored and
verified after the crash.

## W1 diagnostic round (2026-10-05, per-bs try/except hotpatch): architecture verdict

Prints: every narrow capture fails with a wide-width view of narrow data -
`shape '[bs-2, 8]' is invalid for input of size bs*6` (bs 1,2,3,5,6,7) - and the two
"successes" (bs 4, 8) are SILENT CORRUPTION: 4*6=24 and 8*6=48 divide by 8, so the
wrong view is numerically satisfiable. Root cause: the folded draft sampler is
captured INSIDE the verify graph and is 8-wide by construction at this boot; its
internal views (markov stack, sampler buffers, KV inject) all assume the batch rows
match the draft block width. Conclusion: the width swap must reach the folded
sampler inside the captured region (propose 8, verify first 5 - the sampler's
proposal/verify boundary needs a width parameter), which is deeper engine surgery
than the runner-level capture the spec assumed. W1 PARKED at commit 274e3fe
(adapter committed, DSV41_DUAL_GRAPH unset in production - inert; the diagnostic
also proved per-bs try/except must never ship given the silent-corruption mode).
The usable widened line remains the env-gated gamma=8 EXT shim (code +12.3 %).
Revised estimate for a universal line: 3-5 days engine work parameterizing the
folded sampler's proposal/verify boundary.

## Per-position acceptance at gamma=8 (2026-10-05, vcap-g8b.bin, 2840 steps)

Code survival through the untrained positions: 0.99/0.96/0.95/0.94/0.96/0.89/0.84/0.77
- positions 6-8 extrapolate at 77-89 %. Prose conditional survival RISES far out
(0.75-0.77 at positions 5-8 among the ~25 % of steps that reach them - repetitive
stretches), but E[tok] only moves 2.33 -> 2.45 from gamma=5 to 8. With the measured
per-row step cost (prose 4.8 %/row, code 6 %/row, from the three-arm anchors), the
gamma sweep nets: g6 +1.5 %, g7 +1.5 %, g8 +0.3 % on a balanced mix - all below the
2 % bar. Verdict: no single gamma wins mixed traffic. gamma=5 stays the mixed default;
gamma=8 stays the env-gated code-heavy line (+12 %); the dual-graph surgery (folded-
sampler width parameterization, above) remains the only universal path.

## W1 leak point pinned (2026-10-05, traceback round)

All narrow captures fail at ONE site: deepseek_v4_backend.py:_low_ratio_index_topk_decode
-> indexer.fp4_paged_mqa_logits -> deep_gemm `_batch_size == batch_size`. The width
leak: the backend caches `self.speculative_num_draft_tokens = get_spec()
.speculative_num_draft_tokens` (line ~1252, = 9 at this boot) and extends
`seq_lens_cpu` by it and sizes block tables/metadata with it (lines ~1323, ~1340-41),
while the narrow capture's query rows are 6/req. Fix shape: during the narrow capture,
swap BOTH the runner's `captured_req_width` AND the spec width the backend sees (the
getter `get_spec()` - if it is a settable global/contextvar, one swap covers every
reader; else swap the backend instance attr before capture_one_shape and confirm no
metadata was pre-derived). Diagnostic round also re-learned: hotpatch must reach ALL
FOUR containers (a worker running the unpatched version kills the boot at a barrier
before the head's narrow pass runs) and clear the adapter __pycache__ entry alongside.

## W1b (2026-10-05): layer 1 cleared, layer 2 mapped

Swapping the backend's `speculative_num_draft_tokens` during narrow capture (commit
b357534) clears the deep_gemm indexer assertion. Next failure:
`init_forward_metadata_out_graph` (2417) -> `replay_cuda_graph_metadata_from` (2784)
-> `copy_` (1098): the persistent target-verify metadata buffers the backend
allocates AT __INIT__ sized by num_draft_tokens=9 are written with 6-wide metadata.
These buffers are shared cross-graph state - the fix is per-width buffer variants
(real engine state surgery), not an attribute swap. Layered width-coupling map so
far: (1) runner captured_req_width [solved], (2) backend speculative_num_draft_tokens
-> indexer seq_lens/block layout [solved], (3) persistent verify metadata buffers
[mapped, needs per-width variants], (4) folded sampler proposal/verify boundary
[expected, W2 territory]. The 3-5 day engine-surgery estimate stands; three of the
layers are now precisely located.

## W1c/W1d (2026-10-05): layers 3 cleared, layer 4 reached

W1c (commit adf06ef): narrow capture stores its verify metadata in a fresh dict -
first cut crashed KeyError TARGET_VERIFY (the outer dict is directly indexed by
bucket; keep the bucket keys). W1d (hotpatch, committed here): swap to
{k: {} for k in store} - bucket keys preserved, per-bs slots fresh. Result: the
metadata copy_ layer PASSES, the deep_gemm layer stays passed, and the next failure
is `shape '[1, 9]' invalid for input of size 6` - the draft/folded side's per-request
9-wide view on the 6-row batch: layer 4, the folded sampler proposal/verify boundary
the spec flagged as the hard one (W2 territory, also needs the per-family replay
routing for the metadata stores). Layer map final state: runner [solved], indexer
spec-width [solved], metadata stores [solved], folded sampler [next]. Iteration
recipe that worked: per-layer hotpatch cycle, ~15 min each (crash -> cp to ALL FOUR
containers -> restart -> read traceback -> swap the named coupling).

## W1e (2026-10-05): stride view cleared; _accept is the next sub-layer

The verify epilogue's `input_ids.view(bs, self.stride)` (dspark_verify:615) now follows
the narrow captures (install_verify_hook, commits 8a56414+bbd538a; the first commit
briefly shipped sitecustomize without the function - fixed immediately). With the
stride swap in place the plain build reported `captured narrow verify family for bs
[4, 8]` on a healthy boot - but [4, 8] is exactly the silent-corruption pair signature
(6*4 and 6*8 divide by 8), so those are NOT trustworthy successes. The diagnostic
merge then surfaced the real next sub-layer: `_accept` (dspark_verify:766) raises
`tensor a (9) vs b (6) at dim 1` - its own width-coupled buffers (AcceptOuts /
cutoff arrays sized 9) - and the raise escapes the capture try/except (it fires on a
post-capture warmup/replay path), killing the boot. L4 sub-layers: epilogue stride
[solved], _accept buffers [next: size by actual width and keep the exception inside
the capture loop - likely also needs the accept path to slice verify_ids to width].
Layer map: L1 runner [solved], L2 indexer [solved], L3 metadata stores [solved],
L4 epilogue stride [solved], L5 _accept buffers [located].

## W1 MILESTONE (2026-10-05, W1f/W1g): full narrow-family capture achieved

W1f: _accept also mixes self.gamma into BuildOutTokens - the epilogue swap now covers
{stride, gamma}. W1g: the stride-wide instance buffers (out_tokens_buf [max_bs, 9])
get per-family residents (cloned at 6, kept referenced so captured graphs bind 6-wide
storage). RESULT: `[dual_graph] captured narrow verify family rows=6 for bs
[1, 2, 3, 4, 5, 6, 7, 8]` - the FULL list, the trusted-success signature (not the
[4, 8] corruption pair). The boot still fails AFTER capture on the warmup/replay
path: `shape '[0, 8, -1]' invalid for size 30720` - a draft-side 8-wide view outside
the confidence shim's guard, on a path the capture try/except does not cover. That is
W2 territory (the family must follow through replay, including this view). Layer
count: L1-L5 solved (runner, indexer, metadata stores, epilogue stride+gamma,
stride-wide buffers); next failure is replay-side. R1 oracle still gates any trust in
the captured family - build it before wiring selection.
