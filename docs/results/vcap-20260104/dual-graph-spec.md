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

## W1g reproducibility failure (2026-10-05, final round of the session)

The W1g hotpatch round captured the full narrow family (bs 1-8) and then died at a
warmup-side draft view ([0, 8, -1]). A FRESH serve of the identical committed code
instead fails DURING capture at _accept (9 vs 6, dim 1) again. Same code, different
outcomes -> the swap-in-flight buffer strategy is stateful and order-sensitive (the
first narrow call's clone cache, which buffers exist at which point, restore paths).
Verdict recorded for the next session: abandon swap-in-flight; allocate per-family
buffers eagerly at install (clone BOTH families' buffers up front, select by width
from the batch: width = input_ids.shape[0] // bs), and make every epilogue derived
value (stride, gamma, buffers) a function of that inferred width instead of mutating
instance state. That redesign also IS the W2 replay routing. Production restored and
verified after every round of this session; all code env-gated off in production.

## W2 chain progress (2026-10-05, session close)

Width-inference redesign (b958442) REPRODUCES the full narrow capture on fresh serves
(order-sensitivity solved; the families derive from constructor state only). The
warmup chain then peeled two more sub-layers, each solved by a flag-guarded swap:
folded sampler bs-derivation + gamma view (cad4b6de; the [0,8,-1] view error), and
next surfaced `corrected_out[: bs*gamma]` / markov_head.sample_block's internal
width coupling (8 vs 6, dim 1, dspark_draft_sampler.py:181) - the current frontier.
Remaining: that markov-head sub-layer, then the W2 selection wiring
(variants.select + replay routing), then the R1 oracle before any trust. All
adapters env-gated; production restored and verified after every round.

## Draft-side layout conclusion (2026-10-05, session final round)

The markov-head frontier dissolved into a DESIGN finding: the narrow capture's warmup
feeds the 8-wide DRAFTER a 6-row batch. compute_base_logits has an internal
[.., gamma=8, ..] buffer (the current 8-vs-6), and deeper in, sample_hidden =
hidden.view(bs, 5, -1) over 6 anchor-inclusive rows is semantically wrong regardless
of attribute swaps (it would reinterpret 6x5120 as 5x6144). Correct architecture:
slice at the capture_hook level (draft_worker_common:151) - the draft sampler
receives the FULL 8-row draft output while the verify side consumes the 6-row slice;
no attribute swapping inside the sampler at all. That moves the remaining work to one
well-defined seam (the hook that already calls both) and retires the sampler/markov
swap layers from the critical path. W2 selection routing and the R1 oracle remain
as specced. Production restored and verified; every adapter stays env-gated off.

## MILESTONE: healthy dual-graph boot (2026-10-05, commit f6973a81)

ROOT CAUSE of the sampler/markov "layers": the hooked runner class serves BOTH the
draft worker (width = draft query layout, 8) and the target verify runner (width 9);
the narrow loop had been shrinking the DRAFT graphs too. Guarding with
`model_runner.is_draft_worker` - only the target verify runner gets the narrow
family - made everything pass at once:

  HEALTHY + `captured narrow verify family rows=6 for bs [1..8]` + warmup clean.

Bench on the dual boot: code 134.77 tok/s (identical to the EXT arm, +12.3 % kept),
greedy deterministic (rep1=3=5, rep2=4 per prompt). OPEN ITEM: prose read 39.18 with
accept-len 2.91 (EXT arm was 49.9 / 2.76, uncapped 39.2 / 2.96) - single-boot data
inside gamma=8's known 39-50 boot-tactic band; needs an interleaved dual-vs-EXT A/B
next session before any conclusion (selection W2 is not wired, so prose still replays
wide graphs - the 39/50 difference is boot variance, not the narrow family). Next:
W2 selection wiring (variants.select) -> FORCE hook -> R1 oracle -> full gates.

## W2 selection wiring (2026-10-05, commit 6ea35ba): implementation

The width decision follows the live length verify_cap already computes: a step whose
batch-max live is <= 6 (i.e. no EXT extension) is built at width 6 and replays the
narrow family; any EXT step stays wide. Pieces (adapter/dual_graph.py):

1. Decision: `_decide_width(bs)` reads verify_cap's live buffer (rank-0 broadcast,
   rank-consistent). `begin_static_step` (worker, pre-forward) makes it early so the
   epilogue family state and verify_lens fill at the right stride; `run_non_compact`
   (installed BEFORE verify_cap's wrap -> inner) re-derives the same decision after
   vcap set live. A per-step D2H `.max().item()` is the one sync the routing costs;
   it waits on the draft confidence only, which the verify already waits on.
2. Batch at width: the inner run_non_compact truncates `verify_ids_2d[:, :6]`, swaps
   `verify_num_draft_tokens` 8->5, and slices the verify window
   (positions_2d/verify_cache_loc[_2d], request-major prefix). Everything downstream
   (seq_lens +5, draft_token_num, spec_info.num_tokens_per_req=6) derives from those.
3. Graph routing: `_resolve_attention_variant` returns the narrow label for a
   target-verify batch with rows == bs*6; `can_run_graph`/`load_batch`/`execute` run
   under a context that sets `captured_req_width = 6` (the engine's uniform-width
   replay invariant) and swaps the backends' `cuda_graph_metadata_of_bucket_and_bs`
   to the narrow store the capture now SAVES (`_dsv41_narrow_meta_store`) instead of
   discarding.
4. Epilogue families: `_apply_epilogue_width` (W1g eager-family design) applied at
   begin_static_step (step width) and read_accept (results live in the family the
   replay bound); `_static_epilogue` still derives from the batch width at capture.
   Wide restore applies the ORIGINAL wide family (captured graphs bind the
   constructor buffers - a clone would read stale).
5. verify_cap stride override: during narrow capture `stride_ovr=6` makes the
   dead-row remap bake STRIDE_R=6 (bs*6 rows), and the capture-time gate check uses
   the effective stride. Without it bs in {3,6} mis-activate (bs*6 % 9 == 0) with
   wrong req/pos arithmetic and the other bs skip the remap entirely.
6. R1 oracle (`DSV41_DUAL_GRAPH_ORACLE=N`): the first N narrow-eligible steps run
   BOTH families end-to-end (wide reference first, then narrow - the comparison uses
   forward logits only, which accept-length caps cannot contaminate) and print
   per-step argmax agreement + max|dlogit|, then an aggregate PASS line (all
   argmax-equal, max|dlogit| < 0.05). This is the silent-corruption trap check the
   [4,8] capture signature demanded. FORCE=narrow|wide overrides selection for A/B.

Fallback guard: if a narrow forward ever reports can_run_cuda_graph=False, the step
re-runs wide (loud print), keeping downstream shapes on the stock path.

Boot env for the dual arm (over production .env.tp4): DSPARK_BLOCK_SIZE=8,
CUDA_GRAPH_MAX_BS_DECODE=8 (9-row graphs beyond bs8 OOM - gamma8-probe boot 1),
EXTRA_CONTAINER_ENV += DSV41_DUAL_GRAPH=1 DSV41_DUAL_GRAPH_ORACLE=12
DSV41_MOE_B12X_NEXT_ROWS=6,8,9. Backup .env.tp4.pre-w2dual.

## W2 COMPLETE (2026-10-05 evening, commits 6ea35ba..5937239): selection works; EXT found dead under the c32 production env

Mechanism (all verified on the fleet): healthy boot + narrow family bs[1..8] captured +
REAL selection live (verify_cap's live buffer drives it: max live <= 6 -> narrow, EXT
step -> wide) + R1 oracle PASS on real steps (6 steps argmax_agree=1.0000,
max|dlogit|=0.0000 - the narrow family's forward is bitwise the wide reference) +
within-boot greedy deterministic + smoke clean + prose 52.97 vs pure-EXT 53.33.

Eight fix rounds to get here (each a hot-rebuild cycle, lesson-grade):
1. late-binding closure: three runner wraps shared one `orig_m` -> execute<->load_batch
   infinite recursion. Fix: `_bind(orig_m)` factory.
2. `_oracle_due` lost in an edit round -> NameError at first verify step.
3. VerifyWindow is a frozen msgspec.Struct: rebuild via `type(vw)(...)`, never mutate.
4. The eager pre-graph metadata path (prepare_for_verify -> backend.
   init_forward_metadata) reads the backend's spec width + metadata store OUTSIDE the
   runner wraps: narrow_call swaps both (like capture does).
5. `verify_num_draft_tokens` counts ROWS INCL. the anchor (stock gamma=5 -> 6; the
   engram layer and DFlashVerifyInput derive width from it): narrow passes 6, not 5.
6. It is STEP-SCOPED: set at entry, consumed by the worker tail (logprob chain_stride,
   GenerationBatchResult.speculative_num_draft_tokens unflattens out_tokens). A
   stride-9 read over 6-wide out_tokens duplicated committed tokens ("4242").
7. The oracle's wide reference must disarm the in-graph commit inject
   (begin_static_step(bs, False)) - double inject advances the injector twice.
8. FAMILY NORMALIZATION (the deep one): the W1g width-inference minted a NUMERIC
   9-clone at wide capture, so wide graphs bound the clone while W2's reads applied
   the constructor family - stale accepts, runaway streams. W1 had "worked" by riding
   the clone on BOTH sides. Fix: the wide stride maps to the constructor family itself;
   wide graphs now bind constructor buffers, reads agree.

## BLOCKER FOUND (not W2's): gamma=8 EXT extension is dead under the c32 production env

Pure-EXT control (DSV41_DUAL_GRAPH=0, production .env + BLOCK=8/GRAPH_BS=8):
code 102.4 tok/s, accept mean 2.65, extended 0/36 steps - cum5 never reaches 0.5.
FORCE=wide dual arm: same (92.6, 0/37). Real-selection arm: ~9% extend on one probe.
The 134.8 (gamma8-probe boot 3) and 134.77 (W1 milestone) were both measured BEFORE
the c32/MRR=32 production adoption; that env's gamma=8 extension behavior was never
validated. Excluded so far: TUNE_ROWS=6:5+6 raced plans (removed, still dead) and the
dual machinery itself (pure-EXT also dead). Remaining suspects: the c32 GRAPH_BS
tier list (10..32 present at gamma=8 boot? then capped to 8), MRR=32, mem/pool
shifts. Consequence: under the current production env, gamma=8 offers NOTHING (code
102 < gamma=5 production 120; prose 53.3 ~= 57.3 within boot band) - and with EXT
dead, dual-graph's wide family never fires, so its value cannot materialize until
the extension regression is located and fixed. That bisect (one env diff per boot)
is the next session's first move; everything W2 is ready and waiting for it.

Fleet restored to production (gamma=5, MRR=32, raced tiers, dual off) and verified
(health 200, BLOCK_SIZE=5 in the running container, smoke "42").

## EXT-death bisect COMPLETE (2026-10-05 night): two independent factors + a structural finding

Boot matrix (all gamma=8, graphs bs<=8, production c32 line otherwise, ab_bench medians):

| boot | arm | code tok/s | extend rate |
|---|---|---:|---|
| b1 | dual OFF, ROWS derived (8,9) | 132.41 | 6/33 (18% of code window) |
| b4 | dual ON, FORCE=wide | 132.41 | 6/34 - hooks are innocent |
| b7 | dual ON, FORCE=narrow | 116.36 | 0 (accept pinned 5.1-5.8) |
| b2/3/5/6 | dual ON, real selection | 91-96 | ~0-12% |

Factor 1 (SOLVED): DSV41_MOE_B12X_NEXT_ROWS=6,8,9 (added for the narrow family's MoE
plans) poisons the draft hidden distribution -> conf head cum5 drops below the EXT 0.5
gate -> extension dies (0/36, code 102). With ROWS derived (8,9) extension revives
(code 132). The narrow family does NOT need ROWS=6 (ladder pads 6-token Ms up to 8/9
plans; correct, minor waste at small bs).

Factor 2 + STRUCTURAL FINDING: even with extension healthy, real selection loses:
the 5-draft cap breaks the draft feedback loop on code (long match blocks truncated ->
weaker n-gram/engram matches next step -> lower conf -> narrow self-lock). Steady-state
all-narrow code = 116 (-12% vs wide 132); narrow/wide ALTERNATION = 91-96 (-30%,
switching cost not yet localized: metadata store swap per step, graph L2/TLB churn,
or measurement-window effects). Prose gain is real (39.4 -> 52.6-53.7, +34%).

ECONOMIC VERDICT: under mixed traffic the dual-graph narrow=6 design does NOT beat
gamma=5 production (prose 57.3 / code 120): it trades prose +34% for code -12%..-30%.
The per-position survival data said positions 1-5 are safe to cap for prose, but code
straddles the 5/6 boundary - the cap must sit at 7+ rows to leave code's accept
distribution (7.3) intact, which shrinks the prose saving. Options for a future round:
(a) NARROW_ROWS=7 sweep (prose keeps ~+25%?, code unpinched at accept<=7 - needs
  positions-6-7 survival recheck), (b) per-request family pinning (code-heavy reqs
  stay wide for their lifetime - no alternation, no feedback break), (c) accept the
  finding: gamma=5 stays the mixed default and dual-graph remains the gated
  patterned-continuation line.

All W2 machinery is correct and verified (oracle bitwise, greedy deterministic,
healthy boots throughout) - the blocker is the workload economics, not the plumbing.
Fleet restored to production (gamma=5) and verified.
