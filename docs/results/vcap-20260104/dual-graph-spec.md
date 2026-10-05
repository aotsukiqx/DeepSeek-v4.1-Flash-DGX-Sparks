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
