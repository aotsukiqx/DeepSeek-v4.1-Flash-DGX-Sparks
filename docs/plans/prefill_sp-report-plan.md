# prefill_sp Structural Re-port Plan (2-4 sessions, the main optimization lever)

> Expected: +18% cold prefill + ~8% decode (historical measurements on old pin).
> This is the LARGEST remaining optimization — recovering it should push the mig
> line well above the production baseline.

## Background

prefill_sp implements prefill sequence parallelism: it shards the prefill computation
across TP ranks more efficiently than the stock path, reducing per-chunk wall time.

On the old pin (620c76f2) it was worth **+18% cold prefill** (2921→3605, measured
2026-10-14). It has source-hash gates on 16 engine functions (prefill_sp.py `_EXPECTED`)
that ALL DRIFTED on 7d996300 because the upstream mHC/MegaMoE wave renamed the API surface.

## Drift analysis (already measured, 2026-10-24)

### Hash drift: 0/11 same, 9 drifted, 2 MISSING

| Old function (620c76f2) | Status on 7d996300 | Lines (new) |
|---|---|---|
| `v4.DeepseekV4Model._forward_layers_hc_pre_from_prev` | DRIFT | 284 |
| `v4.DeepseekV4Model.forward` | DRIFT | 22 |
| `v4.DeepseekV4DecoderLayer.forward_hc_pre_from_prev` | DRIFT | 77 |
| `v4.DeepseekV4DecoderLayer._hc_combine` | **MISSING** | — |
| `v4.DeepseekV4DecoderLayer._hc_mix_stats` | **MISSING** | — |
| `v4.DeepseekV4DecoderLayer.hc_post` | DRIFT | 8 |
| `v4.DeepseekV4DecoderLayer._run_moe_ffn_dp_sync` | DRIFT | 34 |
| `v4.MQALayer.forward` | DRIFT | 456 |
| `v2.DeepseekV2MoE.forward` | DRIFT | 82 |
| `v2.DeepseekV2MoE.forward_normal` | DRIFT | 162 |
| `engram.Engram.forward` | DRIFT | 39 |

### Where the missing methods went

`_hc_combine` and `_hc_mix_stats` were absorbed into the new `op_*` method family:
- `op_combine_a` / `op_combine_b` (replaces `_hc_combine`)
- `op_mhc_postprocess` / `op_mhc_prepare_attn` / `op_mhc_post_attn_pre_mlp` (replaces `_hc_mix_stats`)
- Plus new: `_can_fuse_attn_mhc`, `_can_fuse_ffn_mhc`, `refresh_mhc_norm_weight_cache`

## Re-port strategy

### Phase A: Read + classify (1 session, offline)

For each of the 9 drifted + 2 missing functions:
1. Read the OLD source (from 620c76f2, already have hashes)
2. Read the NEW source (from 7d996300, candidate tree at `/tmp/cand7`)
3. Classify changes: cosmetic (docstrings/renames) vs structural (logic changes)
4. For MISSING methods: map to the new `op_*` equivalents
5. Output: a function-by-function port plan with effort estimates

Key questions to answer:
- Does the new `op_*` family already implement what prefill_sp's wrappers do?
  (If yes → prefill_sp might be partially OBSOLETE, reducing port work)
- What is the new `_run_moe_ffn_dp_sync` shape? (It's now 34 lines vs larger before)
- How does `MQALayer.forward` (456 lines) differ from the old version?

### Phase B: Port + test (1-2 sessions)

For each function in order of dependency:
1. Update `_EXPECTED` hash after confirming source equivalence
2. For MISSING methods: write NEW hook logic targeting the `op_*` API
3. For DRIFTED methods: re-port the wrapper logic to the new source shape
4. After each function: in-container import chain test (must stay CLEAN)
5. After all functions: bit-identity check (`DSV41_MOE_B12X_NEXT_DETERMINISTIC=1` greedy match)

### Phase C: Boot + measure (1 session)

1. Add `DSV41_PREFILL_SP=1 DSV41_PREFILL_SP_FP8=1` to env
2. Boot window (auto-rollback)
3. Verify boot log shows `PREFILL_SP shard: chunk of N rows -> shards of M`
4. Probes: cold 600k ×2 + decode ×2 + conc6
5. qeval paired (BROKE=0)
6. Compare vs: first mig (3953) + config arm result + production baseline (4101)

### Phase D: Full A/B decision

With all three optimizations active (prefill_sp + hc_fused + config arm):
- cold 600k target: **≥4500** (18% over first mig measurement)
- decode target: **≥44** (parity with production)
- conc6 target: **≥4100** (both above production AND above first mig)
- If met → **ADOPT the mig line as new production**

## prefill_sp internal structure (for the port)

The adapter wraps these entry points:
- `install(module)` — called when `sglang.srt.models.deepseek_v4` imports
- `_load_modules()` — imports v4, v2, engram, hcn, linear (module refs)
- `check_engine()` — validates all 16 hash expectations
- The actual SP logic lives in the wrapped function bodies:
  - Shards prefill rows across TP ranks
  - Manages communication buffers
  - Handles FP8 quantization for sharded projections (when _FP8=1)
  - The `_tail_rows` compatibility layer for attention backend integration

## Dependencies on other adapters

prefill_sp interacts with:
- `hc_fused` (SP requires hc_fused's SP_LOGICAL_ROWS=True gate) — port hc_fused FIRST
- `engram_backend` (SP reads engram row boundaries) — already fixed for 7d996300
- `block_verify` (verify rows use SP's row mapping) — config arm turns it ON
- `fast_load` (SP's weight loading interacts with fast_load slices) — already compatible

## Risk assessment

- **High value**: +18% cold prefill is the single largest remaining optimization
- **Medium risk**: 11 functions to re-port, but the ADAPTER logic (our code) stays
  the same — only the WRAPPER TARGETS change (upstream function signatures)
- **Mitigation**: env-gated (DSV41_PREFILL_SP=0 disables), boot-time hash check,
  bit-identity verification available
