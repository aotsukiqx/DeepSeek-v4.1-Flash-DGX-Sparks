# Plan: prefill_sp Re-Port to 7d996300 (Main Effort, 2-4 Sessions)

> Goal: rebuild the prefill sequence-parallel adapter on the new pin's CUDA op_* seams.
> Expected recovery: +18% cold prefill + ~8% decode = mig arm surpasses production baseline.
> **Phase A (classification) COMPLETE. Phase B (hash tables + dual-pin + CUDA-safe
> install) COMPLETE 2026-10-26, commit 6592aee — verified in-image on BOTH pins
> (7d996: 28/28 MATCH + CPU tests ok; 620c: probe+check PASS). Phase C remains.**

## 0. CORRECTION to §2b (2026-10-26, later session)
`hc_pre_from_prev_sublayer` is **True** for our model: the pin's config CLASS
`configs/deepseek_v41.py:47` defaults it True (raw config.json has no hc keys —
checking json alone is misleading). So on CUDA the driving loop IS
`Model._forward_layers_hc_pre_from_prev` (269 lines, DRIFTED but ACTIVE); inside it,
per-layer dispatch is `if _is_cuda: layers[i](...)` (CudaDecoderLayer.forward, 37
lines) else `layers[i].forward_hc_from_prev(...)` (Platform). The adapter's PRIMARY
wrap point (the Model loop) survives; `_sp_forward_layers` must be rewritten to the
new per-layer signature (no prev_pre/precomputed_attn kwargs; hc state via
residual_batch + state.materialized at loop end).

## 1. Problem Statement

prefill_sp (`adapter/prefill_sp.py`, 1215 lines) hash-guards engine anchors
(20 MAIN + 4 FP8 + 3 TAIL on 7d996; 16+1 legacy on 620c, probe-selected). On
7d996300 most drifted; the CUDA layer architecture was restructured around declared
HC boundaries. The drift guard raises at serve time.

## 2. Phase A Findings (2026-10-26, verified in mig image)

### 2a. Class architecture changed (v4.py split into 5 modules)

- `deepseek_v4.py`: `DeepseekV4DecoderLayerBase` (3070) → `DeepseekV4CudaDecoderLayer` (3517)
  → `DeepseekV4PlatformDecoderLayer` (3704, = NonCudaDecoderMixin + Base);
  alias `DeepseekV4DecoderLayer = Cuda if _is_cuda else Platform` (3708)
- `deepseek_v4_platform_stage.py`: the NonCuda mixin (owns `forward_hc_pre_from_prev`,
  `_run_moe_ffn_dp_sync` — **Platform-only, ABSENT on CUDA class**)
- `deepseek_v4_mhc.py`, `deepseek_v4_replay_tail.py`, `deepseek_v41_vit.py` (vision tower)

⚠️ Gotcha: any check run with `CUDA_VISIBLE_DEVICES=` resolves the alias to the
Platform class — the WRONG target for our fleet. Always verify with CUDA visible.

### 2b. Runtime path on GB10 (CUDA)
Model config has NO hc keys → `hc_pre_from_prev_sublayer=False` → `_forward_impl`
takes the **else-branch**: plain layer loop with `residual_batch.start(forward_batch)`
+ `CudaDecoderLayer.forward` (37 lines: declared boundary stages) + op_* seams +
`AttentionTransport`/`MoeTransport`. The old `Model._forward_layers_hc_pre_from_prev`
path is now Platform-only. `Model.forward` (135→20 lines) wraps `_forward_impl` in
`mega_gate_forward_scope`.

### 2c. Anchor drift classification (old vs new source, function-level diff)

| Class | Anchors | Action |
|---|---|---|
| **MATCH (7)** | EngramEmbedding._lookup; hcn._hc_combine_norm_prefill; FP8: MQALayer._compute_kv_bf16, _compute_kv_to_cache, accepts_mxfp8_swizzled_input; TAIL: LateLayerTail.rows, real_rows, _tail_rows | keep |
| **EASY drift (4)** | EngramEmbedding.forward (sim .92, 4Δ); hcn.hc_combine_norm (.90, 6Δ); linear.RowParallelLinear.forward (.94, 15Δ); Engram.forward (.60, 24Δ) | re-hash + touch wrappers |
| **MODERATE drift (4)** | MQALayer.forward (.91, 77Δ of 396); MQALayer._forward_prepare (.80, 28Δ); MoE.forward_normal (.84, 40Δ); MoE.forward (.66, 34Δ) | re-port wrappers onto new source |
| **REWRITE/REDIRECT (4)** | Model.forward (.10, 135→20); Model._forward_layers_hc_pre_from_prev (.25, 188→269 — Platform path now); hc_post (.12, 73→8 in Base); _hc_combine + _hc_mix_stats (→ op_*) | drop old hooks; re-target to op_* |
| **GONE on CUDA (2)** | Layer.forward_hc_pre_from_prev; Layer._run_moe_ffn_dp_sync | Platform-only; remove from _EXPECTED, replace with op_moe/transport hooks |

### 2d. NEW seam inventory (the Phase B hook surface)

| Seam | Size | Role |
|---|---|---|
| `CudaDecoderLayer.forward` | 37 lines | layer entry: boundary stages, declared residual r/w |
| `op_combine_a` / `op_combine_b` | 4 lines each | thin combine shims (old `_hc_combine`) |
| `op_gather_a` / `op_gather_b` | — | gather seams — **SP inject points** |
| `op_moe` | 8 lines | MoE boundary (old `_run_moe_ffn_dp_sync`) |
| `op_mhc_prepare_attn` | 32 lines | mHC stats chain entry (old `_hc_mix_stats`) |
| `op_mhc_post_attn_pre_mlp` | 15 lines | fused post-attn/pre-MLP stats |
| `op_mhc_postprocess` | 17 lines | mHC stats finalization |
| `AttentionTransport` / `MoeTransport` (hc_boundary) | — | all-gather/reduce transports — **SP inject points** |
| `residual_batch.start` | loop-level | CUDA residual batching state |
| `MQALayer.forward` | 391 lines (sim .91) | attention — most-preserved anchor |

The SP concept maps cleanly: upstream's transports/gathers ARE the comm seams where our
gather/scatter lived. The op_* methods are far thinner than the old 73-188-line
monoliths — cleaner wrap targets.

## 3. Revised Porting Strategy

### Phase B (1 session): seam scaffolding + easy/moderate anchors
1. Rebuild `_EXPECTED` table: drop 2 gone anchors, add op_* anchors
2. Port easy+moderate wrappers (8 anchors: Engram×2, hcn, linear, MQALayer.forward,
   _forward_prepare, MoE.forward/forward_normal) — mostly re-hash + adjusted internals
3. Read `_plan`/`_sp_forward_layers` (the SP core, ~lines 312-1100) and map each
   gather/scatter call site to its new seam (op_gather_*, transports, op_moe)
4. Unit-test the hash table via `scripts/verify/sp_drift_check.py` (extend for CUDA-visible)

### Phase C (1 session): SP core re-target
- Rewrite `_sp_forward_layers` to drive the else-branch loop shape (residual_batch +
  layer.forward) instead of `_forward_layers_hc_pre_from_prev`
- Wire SP row-sharding at: op_gather_a/b (attention input gather), MoeTransport (MoE
  boundary), op_combine (hc stats combine), Engram/wo_b full-row layers
- Keep EXACT/FP8/WKV_SHARD modes behind env as before

### Phase D (validation, 1 window)
- Boot mig arm with `DSV41_PREFILL_SP=1`
- First-chunk bit-identity (DEBUG=compare path), then qeval paired BROKE=0
- 600k cold + decode + conc6 vs baselines (cold 3953/4101, decode 37.8/44.0, conc6 4021/3884)

## 4. Success Criteria
- `sp_drift_check.py` (CUDA-visible) reports all-MATCH on the new anchor table
- Boot with DSV41_PREFILL_SP=1; qeval BROKE=0
- cold 600k ≥ 4500; decode ≥ 42; conc6 ≥ 4021

## 5. Key Files
| File | Role |
|---|---|
| `adapter/prefill_sp.py` | the adapter (1166 lines) — main edit target |
| `scripts/verify/sp_drift_check.py` | anchor drift checker (runs in mig image; **must run CUDA-visible** post-Phase-B) |
| `~/dsv41-srcdiff/{old,new}/` (Mac) | extracted sources for diffing: v4.py v2.py engram.py hcn.py linear.py backend.py |
| Head `/tmp/srcdiff.tgz` | same bundle on cluster |
| tests: `tests/test_prefill_sp.py` | unit tests (SimpleNamespace fixtures) |

## 6. New Anchor Hash Table (Phase B ready — computed 2026-10-26, CUDA-visible, image canary-mig)

```
KEEP (hash unchanged from _EXPECTED):
  engram.EngramEmbedding._lookup                 0a035cac21c4c305
  hcn._hc_combine_norm_prefill                   74f71ebf85b010ee
  v4.MQALayer._compute_kv_bf16                   00c4ab90b403d1e9   (FP8 table)
  v4.MQALayer._compute_kv_to_cache               678224fac8995d64   (FP8)
  v4.MQALayer.accepts_mxfp8_swizzled_input       5bb4fddfc5b329ca   (FP8)
  deepseek_v4_backend.LateLayerTail.rows         c58392d3ef651408   (TAIL)
  deepseek_v4_backend.LateLayerTail.real_rows    c78962c2e01e0e3d   (TAIL)
  deepseek_v4_backend._tail_rows                 54a3cbff5e674e84   (TAIL)

UPDATE (drifted — new hash):
  v4.DeepseekV4Model._forward_layers_hc_pre_from_prev  279b059d580002a0  (Platform path; hash-check harmless)
  v4.DeepseekV4DecoderLayer.hc_post                    3b26f9cbb485778a  (now 8 lines in Base)
  v4.DeepseekV4Model.forward                           5a66a844669ee55d
  v4.MQALayer.forward                                  b0d19428fed97020
  v4.MQALayer._forward_prepare                         e45619be35c3ba0f  (FP8)
  v2.DeepseekV2MoE.forward                             f3a64ad5dbad841b
  v2.DeepseekV2MoE.forward_normal                      8dbf1872ed9ed3b8
  engram.Engram.forward                                1d5bacd6d43c08f2
  engram.EngramEmbedding.forward                       d6fdfbdef32819a7
  hcn.hc_combine_norm                                  c272feb67f8189b6
  linear.RowParallelLinear.forward                     fdc660bd4e36a7d0

REMOVE from _EXPECTED (resolve fails on CUDA → guard raises):
  v4.DeepseekV4DecoderLayer._hc_combine               (→ op_combine_a/b)
  v4.DeepseekV4DecoderLayer._hc_mix_stats             (→ op_mhc_*)
  v4.DeepseekV4DecoderLayer.forward_hc_pre_from_prev  (Platform-only)
  v4.DeepseekV4DecoderLayer._run_moe_ffn_dp_sync      (Platform-only; → op_moe)

ADD (new op_* seams):
  v4.DeepseekV4DecoderLayer.forward                   119ec8cf2f4b24b8
  v4.DeepseekV4DecoderLayer.op_combine_a              e56937a61eaa0223
  v4.DeepseekV4DecoderLayer.op_combine_b              3b54be04ca8233a2
  v4.DeepseekV4DecoderLayer.op_gather_a               ad56ac6cf2ecc430
  v4.DeepseekV4DecoderLayer.op_gather_b               2a879ac1d41fbfec
  v4.DeepseekV4DecoderLayer.op_moe                    3bbe8b5550dc1682
  v4.DeepseekV4DecoderLayer.op_mhc_prepare_attn       dde98ef728494d10
  v4.DeepseekV4DecoderLayer.op_mhc_post_attn_pre_mlp  46cda3fc3d09de38
  v4.DeepseekV4DecoderLayer.op_mhc_postprocess        c6eb37f49f7526a8
```

Hash-stability: computed from the pinned image source (7d996300) — stable across boots.
Re-verify after any image rebuild with `scripts/verify/sp_drift_check.py` (CUDA-visible!).
