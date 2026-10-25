# Plan: prefill_sp Re-Port to 7d996300 (Main Effort, 2-4 Sessions)

> Goal: rebuild the prefill sequence-parallel adapter on the new pin's `op_*` API family.
> Expected recovery: +18% cold prefill + ~8% decode = mig arm surpasses production baseline.

## 1. Problem Statement

prefill_sp (`adapter/prefill_sp.py`) wraps 16 engine functions (hash-checked at install).
On 7d996300:
- **9 functions DRIFTED** (source hash changed)
- **2 functions MISSING** (`_hc_combine`, `_hc_mix_stats` — moved to `op_*` methods)
- The mHC/MegaMoE wave renamed the API surface

The drift guard raises at serve time, so the adapter must be env-OFF (current mig arm).

## 2. What Changed Upstream

The candidate's `DeepseekV4DecoderLayer` now has these methods (replacing our hooks):

| Old (620c76f2) | New (7d996300) | Nature |
|---|---|---|
| `_hc_combine` | `op_combine_a` / `op_combine_b` | Split into two ops |
| `_hc_mix_stats` | `op_mhc_postprocess` etc. | Absorbed into mHC op chain |
| `hc_post` | `hc_post` (still exists, DRIFTED) | Modified internals |
| `forward_hc_pre_from_prev` | same name (DRIFTED) | Modified internals |
| `_run_moe_ffn_dp_sync` | same name (DRIFTED) | Modified internals |

Other drifted targets:
- `DeepseekV4Model._forward_layers_hc_pre_from_prev` (DRIFTED, 284 lines)
- `DeepseekV4Model.forward` (DRIFTED, 22 lines)
- `MQALayer.forward` (DRIFTED, 456 lines)
- `DeepseekV2MoE.forward` / `forward_normal` (DRIFTED)
- `Engram.forward` (DRIFTED)

## 3. Re-Port Strategy

### Phase A: Classification (offline, ~2 hours)
For each of the 9 drifted functions:
1. Diff old-pin source vs candidate source (line-by-line)
2. Classify each change: cosmetic (docstring, variable rename) / moderate (logic addition, same structure) / rewrite (new architecture)
3. For MISSING functions: map old → new op_* equivalents
4. Output: a porting checklist with effort estimates

### Phase B: Core Forward Path (1 session)
The heart of prefill_sp is `_forward_layers_hc_pre_from_prev` + `forward`:
1. Read old adapter's wrapped logic for these two functions
2. Read candidate's new source
3. Re-apply the SP sharding logic to the new source
4. Update `_EXPECTED` hashes
5. Run `test_prefill_sp.py` with updated fixtures

### Phase C: Decoder Layer Hooks (1 session)
- `forward_hc_pre_from_prev`, `hc_post`, `_run_moe_ffn_dp_sync`
- Map `_hc_combine` → `op_combine_a`/`op_combine_b`
- Map `_hc_mix_stats` → `op_mhc_*` chain
- Verify bit-identity: `DSV41_MOE_B12X_NEXT_DETERMINISTIC=1` + greedy OFF/ON comparison

### Phase D: MQA + MoE + Engram (1 session)
- `MQALayer.forward` (456 lines — the largest)
- `DeepseekV2MoE.forward` / `forward_normal`
- `Engram.forward` (already partially fixed with attr defaults)

### Phase E: Validation (1 boot window)
- Boot with prefill_sp ON
- qeval paired (BROKE=0 gate)
- 600k cold + decode + conc6 (vs both baselines)
- If pass → mig arm should now EXCEED production baseline

## 4. Hash Re-generation

After each phase, recompute hashes:
```python
import hashlib, inspect
def _src_hash(obj):
    return hashlib.sha256(inspect.getsource(inspect.unwrap(obj)).encode()).hexdigest()[:16]
# For each expected key, compute new hash and update _EXPECTED dict
```

The adapter has ~16+2=18 hash entries. Some may not need changing (if the function is
identical). Only update the ones that drifted.

## 5. Key Files

| File | Role |
|---|---|
| `adapter/prefill_sp.py` | The adapter (1367 lines) — main edit target |
| `tests/test_prefill_sp.py` | Unit tests (SimpleNamespace fixtures) |
| `/tmp/cand7/python/sglang/srt/models/deepseek_v4.py` | Candidate source (on head) |
| `/tmp/cand7/python/sglang/srt/models/deepseek_v2.py` | Candidate source |
| `/tmp/cand7/python/sglang/srt/layers/engram.py` | Candidate source |

## 6. Success Criteria

- Boot with `DSV41_PREFILL_SP=1` on mig arm
- qeval paired BROKE=0
- cold 600k ≥ 4500 (production baseline 4101 + expected recovery)
- decode ≥ 42 (production baseline 44.0, partial recovery)
- conc6 ≥ 4021 (current mig arm, should not regress)
