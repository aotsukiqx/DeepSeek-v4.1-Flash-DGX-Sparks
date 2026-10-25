# hc_fused Re-port Plan (kernel path update, ~half day)

> Expected: +4.3% cold prefill (historically measured on the old pin).
> The kernel CODE didn't change — only its MODULE PATH moved upstream.

## Current state

- **Old pin (620c76f2)**: kernel at `sglang.kernels.ops.layernorm.mhc`
- **New pin (7d996300)**: kernel at `sglang.kernels.ops.gemm.hc_mix.py` (+ `hc_mix_splitk.py`)
- **Our adapter**: `adapter/hc_fused.py` — imports from the old path, 4 hash-gated kernel sources
- **sitecustomize wiring**: hooks `sglang.kernels.ops.layernorm.mhc` (old path)

## What changed upstream

1. Module moved: `sglang.kernels.ops.layernorm.mhc` → `sglang.kernels.ops.gemm.hc_mix`
2. New split-K variant: `hc_mix_splitk.py` (separate file, may be relevant)
3. New `mhc_mega` family: `sglang.kernels.ops.layernorm.mhc_mega.py` (evaluate separately)
4. The 4 kernel sources (`_hc_mix_stats_partial_kernel` etc.) — **content may be identical** (the move was organizational, part of the kernels regroup commit #45112)

## Steps

### Step 1: Verify kernel source hashes (30 min, offline)

```bash
# In canary-mig container, compare old vs new module's kernel sources:
docker run --rm --network none \
  -v ~/nvmodels/DeepSeek-v4.1-Flash-DGX-Sparks:/repo \
  --entrypoint python3 dsv41-4x-spark:canary-mig -c "
import inspect, hashlib, sys
sys.path.insert(0, '/repo/adapter')
# NEW path
import sglang.kernels.ops.gemm.hc_mix as new_mod
for name in ('_hc_mix_stats_partial_kernel', '_hc_mix_reduce_sinkhorn_kernel',
             'hc_mix_stats_sinkhorn', '_num_slices_for'):
    obj = getattr(new_mod, name, None)
    if obj:
        h = hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()[:16]
        print(f'{name}: {h}')
    else:
        print(f'{name}: MISSING')
"
```

If hashes match the adapter's `_EXPECTED` values → **only update the import path** (30 min).
If hashes differ → compare sources, classify changes, re-port (2-4 hours).

### Step 2: Update adapter import + sitecustomize wiring (30 min)

- `adapter/hc_fused.py`: change `from sglang.kernels.ops.layernorm import mhc` to `from sglang.kernels.ops.gemm import hc_mix as mhc`
- `adapter/sitecustomize.py`: change the watched module name from `sglang.kernels.ops.layernorm.mhc` to `sglang.kernels.ops.gemm.hc_mix`
- Update `_EXPECTED` hashes if Step 1 showed drift

### Step 3: Evaluate mhc_mega family (1 hour, optional)

The new `mhc_mega.py` may already provide what hc_fused does natively. Check:
- `sglang.kernels.ops.layernorm.mhc_mega` — what does it export?
- Is it automatically used for prefill-size batches?
- If so, hc_fused might be OBSOLETE (upstream absorbed the optimization).

### Step 4: Test + boot window

1. `python3 -m py_compile adapter/hc_fused.py adapter/sitecustomize.py`
2. In-container import chain test (32 modules, must stay CLEAN)
3. Boot window with `DSV41_HC_FUSED=1` added to env
4. Probes: cold 600k ×2 + decode ×2
5. Compare vs first mig measurement (cold 3953) — expect ~+4% → ~4110

## Success criteria

- cold 600k ≥ 4050 (recovering the hc_fused historical +4.3% from 3953 baseline)
- No regression in decode or conc6
- qeval BROKE=0

## Rollback

Set `DSV41_HC_FUSED=0` in env; no code changes needed to disable.
