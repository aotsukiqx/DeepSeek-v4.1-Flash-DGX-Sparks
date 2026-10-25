# Plan: hc_fused Re-Port (Half Day)

> Goal: re-port the HC mix-stats fusion kernel to the candidate's new location.
> Expected recovery: +4.3% cold prefill (historical measurement on old pin).

## 1. What Changed

The kernel moved from `sglang.srt.layers.attention.deepseek_v4_backend` (old pin)
to `sglang.kernels.ops.gemm.hc_mix` (new pin). Additionally:
- A split-K variant was split out to `hc_mix_splitk.py`
- A new `mhc_mega` family appeared in `sglang.kernels.ops.layernorm/`
- The adapter's 4 hash-anchored kernel functions need re-verification

## 2. Adapter's Current Anchors

From `adapter/hc_fused.py`:
```python
_EXPECTED = {
    "_hc_mix_stats_partial_kernel": "ca7ec34b9be7abd7",
    "_hc_mix_reduce_sinkhorn_kernel": "8e9e13a25eb9885e",
    "hc_mix_stats_sinkhorn": "8ed53dab69e71a3d",
    "_num_slices_for": "23e78e75dc040b8e",
}
```

These are hashes of the KERNEL functions, not the engine call sites.

## 3. Steps

### Step 1: Verify kernel source stability (30 min)
Check if the 4 kernel functions in `kernels/ops/gemm/hc_mix.py` are source-identical
to their old-location versions. If yes, only the import path changes.

### Step 2: Update import paths (15 min)
In `adapter/hc_fused.py`, update the import from:
```python
# OLD: implicitly via attention backend
# NEW: explicit
from sglang.kernels.ops.gemm import hc_mix as hcm
```

### Step 3: Re-hash and update anchors (15 min)
```python
# In-container, against candidate tree:
import hashlib, inspect
from sglang.kernels.ops.gemm.hc_mix import (
    _hc_mix_stats_partial_kernel, _hc_mix_reduce_sinkhorn_kernel,
    hc_mix_stats_sinkhorn, _num_slices_for,
)
for fn in (...):
    print(fn.__name__, hashlib.sha256(
        inspect.getsource(fn).encode()).hexdigest()[:16])
# Update _EXPECTED in adapter/hc_fused.py
```

### Step 4: sitecustomize wiring (5 min)
The hook at `adapter/sitecustomize.py` needs the module name updated:
```python
# OLD: module.__name__ == 'sglang.srt.layers.attention.deepseek_v4_backend'
# NEW: module.__name__ == 'sglang.kernels.ops.gemm.hc_mix'
```

### Step 5: Evaluate new mhc_mega family (optional, 2 hours)
The `mhc_mega` kernels in `sglang.kernels.ops.layernorm/` may offer better performance
than the original hc_mix path. Evaluate whether to port to those instead.

### Step 6: Validation (one boot window)
- Boot with `DSV41_HC_FUSED=1` on mig arm
- Verify first-call bit-identity (the adapter logs it)
- 600k cold comparison

## 4. Success Criteria

- First-call bit-identity PASS (adapter's built-in check)
- cold 600k improvement ≥2% (bar from eval methodology)
