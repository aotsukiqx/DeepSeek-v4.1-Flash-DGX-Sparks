# Upstream issue draft: #48399 drops SM120/SM121 from the DSV4.1 DeepGEMM indexer — 8x prefill regression on GB10

**Target repo:** sgl-project/sglang (re-verified on main @ 9d547e19, 2026-11-08;
regression introduced by #48399 / 27bc1a06)
**Status:** ready to file — blocked on `gh auth login` locally.

## Summary

#48399 replaced `is_sm100_or_newer()` (device capability major >= 10) with
strict `current_arch().is_sm(10)` in `srt/layers/attention/dsv4/v41_indexer/
__init__.py`. SM120-family devices (GB10, compute capability 12.1) now fall
to `_torch_prefill`, whose `scoring.prefill_requests` calls `Tensor.tolist()`
once per layer — a synchronous D2H inside the layer loop. Measured on a
4x DGX Spark TP4 deployment of DeepSeek-V4.1-Flash: **~125 ms/layer fixed
cost, 8x prefill throughput regression** (4096-token chunks: ~480 tok/s vs
~4000 tok/s before), independent of chunk size.

## Evidence

1. **Torch profiler trace** (TP-0, 18.4 s window, one 120k-token prefill):
   `cudaMemcpyAsync` 9.86 s (n=3117) under `aten::to/_to_copy`, host stack
   `full_topk.py::_torch_prefill -> scoring.py:400 prefill_requests ->
   Tensor.tolist()`. GPU kernels in the window total ~14 s — the time is
   host-side sync.
2. **The SM121 DeepGEMM path is healthy**: our fleet ran it in production for
   weeks on pins up to 8817f63c (paired qeval BROKE=0 across six adoption
   windows; decode 44-77 tok/s). Dual-container microbench on identical
   GB10 hardware: `fp8_fp4_paged_mqa_logits` NEW (0.2.6) vs LEGACY (0.2.3
   semantics) = 18.96 vs 18.90 ms at indexer scale — no kernel regression.
3. **Reproduction**: revert the one-line gate to `is_sm(10) or is_sm(12)` on
   any SM120-family GPU — prefill recovers.

## The dtype subtlety (why a naive revert crashes)

deep-gemm 0.2.6 asserts `weights.scalar_type() == (arch_major == 10 ?
torch::kBFloat16 : torch::kFloat32)` (attention.hpp:50). The dtype gates in
`kernels/ops/attention/dsv4/index_logits.py` legitimately stayed strict
`is_sm(10)` after #48399 — so on SM121 the DeepGEMM path must run with the
**fp32** logits/weights contract. Restoring the PATH gate while keeping the
dtype gates strict works on GB10 (verified in production). If the intent of
#48399 was "SM120 keeps the torch indexer", note the per-layer `tolist()`
makes that fallback pathological for prefill — at minimum the torch path
should batch the D2H, and the arch dispatch deserves an explicit SM120
decision rather than the current silent demotion.

## Suggested fix (either)

- Restore `>= 10` (include SM120) for the path gates, keep dtype gates strict
  (fp32 contract outside arch 10), or
- Keep strict SM100 gating but fix `_torch_prefill`'s per-layer `tolist()`.

--- local provenance: docs/upstream/loop-abort-finished-len-bug.md is the
other pending draft (finished_len contract); both file once gh is authed.
