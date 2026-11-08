#!/usr/bin/env python3
"""Backport bp6v3: SM121 (GB10, CC 12.x) on the DeepGEMM indexer PREFILL path only.

History:
- sglang#48399 (27bc1a06) replaced ``is_sm100_or_newer()`` (capability>=10)
  with strict ``is_sm(10)`` in the v41 indexer gates. GB10 fell to the torch
  prefill fallback whose ``scoring.prefill_requests`` calls ``Tensor.tolist()``
  once per layer (synchronous D2H): ~125 ms/layer, 8x prefill regression
  (torch-profiler evidence 2026-11-08, /tmp/prof-traces).
- bp6 v1/v2 also restored the DECODE gate; v2 measured decode 36.7 tok/s vs
  the >=42 gate (bp5 ran decode under deep-gemm 0.2.3 with bf16 weights;
  0.2.6's arch-12 contract forces fp32 weights, and the measured decode cost
  regressed ~17%). v3 therefore extends ONLY the prefill gate; decode stays
  strict (torch decode path = the upstream default for every non-SM100
  device). SGLANG_DSV41_DG_DECODE=1 env can re-enable DeepGEMM decode on
  SM120 for a later measured A/B if wanted.

Patch: ``_use_deep_gemm_prefill`` includes arch 12. The dtype gates in
index_logits.py stay STRICT (deep-gemm 0.2.6 attention.hpp:50 asserts fp32
weights outside arch 10), so SM121 runs DeepGEMM prefill under the fp32
contract. ``is_sm100()`` itself is untouched. Idempotent. Exits non-zero on
anchor drift. Optional argv[1] overrides the sglang python tree root.
"""
import pathlib
import sys

DEFAULT_BASE = "/sgl-workspace/sglang/python"
PR = "48399"

FILES = [
    (
        '''sglang/srt/layers/attention/dsv4/v41_indexer/__init__.py''',
        '''    return (is_sm100() or current_arch().is_sm(12)) and has_dense_fp4_indexer()
''',
        [
            ('''def _use_deep_gemm_prefill() -> bool:
    return is_sm100() and has_dense_fp4_indexer()
''',
             '''def _use_deep_gemm_prefill() -> bool:
    # DSV41 bp6v3: include the SM120 family (GB10, CC 12.1) on the DeepGEMM
    # PREFILL path only (weeks-proven on this fleet pre-#48399; the torch
    # fallback's per-layer tolist() D2H cost 8x prefill). The decode gate at
    # the construction site stays strict is_sm100(): 0.2.6's fp32-weights
    # contract measured -17% decode on SM121 (bp6v2, 2026-11-08).
    return (is_sm100() or current_arch().is_sm(12)) and has_dense_fp4_indexer()
'''),
        ],
    ),
]

def main() -> int:
    base = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE)
    rc = 0
    for rel, sentinel, patches in FILES:
        target = base / rel
        if not target.is_file():
            print("ERROR: target not found: %s (base image changed?)" % target,
                  file=sys.stderr)
            return 1
        src = target.read_text()
        if sentinel in src:
            print("already patched: %s" % target)
            continue
        for old, _ in patches:
            n = src.count(old)
            if n != 1:
                print("ERROR: %s: expected exactly 1 anchor match, found %d -- "
                      "base changed, re-derive from sglang#%s" % (target, n, PR),
                      file=sys.stderr)
                rc = 1
        if rc:
            return rc
        for old, new in patches:
            src = src.replace(old, new)
        if sentinel not in src or any(new not in src for _, new in patches):
            print("ERROR: post-patch verification failed: %s" % target,
                  file=sys.stderr)
            return 1
        target.write_text(src)
        print("patched %s (1 hunk)" % target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
