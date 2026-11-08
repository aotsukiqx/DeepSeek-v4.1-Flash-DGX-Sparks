#!/usr/bin/env python3
"""Backport bp6r2: SM121 (GB10, CC 12.x) on the DeepGEMM indexer paths; decode
env-switchable; pairs with sgl-flash-mla pinned back to 1.0.0 in the image.

Lineage (all measured on this fleet, 2026-11-08):
- sglang#48399 dropped SM121 from the DeepGEMM indexer gates -> torch prefill
  fallback with a per-layer ``Tensor.tolist()`` D2H = 8x prefill regression
  (torch-profiler evidence, /tmp/prof-traces). DeepGEMM prefill is restored
  unconditionally here.
- Decode on the v2 image (DeepGEMM + fp32 weights per deep-gemm 0.2.6's arch
  contract + sgl-flash-mla 1.0.2) measured 36.7 tok/s vs bp5's 43-45; torch
  decode (v3) measured 22.6. This branch keeps DeepGEMM decode ON by default
  (SGLANG_DSV41_DG_DECODE=0 opts out to the strict upstream gate) AND pins
  sgl-flash-mla back to 1.0.0 — the single-variable test of whether the
  decode cost was the 1.0.2 wheel.
- The dtype gates in kernels/ops/attention/dsv4/index_logits.py stay STRICT
  (deep-gemm 0.2.6 attention.hpp:50 asserts fp32 weights outside arch 10;
  flipping them crashes the kernel, learned on the first bp6 boot).

Idempotent. Exits non-zero on anchor drift. Optional argv[1] overrides the
sglang python tree root (default: the in-image path).
"""
import functools
import os
import pathlib
import sys

DEFAULT_BASE = "/sgl-workspace/sglang/python"
PR = "48399"

GATE_COMMENT = (
    "# DSV41 bp6r2: SM120 family (GB10, CC 12.1) runs the DeepGEMM indexer\n"
    "# paths (weeks-proven on this fleet pre-#48399; see patch script header).\n"
)

FILES = [
    (
        'sglang/srt/layers/attention/dsv4/v41_indexer/__init__.py',
        '    return (is_sm100() or current_arch().is_sm(12)) and has_dense_fp4_indexer()\n',
        [
            (
                'def _use_deep_gemm_prefill() -> bool:\n'
                '    return is_sm100() and has_dense_fp4_indexer()\n',
                'def _use_deep_gemm_prefill() -> bool:\n'
                + GATE_COMMENT +
                '    return (is_sm100() or current_arch().is_sm(12)) and has_dense_fp4_indexer()\n'
                '\n'
                '\n'
                '@functools.cache\n'
                'def _dg_decode() -> bool:\n'
                '    import os\n'
                '    # bp6r2: DeepGEMM decode ON by default (SGLANG_DSV41_DG_DECODE=0 -> strict upstream gate).\n'
                '    return os.environ.get("SGLANG_DSV41_DG_DECODE", "1") == "1" and _use_deep_gemm_prefill()\n',
            ),
            (
                '        use_deep_gemm_prefill=_use_deep_gemm_prefill(),\n'
                '        use_deep_gemm_decode=is_sm100(),\n',
                '        use_deep_gemm_prefill=_use_deep_gemm_prefill(),\n'
                '        use_deep_gemm_decode=_dg_decode(),\n',
            ),
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
        print("patched %s (%d hunks)" % (target, len(patches)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
