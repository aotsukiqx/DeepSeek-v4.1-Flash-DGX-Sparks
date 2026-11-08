#!/usr/bin/env python3
"""Backport bp6: restore SM121 (GB10, CC 12.x) to the DeepGEMM indexer paths.

sglang#48399 (27bc1a06, 2026-11-06) replaced ``is_sm100_or_newer()``
(capability major >= 10, which included our SM121) with strict
``current_arch().is_sm(10)`` in the v41 indexer gates. GB10 fell out of the
DeepGEMM fp8_fp4 mqa-logits path onto the torch fallback, whose
``scoring.prefill_requests`` calls ``Tensor.tolist()`` once per layer (a
synchronous D2H): ~125 ms/layer fixed cost, 8x prefill regression, measured
2026-11-08 (torch-profiler trace /tmp/prof-traces, cudaMemcpyAsync 9.9 s of an
18.4 s window under `aten::to` from full_topk._torch_prefill).

Our fleet ran the DeepGEMM path on SM121 for weeks on the 7d996300/8817f63c
lines (qeval BROKE=0 repeatedly, decode 44-77 tok/s), and the 2026-11-08
dual-container kernel bench proved the 0.2.6 kernels equal to 0.2.3 on GB10
(fp8_fp4_paged_mqa_logits 18.96 vs 18.90 ms at indexer scale). The SM120
exclusion is upstream conservatism, not a known defect.

Patch: include the SM120 family (arch major 12) in the PATH gate only
(v41_indexer.is_sm100). The dtype gates (index_logits._is_sm100) stay STRICT:
deep-gemm 0.2.6 asserts fp32 logits/weights outside arch 10 (attention.hpp:50,
learned from the first bp6 boot: bf16 weights on SM121 crash the kernel), so
SM121 must take the DeepGEMM path with the fp32 dtype contract — which the
strict dtype gates produce automatically.
Call sites untouched (line-101 candidate gate also flips, but our
sm120_candidates adapter already forces the dense full-topk path there).
Applies clean on pristine 9d547e19 (verified). Idempotent. Exits non-zero on
anchor drift. Optional argv[1] overrides the sglang python tree root.
"""
import pathlib
import sys

DEFAULT_BASE = "/sgl-workspace/sglang/python"
PR = "48399"

GATE_COMMENT = "# DSV41 bp6: SM120 family (GB10, CC 12.1) runs the DeepGEMM\n# indexer paths (weeks-proven on this fleet; see patch script header).\n"

FILES = [
    (
        '''sglang/srt/layers/attention/dsv4/v41_indexer/__init__.py''',
        '''    return current_arch().is_sm(10) or current_arch().is_sm(12)
''',
        [
            ('''def is_sm100() -> bool:
    # The DeepGEMM fp8_fp4 mqa-logits paths are SM100-family only; other devices
    # (Hopper, SM120, non-CUDA) take the torch / Triton indexer.
    return current_arch().is_sm(10)
''',
             '''def is_sm100() -> bool:
    # The DeepGEMM fp8_fp4 mqa-logits paths are SM100-family only; other devices
    # (Hopper, SM120, non-CUDA) take the torch / Triton indexer.
''' + GATE_COMMENT + '''    return current_arch().is_sm(10) or current_arch().is_sm(12)
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
