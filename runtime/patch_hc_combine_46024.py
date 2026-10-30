#!/usr/bin/env python3
"""Backport sglang#46024 (302b78cc) onto sglang main@7d996300.

[Perf] mhc: Split hc_combine_norm_mxfp8 rows by batch size and zero padding
scales in the kernel.

The old row-split heuristic (4/2/1 parts by m) was tuned for 148-SM parts;
the new _parts_for(m, K) picks splits by batch size and the kernel zeroes
the padding-scale region itself. EXECUTES HERE when the mxfp8 quantized HC
combine fires: deepseek_v4_mhc.py gates on is_blackwell + bf16 + cfg
mult==4 hidden==5120 + quantize + batch<=128 -- i.e. DECODE-shaped batches.
GB10 has 48 SMs (not 148), so the mistuned split is plausibly hurting
decode latency here. Gate = paired decode600k, adopt only if not slower.

Applies clean with `git apply` on the pristine 7d996300 file (verified
2026-10-30). PIN-MIGRATION NOTE: hc kernels move with the mhc wave;
re-derive against kernels/ops/layernorm/hc_combine_norm.py then.

Idempotent. Exits non-zero on anchor drift (these scripts are the
build's drift probes). Optional argv[1] overrides the sglang python
tree root (default: the in-image path).
"""
import pathlib
import sys

DEFAULT_BASE = "/sgl-workspace/sglang/python"
PR = '46024'

FILES = [
    (
        '''sglang/kernels/ops/layernorm/hc_combine_norm.py''',
        '''    parts = _parts_for(m, 5120)
''',
        [
            ('''    parts = 4 if m <= 8 else (2 if m <= 48 else 1)
''',
             '''    parts = _parts_for(m, 5120)
'''),
            ('''    S,
''',
             '''    S,
    M,
'''),
            ('''    h = tl.arange(0, BLOCK)
    m = h < K
    value = tl.full((BLOCK,), 0, tl.float32)
    for c in tl.static_range(4):
        pre = tl.load(P + row * SP + c).to(tl.float32)
        x = tl.load(X + row * SX + c * K + h, m, 0).to(tl.float32)
        value += x * pre
    # The unfused combine stores BF16 before RMSNorm reads it.
    value = value.to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(value * value, 0) / K + EPS)
    weight = tl.load(W + h, m, 0).to(tl.float32)
    y = (value * inv_rms * weight).to(tl.bfloat16)
    tl.store(Y + row * K + h, y, m & (h >= part * SLICE) & (h < (part + 1) * SLICE))
    mxfp8_epilogue(
        y, row, Q, S, K, BLOCK, GROUPS, part * (SLICE // 32), (part + 1) * (SLICE // 32)
    )
''',
             '''    if row < M:
        h = tl.arange(0, BLOCK)
        m = h < K
        value = tl.full((BLOCK,), 0, tl.float32)
        for c in tl.static_range(4):
            pre = tl.load(P + row * SP + c).to(tl.float32)
            x = tl.load(X + row * SX + c * K + h, m, 0).to(tl.float32)
            value += x * pre
        # The unfused combine stores BF16 before RMSNorm reads it.
        value = value.to(tl.bfloat16).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(value * value, 0) / K + EPS)
        weight = tl.load(W + h, m, 0).to(tl.float32)
        y = (value * inv_rms * weight).to(tl.bfloat16)
        slice_mask = (h >= part * SLICE) & (h < (part + 1) * SLICE)
        tl.store(Y + row * K + h, y, m & slice_mask)
        g_lo, g_hi = part * (SLICE // 32), (part + 1) * (SLICE // 32)
        mxfp8_epilogue(y, row, Q, S, K, BLOCK, GROUPS, g_lo, g_hi)
    else:
        # Padding scale entries are disjoint from the live rows above.
        off = ((row - M) * tl.num_programs(1) + part) * 512 + tl.arange(0, 512)
        pad_row = ((off // 4) % 4) * 32 + ((off // 16) % 32)
        tl.store(S + off, 0, (off < GROUPS * 128) & (pad_row >= M))
'''),
            ('''def _parts_for(k: int) -> int:
    # Row splits: recomputing the statistic beats running 6 CTAs on 148 SMs.
    parts = 4
''',
             '''def _parts_for(m: int, k: int) -> int:
    parts = 4 if m <= 8 else (2 if m <= 48 else 1)
'''),
            ('''    s = torch.zeros(
''',
             '''    # The kernel zeroes the padding rows' scales; no separate memset.
    s = torch.empty(
'''),
            ('''    """Four-stream combine + RMSNorm returning ``(y_bf16, y_q, y_sf)``."""
''',
             '''    """Four-stream combine + RMSNorm returning ``(y_bf16, y_q, y_sf)``.

    Every CTA of a row computes the whole row's reduction in the same order, so
    the outputs do not depend on the grid split."""
'''),
            ('''    parts = _parts_for(k)
    _hc_combine_norm_mxfp8_kernel[(m, parts)](
''',
             '''    parts = _parts_for(m, k)
    # Programs past the last row each zero one 512-byte block of padding scales.
    pad_blocks = triton.cdiv(s.numel(), 512) if m % 128 else 0
    _hc_combine_norm_mxfp8_kernel[(m + triton.cdiv(pad_blocks, parts), parts)](
'''),
            ('''        s,
''',
             '''        s,
        m,
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
        print("patched %s (%d hunk%s)" % (target, len(patches),
              "" if len(patches) == 1 else "s"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
