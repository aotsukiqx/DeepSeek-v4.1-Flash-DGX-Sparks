#!/usr/bin/env python3
"""runtime/patch_hc_combine_46024.py against a synthetic stand-in.

Reproduces the anchor regions of sglang main@7d996300's
kernels/ops/layernorm/hc_combine_norm.py: the _hc_combine_norm_mxfp8_kernel
body, the python-side split heuristic (_parts_for + the launch that passes
the split count), and hc_combine_norm_mxfp8's dispatch. Triton is stubbed
(jit = identity, tl = permissive namespace) so the module imports CPU-side;
the KERNEL bodies are not executed -- the checks are on the split heuristic
(now batch-size-aware via _parts_for(m, 5120) and zeroing padding scales
inside the kernel instead of host-side) plus structural markers.
"""
import importlib.util
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / 'runtime' / 'patch_hc_combine_46024.py'

FIXTURE = '''import triton
import triton.language as tl


class _T:
    def __init__(self, shape):
        self.shape = shape

    def numel(self):
        n = 1
        for v in self.shape:
            n *= v
        return n


class _torch:
    @staticmethod
    def zeros(shape):
        return _T(shape)

    @staticmethod
    def empty(shape):
        return _T(shape)


torch = _torch


def _parts_for(k: int) -> int:
    # Row splits: recomputing the statistic beats running 6 CTAs on 148 SMs.
    parts = 4
    if k > 8192:
        parts = 2
    if k > 32768:
        parts = 1
    return parts


@triton.jit
def _hc_combine_norm_mxfp8_kernel(
    Y, X, P, W, Q,
    S,
    K: tl.constexpr, BLOCK: tl.constexpr, GROUPS: tl.constexpr,
    part: tl.constexpr, SLICE: tl.constexpr, EPS: tl.constexpr,
):
    row = tl.program_id(0)
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
    tl.store(Y + row * K + h, y, m & (h >= part * SLICE) & (h < (part + 1) * SLICE))
    mxfp8_epilogue(
        y, row, Q, S, K, BLOCK, GROUPS, part * (SLICE // 32), (part + 1) * (SLICE // 32)
    )


def hc_combine_norm(x, apply_pre, weight, eps):
    m = x.shape[0]
    k = x.shape[-1]
    parts = 4 if m <= 8 else (2 if m <= 48 else 1)
    return parts


def hc_combine_norm_mxfp8(x, apply_pre, weight, eps):
    """Four-stream combine + RMSNorm returning ``(y_bf16, y_q, y_sf)``."""
    m, k = x.shape
    q = torch.zeros((m, k))
    s = torch.zeros(
        (m, 4)
    )
    parts = _parts_for(k)
    _hc_combine_norm_mxfp8_kernel[(m, parts)](
        x,
        apply_pre,
        weight,
        s,
        K=k,
    )
    return parts, s
'''


class _T:
    def __init__(self, shape):
        self.shape = shape

    def numel(self):
        n = 1
        for v in self.shape:
            n *= v
        return n


def install_triton_stub():
    class _JitFn:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                return None  # kernels are not executed CPU-side
            return launch

    def jit(fn=None, **kw):
        return _JitFn(fn) if fn is not None else (lambda f: _JitFn(f))

    triton = types.ModuleType('triton')
    triton.jit = jit
    triton.cdiv = lambda a, b: -(-a // b)
    tl = types.ModuleType('triton.language')

    class _TL:
        def __getattr__(self, name):
            return name

    tl.__getattr__ = lambda name: name  # permissive: tl.constexpr etc.
    sys.modules['triton'] = triton
    sys.modules['triton.language'] = tl


def run(base):
    return subprocess.run([sys.executable, '-I', str(PATCHER), str(base)],
                          capture_output=True, text=True)


def load(path, name):
    install_triton_stub()
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        path = base / 'sglang/kernels/ops/layernorm/hc_combine_norm.py'
        path.parent.mkdir(parents=True)
        path.write_text(FIXTURE)

        stock = load(path, 'hc_stock')
        x4 = _T((4, 5120))
        assert stock.hc_combine_norm(x4, None, None, 1e-5) == 4, \
            'fixture: stock bf16 path uses the inline 4/2/1 heuristic'
        assert stock._parts_for(5120) == 4, 'fixture: stock _parts_for is k-only'

        first = run(base)
        assert first.returncode == 0, first.stderr
        text_after = path.read_text()
        second = run(base)
        assert second.returncode == 0 and 'already patched' in second.stdout, second.stdout
        assert path.read_text() == text_after

        mod = load(path, 'hc_patched')
        text = text_after
        # _parts_for is now (m, k): batch-size table, k thresholds kept
        assert mod._parts_for(4, 5120) == 4 and mod._parts_for(100, 5120) == 1
        assert mod._parts_for(100, 65536) == 1
        # bf16 path now shares the batch-aware heuristic
        assert mod.hc_combine_norm(x4, None, None, 1e-5) == 4
        assert mod.hc_combine_norm(_T((100, 5120)), None, None, 1e-5) == 1
        # mxfp8 wrapper: no host-side zeroing, padding blocks priced into the grid
        assert 's = torch.empty(' in text and 's = torch.zeros(' not in text.split('hc_combine_norm_mxfp8')[1]
        assert 'pad_blocks = triton.cdiv(s.numel(), 512) if m % 128 else 0' in text
        assert 'm + triton.cdiv(pad_blocks, parts)' in text
        parts, s = mod.hc_combine_norm_mxfp8(x4, None, None, 1e-5)
        assert parts == 4 and s.numel() == 16, (parts, s.shape)
        # kernel body guarded by the live-row count with an in-kernel zeroing else
        assert 'if row < M:' in text and 'tl.store(S + off, 0,' in text

        # anchor drift fails loudly
        path.write_text(FIXTURE.replace(
            '    parts = _parts_for(k)\n',
            '    parts = _parts_for(k)  # d\n'))
        drifted = run(base)
        assert drifted.returncode == 1 and 're-derive from sglang#46024' in drifted.stderr, \
            drifted.stderr

    print('test_patch_hc_combine_46024 PASS')


if __name__ == '__main__':
    main()
