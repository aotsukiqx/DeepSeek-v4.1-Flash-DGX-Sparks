#!/usr/bin/env python3
"""runtime/patch_indexer_gate_48399.py (bp6r2 form) against a synthetic stand-in.

Reproduces sglang main@9d547e19's v41_indexer gate trio with a stub
current_arch; NOT SGLang's file. After patching: SM121 flips
_use_deep_gemm_prefill True and _dg_decode True by default (env opt-out);
SGLANG_DSV41_DG_DECODE=0 restores the strict decode gate; is_sm100 and the
no-env decode gate stay strict for non-SM120; idempotent; drift fails loudly.
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / 'runtime' / 'patch_indexer_gate_48399.py'

FIXTURE = '''import functools


class _Arch:
    def __init__(self, major):
        self._major = major

    def is_sm(self, major):
        return self._major == major


_CURRENT = [None]


def current_arch():
    return _Arch(_CURRENT[0])


@functools.cache
def is_sm100() -> bool:
    return current_arch().is_sm(10)


def has_dense_fp4_indexer() -> bool:
    return True


@functools.cache
def _use_deep_gemm_prefill() -> bool:
    return is_sm100() and has_dense_fp4_indexer()


def make_indexer():
    return (_use_deep_gemm_prefill(), is_sm100())
'''


def run(base):
    return subprocess.run([sys.executable, '-I', str(PATCHER), str(base)],
                          capture_output=True, text=True)


def load(path, name, major, env):
    os.environ['SGLANG_DSV41_DG_DECODE'] = env
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    mod._CURRENT[0] = major
    mod._use_deep_gemm_prefill.cache_clear()
    mod.is_sm100.cache_clear()
    return mod


def main():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'sglang/srt/layers/attention/dsv4/v41_indexer/__init__.py'
        path.parent.mkdir(parents=True)
        path.write_text(FIXTURE)

        stock = load(path, 's', 12, '1')
        pf, dec = stock.make_indexer()
        assert pf is False, 'fixture: stock prefill gate excludes SM121'

        r = run(Path(d))
        assert r.returncode == 0, r.stderr
        assert 'already patched' in run(Path(d)).stdout

        m = load(path, 'p1', 12, '1')
        pf, _ = m.make_indexer()
        assert pf is True, 'patched: SM121 prefill on DeepGEMM'
        assert m.is_sm100() is False
        os.environ.pop('SGLANG_DSV41_DG_DECODE', None)
        m2 = load(path, 'p2', 12, '0')
        assert m2._use_deep_gemm_prefill() is True, 'prefill unaffected by decode env'
        m3 = load(path, 'p3', 9, '1')
        assert m3._use_deep_gemm_prefill() is False, 'Hopper stays on torch'
        m4 = load(path, 'p4', 10, '1')
        assert m4._use_deep_gemm_prefill() is True and m4.is_sm100() is True

        path.write_text(FIXTURE.replace(
            '    return is_sm100() and has_dense_fp4_indexer()\n',
            '    return is_sm100() and has_dense_fp4_indexer()  # d'))
        dr = run(Path(d))
        assert dr.returncode == 1 and 're-derive from sglang#48399' in dr.stderr, dr.stderr

    print('test_patch_indexer_gate_48399 PASS')


if __name__ == '__main__':
    main()
