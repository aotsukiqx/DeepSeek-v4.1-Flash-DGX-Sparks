#!/usr/bin/env python3
"""runtime/patch_indexer_gate_48399.py (v3, prefill-only) against a synthetic stand-in.

Reproduces sglang main@9d547e19's v41_indexer __init__ gate trio
(is_sm100 / has_dense_fp4_indexer / _use_deep_gemm_prefill) with a stub
current_arch; NOT SGLang's file. After patching: SM121 flips
_use_deep_gemm_prefill True (DeepGEMM prefill restored) while is_sm100 and
the decode gate stay strict; Hopper stays False everywhere; idempotent;
anchor drift fails loudly.
"""
import importlib.util
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
'''


def run(base):
    return subprocess.run([sys.executable, '-I', str(PATCHER), str(base)],
                          capture_output=True, text=True)


def load(path, name, major):
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

        stock = load(path, 'g_stock', 12)
        assert stock._use_deep_gemm_prefill() is False, \
            'fixture: stock prefill gate excludes SM121 (the regression)'

        first = run(Path(d))
        assert first.returncode == 0, first.stderr
        text_after = path.read_text()
        second = run(Path(d))
        assert second.returncode == 0 and 'already patched' in second.stdout, second.stdout
        assert path.read_text() == text_after

        p121 = load(path, 'g_121', 12)
        assert p121._use_deep_gemm_prefill() is True, 'patched: SM121 prefill on DeepGEMM'
        assert p121.is_sm100() is False, 'is_sm100 stays strict (decode gate)'
        p100 = load(path, 'g_100', 10)
        assert p100._use_deep_gemm_prefill() is True and p100.is_sm100() is True
        p90 = load(path, 'g_90', 9)
        assert p90._use_deep_gemm_prefill() is False, 'Hopper stays on torch'

        path.write_text(FIXTURE.replace(
            '    return is_sm100() and has_dense_fp4_indexer()',
            '    return is_sm100() and has_dense_fp4_indexer()  # d'))
        drifted = run(Path(d))
        assert drifted.returncode == 1 and 're-derive from sglang#48399' in drifted.stderr, \
            drifted.stderr

    print('test_patch_indexer_gate_48399 PASS')


if __name__ == '__main__':
    main()
