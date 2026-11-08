#!/usr/bin/env python3
"""runtime/patch_indexer_gate_48399.py against a synthetic stand-in.

Reproduces the gate definition of sglang main@9d547e19 (v41_indexer
is_sm100 and index_logits _is_sm100) with a stub current_arch; NOT SGLang's
files. After patching, an SM120-family arch flips the gates True (SM121 runs
the DeepGEMM indexer paths, the weeks-proven bp5 behavior), Hopper stays
False, and the patch is idempotent with loud anchor-drift failure.
"""
import importlib.util
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / 'runtime' / 'patch_indexer_gate_48399.py'

FIX_INIT = '''import functools


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
    # The DeepGEMM fp8_fp4 mqa-logits paths are SM100-family only; other devices
    # (Hopper, SM120, non-CUDA) take the torch / Triton indexer.
    return current_arch().is_sm(10)
'''

def run(base):
    return subprocess.run([sys.executable, '-I', str(PATCHER), str(base)],
                          capture_output=True, text=True)


def load(path, name, major):
    import functools
    functools.cache.clear_key = None  # noqa: keep caches per-module instance
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    mod._CURRENT[0] = major
    mod.is_sm100.cache_clear()
    return mod


def main():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        files = {
            'sglang/srt/layers/attention/dsv4/v41_indexer/__init__.py': FIX_INIT,
                    }
        for rel, text in files.items():
            p = base / rel
            p.parent.mkdir(parents=True)
            p.write_text(text)

        init_path = base / 'sglang/srt/layers/attention/dsv4/v41_indexer/__init__.py'
        stock = load(init_path, 'gate_stock', 12)
        assert stock.is_sm100() is False, \
            'fixture: stock gate excludes SM121 (the regression)'

        first = run(base)
        assert first.returncode == 0, first.stderr
        text_after = init_path.read_text()
        second = run(base)
        assert second.returncode == 0 and 'already patched' in second.stdout, second.stdout
        assert init_path.read_text() == text_after

        patched = load(init_path, 'gate_patched', 12)
        assert patched.is_sm100() is True, 'patched: SM121 back on DeepGEMM'
        patched._CURRENT[0] = 10
        patched.is_sm100.cache_clear()
        assert patched.is_sm100() is True
        patched._CURRENT[0] = 9
        patched.is_sm100.cache_clear()
        assert patched.is_sm100() is False, 'Hopper stays on torch'

        init_path.write_text(FIX_INIT.replace(
            'return current_arch().is_sm(10)',
            'return current_arch().is_sm(10)  # d'))
        drifted = run(base)
        assert drifted.returncode == 1 and 're-derive from sglang#48399' in drifted.stderr, \
            drifted.stderr

    print('test_patch_indexer_gate_48399 PASS')


if __name__ == '__main__':
    main()
