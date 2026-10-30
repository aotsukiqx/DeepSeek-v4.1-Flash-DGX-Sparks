#!/usr/bin/env python3
"""runtime/patch_pageidx_pad_46835.py against a synthetic stand-in.

Reproduces only the anchor region of sglang main@7d996300's
deepseek_v4_backend.py _pad_last_dim (+ ceil_align); NOT SGLang's file.

Stock: always F.pad (a full copy). Patched: already-aligned input returns
x itself; unaligned input still pads identically.
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / 'runtime' / 'patch_pageidx_pad_46835.py'

FIXTURE = '''PAGE_INDEX_ALIGNED_SIZE = 64

PAD_CALLS = []


class _F:
    @staticmethod
    def pad(x, pad, mode, value):
        PAD_CALLS.append((id(x), pad, value))
        return "PADDED:" + x.tag


def ceil_align(n, m):
    return (n + m - 1) // m * m


class X:
    def __init__(self, last, tag="x"):
        self.shape = (3, last)
        self.tag = tag


def _pad_last_dim(x, multiples_of=PAGE_INDEX_ALIGNED_SIZE):
    if x is None:
        return None
    curr_size = x.shape[-1]
    target_size = ceil_align(curr_size, multiples_of)
    return _F.pad(x, pad=(0, target_size - curr_size), mode="constant", value=-1)
'''


def run(base):
    return subprocess.run([sys.executable, '-I', str(PATCHER), str(base)],
                          capture_output=True, text=True)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        path = base / 'sglang/srt/layers/attention/deepseek_v4_backend.py'
        path.parent.mkdir(parents=True)
        path.write_text(FIXTURE)

        stock = load(path, 'bk_stock')
        aligned = stock.X(128)
        ret = stock._pad_last_dim(aligned)
        assert ret.startswith('PADDED:'), 'fixture: stock pads even when aligned'
        assert stock.PAD_CALLS[-1][1] == (0, 0)

        first = run(base)
        assert first.returncode == 0, first.stderr
        text_after_first = path.read_text()
        second = run(base)
        assert second.returncode == 0 and 'already patched' in second.stdout, second.stdout
        assert path.read_text() == text_after_first

        mod = load(path, 'bk_patched')
        calls_before = len(mod.PAD_CALLS)
        # aligned -> same object back, no pad call
        a = mod.X(128)
        assert mod._pad_last_dim(a) is a
        assert len(mod.PAD_CALLS) == calls_before
        # unaligned -> still pads to the next multiple
        u = mod.X(65)
        ret = mod._pad_last_dim(u)
        assert ret == 'PADDED:' + u.tag
        assert mod.PAD_CALLS[-1][1] == (0, 63)
        # None passthrough intact
        assert mod._pad_last_dim(None) is None

        # anchor drift fails loudly
        path.write_text(FIXTURE.replace(
            '    target_size = ceil_align(curr_size, multiples_of)',
            '    target_size = ceil_align(curr_size, multiples_of)  # d'))
        drifted = run(base)
        assert drifted.returncode == 1 and 're-derive from sglang#46835' in drifted.stderr, \
            drifted.stderr

    print('test_patch_pageidx_pad_46835 PASS')


if __name__ == '__main__':
    main()
