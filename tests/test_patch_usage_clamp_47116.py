#!/usr/bin/env python3
"""runtime/patch_usage_clamp_47116.py against a synthetic stand-in.

Reproduces only the anchor region of sglang main@7d996300's
scheduler_components/output_streamer.py (the _GenerationStreamAccumulator
capture appends); NOT SGLang's file.

Stock: reasoning_tokens records req.reasoning_tokens verbatim. Patched:
min(req.reasoning_tokens, len(output_ids_)) -- a DSpark verified block that
runs past the stop position must not inflate reported reasoning usage.
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / 'runtime' / 'patch_usage_clamp_47116.py'

FIXTURE = '''class _GenerationStreamAccumulator:
    def __init__(self):
        self.output_ids = []
        self.prompt_tokens = []
        self.reasoning_tokens = []
        self.cached_tokens = []

    def capture(self, req, output_ids_):
        self.output_ids.append(output_ids_[req.send_token_offset:])
        req.send_token_offset = len(output_ids_)
        self.prompt_tokens.append(len(req.origin_input_ids))
        self.reasoning_tokens.append(req.reasoning_tokens)
        self.cached_tokens.append(req.cached_tokens)
        return self
'''


class Req:
    def __init__(self, reasoning_tokens, sent, cached=0):
        self.origin_input_ids = [0] * 8
        self.reasoning_tokens = reasoning_tokens
        self.cached_tokens = cached
        self.send_token_offset = 0
        self._sent = sent

    @property
    def output_ids(self):
        return list(range(self._sent))


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
        path = base / 'sglang/srt/managers/scheduler_components/output_streamer.py'
        path.parent.mkdir(parents=True)
        path.write_text(FIXTURE)

        stock = load(path, 'os_stock')
        acc = stock._GenerationStreamAccumulator().capture(Req(50, 30), list(range(30)))
        assert acc.reasoning_tokens == [50], \
            'fixture: stock records the raw reasoning_tokens'

        first = run(base)
        assert first.returncode == 0, first.stderr
        text_after_first = path.read_text()
        second = run(base)
        assert second.returncode == 0 and 'already patched' in second.stdout, second.stdout
        assert path.read_text() == text_after_first

        mod = load(path, 'os_patched')
        # verified block ran past the stop: reasoning 50 > sent 30 -> clamped to 30
        acc2 = mod._GenerationStreamAccumulator().capture(Req(50, 30), list(range(30)))
        assert acc2.reasoning_tokens == [30], acc2.reasoning_tokens
        assert acc2.output_ids == [list(range(30))], 'capture body otherwise intact'
        # reasoning shorter than the output is untouched
        acc3 = mod._GenerationStreamAccumulator().capture(Req(12, 30), list(range(30)))
        assert acc3.reasoning_tokens == [12], acc3.reasoning_tokens

        # anchor drift fails loudly
        path.write_text(FIXTURE.replace(
            'self.reasoning_tokens.append(req.reasoning_tokens)',
            'self.reasoning_tokens.append(req.reasoning_tokens)  # d'))
        drifted = run(base)
        assert drifted.returncode == 1 and 're-derive from sglang#47116' in drifted.stderr, \
            drifted.stderr

    print('test_patch_usage_clamp_47116 PASS')


if __name__ == '__main__':
    main()
