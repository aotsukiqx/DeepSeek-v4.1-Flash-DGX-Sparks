#!/usr/bin/env python3
"""runtime/patch_poolcap_41275.py against a synthetic stand-in.

Reproduces only the three anchor regions of sglang main@7d996300's
pool_configurator.py (the low-ratio coefficient, the method boundary after
the ratio-128 return, and the auto-sized full_token computation) plus the
helpers they name; NOT SGLang's file. The patched _get_low_ratio_kv_padding_bytes
imports sglang.srt.mem_cache.deepseek_v4_memory_pool, resolved here via a
sys.modules stub.

Behavior checks after patching: the low-ratio coefficient uses the PAGED
per-token bytes, the padding method prices the reserved main page + indexer
page rounding (zero on _unified / trtllm), and the auto-sized full_token is
page-aligned and stepped down until tokens + padding fit the budget.
"""
import importlib.util
import math
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / 'runtime' / 'patch_poolcap_41275.py'

FIXTURE = '''_is_npu = False


class _Exec:
    class kernel:
        dsv4_attn_backend = "flashmla"


def get_exec():
    return _Exec()


class DSV4PoolConfigurator:
    def __init__(self, page_size=64, kv_bytes=32, paged_bytes=40,
                 low_ratio_index_bytes=8):
        self.page_size = page_size
        self.kv_bytes = kv_bytes
        self.paged_bytes = paged_bytes
        self.low_ratio_index_bytes = low_ratio_index_bytes
        self._unified = False
        self.bytes_per_full_token = None

    def _get_paged_kv_bytes_per_token(self, ratio):
        return self.paged_bytes

    def num_layers(self, ratio):
        return {1: 0, 2: 2, 128: 1}.get(ratio, 0)

    def _get_low_ratio_kv_bytes_per_token(self, ratio):
        """Compressed KV (+ indexer) bytes one full token adds per layer."""
        if ratio in (1, 2):
            return (self.kv_bytes + self.low_ratio_index_bytes) / ratio
        if ratio == 4:
            return 1.0
        assert ratio == 128, f"unsupported compression ratio: {ratio}"
        return 1 / 128 * self._get_paged_kv_bytes_per_token(128)

    def _get_bytes_per_full_token(self) -> float:
        return 10.0

    def calculate_pool_sizes(self, available_bytes_for_tokens, page_size):
        self.page_size = page_size
        full_token = int(available_bytes_for_tokens / self.bytes_per_full_token)
        return full_token
'''


def install_pool_stubs(index_page_size=256):
    def mod(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    for name in ("sglang", "sglang.srt", "sglang.srt.mem_cache"):
        mod(name)
    pool = mod("sglang.srt.mem_cache.deepseek_v4_memory_pool")
    pool.DSV41_INDEX_PAGE_SIZE = index_page_size
    # upstream helper returns a PAGE COUNT; the caller multiplies by page size
    pool._num_dsv4_physical_kv_pages = lambda n, page, page2: math.ceil(n / page)


def run(base):
    return subprocess.run([sys.executable, '-I', str(PATCHER), str(base)],
                          capture_output=True, text=True)


def load(path, name):
    install_pool_stubs()
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def expected_padding(cfg, full_tokens):
    # mirror of the upstream formula against the stubs, for cross-checking
    pad = 0
    for ratio in (1, 2):
        layers = cfg.num_layers(ratio)
        if not layers:
            continue
        main = cfg.page_size * cfg._get_paged_kv_bytes_per_token(ratio)
        index_size = (full_tokens + cfg.page_size) // ratio
        slots = math.ceil(index_size / 256) * 256
        index_pad = (slots - full_tokens // ratio) * cfg.low_ratio_index_bytes
        pad += (main + index_pad) * layers
    return pad


def main():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        path = base / 'sglang/srt/model_executor/pool_configurator.py'
        path.parent.mkdir(parents=True)
        path.write_text(FIXTURE)

        stock = load(path, 'pool_stock')
        cfg = stock.DSV4PoolConfigurator()
        cfg.bytes_per_full_token = 10.0
        assert cfg._get_low_ratio_kv_bytes_per_token(2) == 20.0, \
            'fixture: stock coefficient uses kv_bytes'
        assert cfg.calculate_pool_sizes(10000, 64) == 1000, \
            'fixture: stock full_token is the naive quotient'

        first = run(base)
        assert first.returncode == 0, first.stderr
        text_after_first = path.read_text()
        second = run(base)
        assert second.returncode == 0 and 'already patched' in second.stdout, second.stdout
        assert path.read_text() == text_after_first

        mod = load(path, 'pool_patched')
        cfg2 = mod.DSV4PoolConfigurator()
        cfg2.bytes_per_full_token = 10.0
        # coefficient now prices the paged per-token bytes
        assert cfg2._get_low_ratio_kv_bytes_per_token(2) == 24.0
        # padding: reserved main page + indexer page rounding, per low-ratio layer
        assert cfg2._get_low_ratio_kv_padding_bytes(960) == expected_padding(cfg2, 960)
        assert cfg2._get_low_ratio_kv_padding_bytes(0) == expected_padding(cfg2, 0)
        # zero padding on the exempt layouts
        cfg2._unified = True
        assert cfg2._get_low_ratio_kv_padding_bytes(960) == 0
        cfg2._unified = False
        mod._Exec.kernel.dsv4_attn_backend = "trtllm"
        assert cfg2._get_low_ratio_kv_padding_bytes(960) == 0
        mod._Exec.kernel.dsv4_attn_backend = "flashmla"
        # auto-sizing: page-aligned and tokens + padding within the budget,
        # and one page MORE would not fit (minimality)
        ft = cfg2.calculate_pool_sizes(10000, 64)
        assert ft % 64 == 0 and ft > 0, ft
        used = ft * cfg2.bytes_per_full_token + cfg2._get_low_ratio_kv_padding_bytes(ft)
        assert used <= 10000, (ft, used)
        ft_up = ft + 64
        up = ft_up * cfg2.bytes_per_full_token + cfg2._get_low_ratio_kv_padding_bytes(ft_up)
        assert up > 10000, (ft_up, up)

        # anchor drift fails loudly
        path.write_text(FIXTURE.replace(
            'full_token = int(available_bytes_for_tokens / self.bytes_per_full_token)',
            'full_token = int(available_bytes_for_tokens / self.bytes_per_full_token)  # d'))
        drifted = run(base)
        assert drifted.returncode == 1 and 're-derive from sglang#41275' in drifted.stderr, \
            drifted.stderr

    print('test_patch_poolcap_41275 PASS')


if __name__ == '__main__':
    main()
