#!/usr/bin/env python3
"""Backport sglang#41275 (20ff40cb) onto sglang main@7d996300.

[Fix] deepseek-v4.1: Match KV capacity to physical layouts.

Auto-sized pools under-counted: the low-ratio coefficient used kv_bytes
instead of the paged per-token bytes, and the reserved main-KV page plus the
indexer pool's own page rounding were never priced in — an auto-computed
full_token could over-commit physical pages. The fix prices both into
_get_low_ratio_kv_bytes_per_token / the new _get_low_ratio_kv_padding_bytes
and steps full_token down page-by-page until tokens + padding fit.

Expected NO-OP on this fleet: we pass an explicit --max-total-tokens, which
routes through calculate_pool_sizes_from_max_tokens (untouched). Boot-gate
proof: the pool-sizing log line must be byte-identical pre/post patch.

Applies clean with ``git apply`` on the pristine 7d996300 file (verified
2026-10-27). PIN-MIGRATION NOTE: pool_configurator.py is under active churn;
re-derive against the then-current layout.

Idempotent. Exits non-zero on anchor drift (these scripts are the
build's drift probes). Optional argv[1] overrides the sglang python
tree root (default: the in-image path).
"""
import pathlib
import sys

DEFAULT_BASE = "/sgl-workspace/sglang/python"
PR = "41275"

FILES = [
    (
        '''sglang/srt/model_executor/pool_configurator.py''',
        '''                self._get_paged_kv_bytes_per_token(ratio) + self.low_ratio_index_bytes''',
        [
            ('''            return (self.kv_bytes + self.low_ratio_index_bytes) / ratio
''',
             '''            return (
                self._get_paged_kv_bytes_per_token(ratio) + self.low_ratio_index_bytes
            ) / ratio
'''),
            ('''        return 1 / 128 * self._get_paged_kv_bytes_per_token(128)
''',
             '''        return 1 / 128 * self._get_paged_kv_bytes_per_token(128)

    def _get_low_ratio_kv_padding_bytes(self, full_tokens: int) -> float:
        """Reserved main/indexer pages beyond the token-scaled low-ratio bytes."""
        if self._unified or _is_npu or get_exec().kernel.dsv4_attn_backend == "trtllm":
            return 0
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DSV41_INDEX_PAGE_SIZE,
            _num_dsv4_physical_kv_pages,
        )

        padding_bytes = 0
        for ratio in (1, 2):
            num_layers = self.num_layers(ratio)
            if not num_layers:
                continue
            # Main KV reserves a full logical page of compressed slots. Indexer
            # size includes the compressed logical page, then the indexer pool
            # reserves its own page on top (same page count as the pool).
            main_padding = self.page_size * self._get_paged_kv_bytes_per_token(ratio)
            index_size = (full_tokens + self.page_size) // ratio
            index_slots = (
                _num_dsv4_physical_kv_pages(
                    index_size, DSV41_INDEX_PAGE_SIZE, DSV41_INDEX_PAGE_SIZE
                )
                * DSV41_INDEX_PAGE_SIZE
            )
            index_padding = (
                index_slots - full_tokens // ratio
            ) * self.low_ratio_index_bytes
            padding_bytes += (main_padding + index_padding) * num_layers
        return padding_bytes
'''),
            ('''        full_token = int(available_bytes_for_tokens / self.bytes_per_full_token)
''',
             '''        full_token = int(available_bytes_for_tokens / self.bytes_per_full_token)
        full_token = full_token // page_size * page_size
        # The coefficient excludes reserved pages. Remove the few full pages
        # needed to cover them, including indexer rounding at this capacity.
        while full_token > 0 and (
            full_token * self.bytes_per_full_token
            + self._get_low_ratio_kv_padding_bytes(full_token)
            > available_bytes_for_tokens
        ):
            full_token -= page_size
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
        # One atomic write per file: the file is either pristine (sentinel
        # absent) or fully patched (sentinel present). OLD blocks carry anchor
        # context lines that legitimately survive the patch, so presence of the
        # sentinel -- absent from the pristine file, unique in the patched one
        # -- is the already-patched test.
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
