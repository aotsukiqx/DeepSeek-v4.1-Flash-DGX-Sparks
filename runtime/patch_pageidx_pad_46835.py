#!/usr/bin/env python3
"""Backport sglang#46835 (ec3365de) onto sglang main@7d996300.

[Perf] dsv4: Skip the copy when the page-index table is already aligned.

_pad_last_dim returned F.pad(...) unconditionally; when the last dim is
already a multiple of PAGE_INDEX_ALIGNED_SIZE the pad is a full copy of
the table for nothing. Return x instead.

EXECUTES HERE: deepseek_v4_backend is our attention backend; the page-index
table build runs per metadata refresh. Micro-perf; no wire surface. Gate =
decode/cold probes must not regress.

Applies clean with `git apply` on the pristine 7d996300 file (verified
2026-10-30). PIN-MIGRATION NOTE: absorbed at any pin >= ec3365de; drop
this script then.

Idempotent. Exits non-zero on anchor drift (these scripts are the
build's drift probes). Optional argv[1] overrides the sglang python
tree root (default: the in-image path).
"""
import pathlib
import sys

DEFAULT_BASE = "/sgl-workspace/sglang/python"
PR = '46835'

FILES = [
    (
        '''sglang/srt/layers/attention/deepseek_v4_backend.py''',
        '''    if target_size == curr_size:
''',
        [
            ('''    target_size = ceil_align(curr_size, multiples_of)
''',
             '''    target_size = ceil_align(curr_size, multiples_of)
    if target_size == curr_size:
        return x  # F.pad would still copy
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
