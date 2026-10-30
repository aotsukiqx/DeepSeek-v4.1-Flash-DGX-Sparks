#!/usr/bin/env python3
"""Backport sglang#47116 (68e57f40) onto sglang main@7d996300.

[Fix] usage: Clamp reasoning tokens after stop truncation.

A DSpark verified token block can extend past the stop position, so
req.reasoning_tokens can exceed the tokens actually sent; the usage report
then over-states reasoning tokens. Clamp to the output prefix (min with
len(output_ids_)), the same prefix completion usage reports.

EXECUTES HERE: DSpark MTP verify is our spec path; every stop-truncated
finish hits this accumulator. Wire probe in the deploy window: reasoning
usage must not exceed completion usage on a stop-truncated request.

Applies clean with `git apply` on the pristine 7d996300 file (verified
2026-10-30). PIN-MIGRATION NOTE: output_streamer.py churns with the
scheduler observability wave (#47166/#47178 lineage) -- re-derive.

Idempotent. Exits non-zero on anchor drift (these scripts are the
build's drift probes). Optional argv[1] overrides the sglang python
tree root (default: the in-image path).
"""
import pathlib
import sys

DEFAULT_BASE = "/sgl-workspace/sglang/python"
PR = '47116'

FILES = [
    (
        '''sglang/srt/managers/scheduler_components/output_streamer.py''',
        '''        # A verified token block can extend past the stop position. Report
''',
        [
            ('''        self.reasoning_tokens.append(req.reasoning_tokens)
''',
             '''        # A verified token block can extend past the stop position. Report
        # reasoning usage only within the same prefix as completion usage.
        self.reasoning_tokens.append(min(req.reasoning_tokens, len(output_ids_)))
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
