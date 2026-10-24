#!/usr/bin/env python3
"""Backport sglang#43941 (a2e4ffbc) onto sglang main@620c76f2.

fix(openai)/parser: Accept whole-number floats for integer tool arguments.

In ``chat_parsing.response_parser._coerce``, an integer argument written as a
whole number float ("3.0" — legal JSON for a JSON-Schema integer, common from
JS/Python agent clients) raised ValueError inside the try, hit ``continue``,
and the ENTIRE argument set was silently dropped. The fix falls back to
``float(raw)`` and accepts it only when ``is_integer()``.

Applies clean with ``git apply`` on the pristine 620c76f2 file (verified
2026-10-24); the anchor ``return int(raw)`` is unique in the file. NOTE for the
next pin migration: upstream #27ac614f/#40479 restructures the parser around
checkpoint-driven response-template adapters — re-derive this patch against
that layout (or drop it if absorbed).

Idempotent. Exits non-zero on anchor drift. Optional argv[1] overrides the
target path.
"""
import pathlib
import sys


DEFAULT_TARGET = (
    "/sgl-workspace/sglang/python/sglang/srt/parser/chat_parsing/response_parser.py"
)

OLD = '''            if type_name == "integer":
                return int(raw)
            if type_name == "number":
'''

NEW = '''            if type_name == "integer":
                try:
                    return int(raw)
                except ValueError:
                    # JSON Schema integers include whole numbers written as floats, e.g. "3.0"
                    number = float(raw)
                    if not number.is_integer():
                        continue
                    return int(number)
            if type_name == "number":
'''

SENTINEL = "whole numbers written as floats"


def main() -> int:
    target = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET)
    if not target.is_file():
        print("ERROR: target not found: %s (base image changed?)" % target, file=sys.stderr)
        return 1
    src = target.read_text()
    if SENTINEL in src and OLD not in src:
        print("already patched: %s" % target)
        return 0
    n = src.count(OLD)
    if n != 1:
        print("ERROR: expected exactly 1 anchor match, found %d -- base changed, "
              "re-derive from sglang#43941" % n, file=sys.stderr)
        return 1
    src = src.replace(OLD, NEW)
    if SENTINEL not in src or OLD in src:
        print("ERROR: post-patch verification failed", file=sys.stderr)
        return 1
    target.write_text(src)
    print("patched %s: whole-number floats accepted for integer tool arguments" % target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
