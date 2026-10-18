#!/usr/bin/env python3
"""Backport sglang#43006 (4d5692b6) onto sglang main@620c76f2.

fix(openai): honor tool_choice=none in DeepSeek encoders.

In ``serving_chat._apply_jinja_template``, when the chat template resolves
``tools`` itself, the request's tools were unconditionally hosted in an
injected empty system message — even when the client sent
``tool_choice="none"``. The model then saw (and sometimes called) tools on a
turn that was explicitly told not to use any. The fix derives
``request_tools = request.tools if tools else None`` and keys both the
system-message injection and the tool hosting off that, so a ``none`` choice
renders a plain turn.

The upstream commit applies clean with ``git apply`` on the pristine 620c76f2
file (verified 2026-10-18); this script re-expresses it as a count==1 string
anchor so a base change fails loudly instead of silently dropping the fix.
Anchors are NOT valid for the dev-dsv41/dsv4.1-branch file.

Idempotent. Exits non-zero on anchor drift. Optional argv[1] overrides the
target path.
"""
import pathlib
import sys

DEFAULT_TARGET = (
    "/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py"
)

OLD = '''            # An empty system message hosts the request tools; dsv41 renders a
            # system token for it, so it only gets one when tools need the host.
            if messages[0]["role"] != "system" and (request.tools or not is_dsv41):
                messages.insert(0, {"role": "system", "content": ""})
            if request.tools:
                messages[0]["tools"] = [
                    (
                        chat_encoding.dsv41_tool_payload(tool)
                        if is_dsv41
                        else tool.model_dump()
                    )
                    for tool in request.tools
                ]
'''

NEW = '''            request_tools = request.tools if tools else None

            # An empty system message hosts the request tools; dsv41 renders a
            # system token for it, so it only gets one when tools need the host.
            if messages[0]["role"] != "system" and (request_tools or not is_dsv41):
                messages.insert(0, {"role": "system", "content": ""})
            if request_tools:
                messages[0]["tools"] = [
                    (
                        chat_encoding.dsv41_tool_payload(tool)
                        if is_dsv41
                        else tool.model_dump()
                    )
                    for tool in request_tools
                ]
'''

SENTINEL = "request_tools = request.tools if tools else None"


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
              "re-derive from sglang#43006" % n, file=sys.stderr)
        return 1
    src = src.replace(OLD, NEW)
    if SENTINEL not in src or OLD in src:
        print("ERROR: post-patch verification failed", file=sys.stderr)
        return 1
    target.write_text(src)
    print("patched %s: tool_choice=none honored in DeepSeek encoders" % target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
