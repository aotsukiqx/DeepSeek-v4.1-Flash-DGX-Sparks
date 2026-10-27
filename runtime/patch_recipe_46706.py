#!/usr/bin/env python3
"""Backport sglang#46706 (449617bc, py half) onto sglang main@7d996300.

[Fix] dsv4: Render null contents, trailing tool-call turns and numeric string
efforts with deepseek-recipe.

Three serving-layer bugs in DeepSeekRecipeChatEncodingAdapter: (1) a message
with ``content: null`` reached the recipe encoder as None instead of ""
(_null_content_as_empty); (2) the trailing assistant turn kept its tool_calls
when re-sent/continued — the recipe discards them, so the round-trip must strip
them (_trailing_assistant_turn + the else-branch write-back); (3) a numeric
STRING effort ("30", "0.3") stayed a string instead of becoming the number it
spells (_numeric_effort, applied to both the request effort and the env default).

Only the python half is backported: the commit's rust half touches
rust/sglang-processor (the sgl-router renderer), which this image does not
build (canary-mig compiles only the sglang-mm package).

Applies clean with ``git apply`` on the pristine 7d996300 file (verified
2026-10-27). PIN-MIGRATION NOTE: serving_deepseek_recipe.py is the successor
surface of the retired encoding_dsv41.py line; re-derive this patch against
whatever #40479/#46xx does to the recipe entrypoint, or drop it if absorbed.

Idempotent. Exits non-zero on anchor drift (these scripts are the
build's drift probes). Optional argv[1] overrides the sglang python
tree root (default: the in-image path).
"""
import pathlib
import sys

DEFAULT_BASE = "/sgl-workspace/sglang/python"
PR = "46706"

FILES = [
    (
        '''sglang/srt/entrypoints/openai/serving_deepseek_recipe.py''',
        '''        self._default_effort = _numeric_effort(raw) if raw else None''',
        [
            ('''from __future__ import annotations

''',
             '''from __future__ import annotations

import re
'''),
            ('''        self._default_effort = int(raw) if raw.isdigit() else raw or None
''',
             '''        self._default_effort = _numeric_effort(raw) if raw else None
'''),
            ('''        wire_messages = wire["messages"]
''',
             '''        wire_messages = _null_content_as_empty(wire["messages"])
'''),
            ('''                continue_final = False
''',
             '''                continue_final = False
            else:
                wire_messages[-1] = last
'''),
            ('''        image_urls: Optional[list[str]] = None,
    ) -> list[int]:
''',
             '''        image_urls: Optional[list[str]] = None,
    ) -> list[int]:
        effort = _numeric_effort(effort)
'''),
            ('''                [
                    message.model_dump(exclude_unset=True, by_alias=True, mode="json")
                    for message in request.messages[-1:]
                ]
''',
             '''                _null_content_as_empty(
                    [
                        message.model_dump(
                            exclude_unset=True, by_alias=True, mode="json"
                        )
                        for message in request.messages[-1:]
                    ]
                )
'''),
            ('''        return delta.reasoning, delta.content, _tool_call_items(delta.tool_calls)


''',
             '''        return delta.reasoning, delta.content, _tool_call_items(delta.tool_calls)


def _null_content_as_empty(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """As `_process_messages` renders them: a null content is ""."""
    return [
        message if message.get("content") is not None else {**message, "content": ""}
        for message in messages
    ]


'''),
            ('''    re-sends as a user turn (as `_handle_last_assistant_message`)."""
''',
             '''    re-sends as a user turn (as `_handle_last_assistant_message`), without the
    tool calls SGLang discards."""
'''),
            ('''        return last
''',
             '''        return {key: value for key, value in last.items() if key != "tool_calls"}
'''),
            ('''    return None
''',
             r'''    return None


def _numeric_effort(
    effort: Union[str, int, float, None],
) -> Union[str, int, float, None]:
    """A numeric string effort ("30", "0.3") as the number it spells."""
    if isinstance(effort, str) and re.fullmatch(r"\d+(\.\d+)?", effort.strip()):
        return float(effort) if "." in effort else int(effort)
    return effort
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
