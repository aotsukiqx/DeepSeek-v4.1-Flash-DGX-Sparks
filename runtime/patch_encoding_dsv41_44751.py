#!/usr/bin/env python3
"""Backport sglang#44751 (a038a0dc) onto sglang main@620c76f2.

[Fix] dsv41: Preserve tool-result image order and current-turn thinking.
Four changes to python/sglang/srt/entrypoints/openai/encoding_dsv41.py:

1. ``find_last_user_index``: a user message whose content is only tool-result
   blocks does not start a new assistant turn. Without this, every
   tool-result round-trip reset the turn boundary and dropped the current
   turn's thinking header (long agent sessions lost reasoning mid-flight).
2. ``merge_tool_messages`` (tool branch): a tool message that carries
   ``content_blocks`` (e.g. images captured by a tool) passes them through
   instead of being flattened to ``msg.get("content", "")``.
3. ``merge_tool_messages`` (user branch): list-form user content is used
   as-is instead of being wrapped as one text block.
4. ``encode_messages`` validates and merges BEFORE image collection, so
   images are gathered only after tool results are in their final prompt
   order (the old code sorted tool results per-segment, scrambling
   multi-image tool transcripts); ``_encode_messages_text`` no longer
   re-merges (its only caller now pre-merges).

The upstream commit applies clean with ``git apply`` on the pristine
620c76f2 file (verified 2026-10-18) and after patch_encoding_dsv41.py;
this script re-expresses it as count==1 string anchors so a base change
fails loudly instead of silently dropping the fix. Anchors are NOT valid
for the dev-dsv41/dsv4.1-branch file.

Idempotent per anchor. Exits non-zero on anchor drift. Optional argv[1]
overrides the target path.
"""
import pathlib
import sys

DEFAULT_TARGET = (
    "/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/encoding_dsv41.py"
)

# ------------------------------------------------------------------ anchor 1
OLD1 = '''def find_last_user_index(messages: List[Dict[str, Any]]) -> int:
    """Mid-conversation system messages also count as user messages here;
    they trigger the assistant generation header.
    """
    last_user_index = -1
    for idx in range(len(messages) - 1, -1, -1):
        role = messages[idx].get("role")
        if role in ["user", "developer"] or (role == "system" and idx > 0):
            last_user_index = idx
            break
    return last_user_index
'''

NEW1 = '''def find_last_user_index(messages: List[Dict[str, Any]]) -> int:
    """Mid-conversation system messages also count as user messages here;
    they trigger the assistant generation header.
    User messages containing only tool results do not start a new turn.
    """
    last_user_index = -1
    for idx in range(len(messages) - 1, -1, -1):
        role = messages[idx].get("role")
        if role in ["user", "developer"] or (role == "system" and idx > 0):
            content_blocks = messages[idx].get("content_blocks")
            if content_blocks and not any(
                block.get("type") in ("text", "image", "image_url")
                for block in content_blocks
            ):
                continue
            last_user_index = idx
            break
    return last_user_index
'''

# ------------------------------------------------------------------ anchor 2
OLD2 = '''        if role == "tool":
            # Convert tool message to a user message with tool_result block
            tool_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
'''

NEW2 = '''        if role == "tool":
            # Convert tool message to a user message with tool_result block
            content = msg.get("content", "")
            if msg.get("content_blocks") and not isinstance(msg.get("content"), str):
                content = msg["content_blocks"]
            tool_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": content,
            }
'''

# ------------------------------------------------------------------ anchor 3
OLD3 = '''            content_blocks = msg.get("content_blocks")
            if content_blocks is None:
                content_blocks = [{"type": "text", "text": msg.get("content", "")}]
'''

NEW3 = '''            content_blocks = msg.get("content_blocks")
            if content_blocks is None:
                content = msg.get("content", "")
                content_blocks = (
                    content
                    if isinstance(content, list)
                    else [{"type": "text", "text": content}]
                )
'''

# ------------------------------------------------------------------ anchor 4
# _encode_messages_text: drop the per-segment preprocessing (the single
# caller, encode_messages, now merges once before image collection).
OLD4 = '''    context = context if context else []

    # Preprocess: merge tool messages and sort tool results
    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context) :]
    if context:
        context = merge_tool_messages(context)
        context = sort_tool_results_by_call_order(context)

    full_messages = context + messages
'''

NEW4 = '''    context = context if context else []
    full_messages = context + messages
'''

# ------------------------------------------------------------------ anchor 5
OLD5 = '''    context = context or []
    processed_context, _ = process_image_messages(context) if context else ([], [])
'''

NEW5 = '''    # Validate before merging can discard message-level textual fields.
    for msg in (context or []) + messages:
        _validate_no_image_sp_tokens(msg)

    # Collect images only after tool results are in their final prompt order.
    context = merge_tool_messages(context or [])
    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context) :]
    processed_context, _ = process_image_messages(context) if context else ([], [])
'''

ANCHORS = [
    ("find_last_user_index", OLD1, NEW1),
    ("merge_tool_messages(tool)", OLD2, NEW2),
    ("merge_tool_messages(user)", OLD3, NEW3),
    ("_encode_messages_text head", OLD4, NEW4),
    ("encode_messages head", OLD5, NEW5),
]

SENTINELS = [
    "User messages containing only tool results do not start a new turn.",
    'content = msg["content_blocks"]',
    "if isinstance(content, list)",
    "Collect images only after tool results are in their final prompt order.",
]


def verify(src: str) -> list:
    problems = []
    for name, old, _ in ANCHORS:
        if old in src:
            problems.append("anchor %s still pristine" % name)
    for s in SENTINELS:
        if s not in src:
            problems.append("sentinel missing: %s" % s[:48])
    return problems


def main() -> int:
    target = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET)
    if not target.is_file():
        print("ERROR: target not found: %s (base image changed?)" % target, file=sys.stderr)
        return 1
    src = target.read_text()
    if verify(src) == []:
        print("already patched: %s" % target)
        return 0
    changed = []
    for name, old, new in ANCHORS:
        if new in src and old not in src:
            continue  # this anchor already applied
        n = src.count(old)
        if n != 1:
            print("ERROR: anchor %s: expected exactly 1 match, found %d -- "
                  "base changed, re-derive from sglang#44751" % (name, n), file=sys.stderr)
            return 1
        src = src.replace(old, new)
        changed.append(name)
    problems = verify(src)
    if problems:
        print("ERROR: post-patch verification failed: %s" % "; ".join(problems),
              file=sys.stderr)
        return 1
    target.write_text(src)
    print("patched %s: %s" % (target, ", ".join(changed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
