#!/usr/bin/env python3
"""runtime/patch_encoding_dsv41_44751.py against a synthetic stand-in.

The stand-in reproduces only the five anchor regions of sglang
main@620c76f2's encoding_dsv41.py plus the helpers they call; it is NOT
SGLang's file. Behavior checks after patching: tool-result-only user
messages do not start a turn, tool content_blocks survive the merge, and
images are collected after the tool-result sort.
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / 'runtime' / 'patch_encoding_dsv41_44751.py'

FIXTURE = '''from typing import Any, Dict, List, Optional, Tuple

def find_last_user_index(messages: List[Dict[str, Any]]) -> int:
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

def merge_tool_messages(messages):
    merged = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            # Convert tool message to a user message with tool_result block
            tool_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            if merged and merged[-1].get("role") == "user" and "tool_result" in str(merged[-1].get("content")):
                merged[-1]["content"] = merged[-1]["content"] + [tool_block] if isinstance(merged[-1].get("content"), list) else [tool_block]
            else:
                merged.append({"role": "user", "content": [tool_block]})
        elif role == "user":
            content_blocks = msg.get("content_blocks")
            if content_blocks is None:
                content_blocks = [{"type": "text", "text": msg.get("content", "")}]
            merged.append({"role": "user", "content": content_blocks})
        else:
            merged.append(msg)
    return merged

def sort_tool_results_by_call_order(messages):
    return messages

def _validate_no_image_sp_tokens(msg):
    pass

def _encode_messages_text(messages, context):
    """Encode preprocessed (text-only) messages into the V4.1 prompt format."""
    context = context if context else []

    # Preprocess: merge tool messages and sort tool results
    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context) :]
    if context:
        context = merge_tool_messages(context)
        context = sort_tool_results_by_call_order(context)

    full_messages = context + messages
    return full_messages

def encode_messages(messages, context):
    context = context or []
    processed_context, _ = process_image_messages(context) if context else ([], [])
    processed_messages, images = process_image_messages(messages)
    prompt = _encode_messages_text(
        processed_messages, processed_context
    )
    return prompt, images

def process_image_messages(messages):
    images = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if block.get("type") == "tool_result":
                    for sub in block["content"] if isinstance(block["content"], list) else []:
                        if sub.get("type") == "image":
                            images.append(sub)
    return messages, images
'''


def run(path):
    return subprocess.run([sys.executable, '-I', str(PATCHER), str(path)],
                          capture_output=True, text=True)


def load(path):
    spec = importlib.util.spec_from_file_location('enc44751', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'enc.py'
        path.write_text(FIXTURE)
        stock = load(path)
        # Stock behavior this backport exists to fix:
        tool_only_user = {"role": "user",
                          "content_blocks": [{"type": "tool_result", "content": "r"}]}
        chat = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
                tool_only_user]
        assert stock.find_last_user_index(chat) == 2, 'fixture: stock sees a new turn'
        first = run(path)
        assert first.returncode == 0, first.stderr
        text_after_first = path.read_text()
        second = run(path)
        assert second.returncode == 0 and 'already patched' in second.stdout, second.stdout
        assert path.read_text() == text_after_first, 'second run must not change the file'

        mod = load(path)
        assert mod.find_last_user_index(chat) == 0, 'tool-result-only turn must be skipped'
        assert mod.find_last_user_index(chat[:2]) == 0
        merged = mod.merge_tool_messages([
            {"role": "assistant", "content": "t",
             "tool_calls": [{"id": "c1", "function": {"name": "f"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": None,
             "content_blocks": [{"type": "image", "source": {"x": 1}}]},
        ])
        blocks = [b for m in merged if isinstance(m.get("content"), list)
                  for b in m["content"]]
        assert any(isinstance(b, dict) and b.get("type") == "tool_result"
                   and any(isinstance(s, dict) and s.get("type") == "image"
                           for s in b["content"])
                   for b in blocks), merged
        img = {"type": "image", "source": {"x": 1}}
        tool_msg = {"role": "tool", "tool_call_id": "c1", "content": None,
                    "content_blocks": [img]}
        prompt, images = mod.encode_messages(
            [tool_msg, {"role": "user", "content": "go"}], [])
        assert images, 'images inside tool results must be collected'

        bad = Path(d) / 'bad.py'
        bad.write_text('def other():\n    pass\n')
        assert run(bad).returncode != 0, 'a missing anchor must fail loudly'
    print('ok: 44751 backport applies, is idempotent, fails on a missing anchor')
    return 0


if __name__ == '__main__':
    sys.exit(main())
