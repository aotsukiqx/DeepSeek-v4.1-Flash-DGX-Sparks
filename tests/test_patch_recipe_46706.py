#!/usr/bin/env python3
"""runtime/patch_recipe_46706.py against a synthetic stand-in.

Reproduces only the ten anchor regions of sglang main@7d996300's
serving_deepseek_recipe.py plus the stubs they name; it is NOT SGLang's file.
Behavior checks after patching: numeric-string efforts become numbers (request
effort AND the env default), null contents render as "", and a trailing
assistant turn loses the tool_calls SGLang's recipe discards (both the
continue path and the re-send path).
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / 'runtime' / 'patch_recipe_46706.py'

FIXTURE = '''from __future__ import annotations

from typing import Any, Optional, Union


class Req:
    def __init__(self, messages, continue_final=False, task=None, thinking=False):
        self._messages = messages
        self.continue_final_message = continue_final
        self.task = task
        self.thinking = thinking
        self.response_format = None
        self.messages = [Msg(m) for m in messages]

    def model_dump(self, **kw):
        return {"messages": [dict(m) for m in self._messages]}


class Msg:
    def __init__(self, m):
        self._m = m

    def model_dump(self, **kw):
        return dict(self._m)


class DeepSeekRecipeChatEncodingAdapter:
    def __init__(self, raw: str = ""):
        self._default_effort = int(raw) if raw.isdigit() else raw or None

    def encode_messages(self, request):
        wire = request.model_dump(exclude_unset=True, by_alias=True, mode="json")
        wire_messages = wire["messages"]
        continue_final = request.continue_final_message
        task_prefix = None
        last = _trailing_assistant_turn(wire_messages)
        if last is not None:
            if request.task:
                task_prefix = wire_messages.pop()["content"]
                continue_final = False
        body = {
            "messages": wire_messages,
            "thinking": request.thinking,
        }
        return body

    def encode_effort(
        self,
        messages=None,
        thinking=False,
        effort=None,
        task=None,
        image_urls: Optional[list[str]] = None,
    ) -> list[int]:
        if effort is None or effort == "none":
            effort = self._default_effort
        return effort

    def continued_turn(self, request):
        response_format = request.response_format
        continued = (
            _trailing_assistant_turn(
                [
                    message.model_dump(exclude_unset=True, by_alias=True, mode="json")
                    for message in request.messages[-1:]
                ]
            )
            if request.continue_final_message
            else None
        )
        return continued


class DeepSeekRecipeResponsesStream:
    def delta_items(self, delta):
        return delta.reasoning, delta.content, _tool_call_items(delta.tool_calls)


def _tool_call_items(items):
    return items


def _trailing_assistant_turn(
    messages: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """The trailing text turn SGLang continues or, without continue_final_message,
    re-sends as a user turn (as `_handle_last_assistant_message`)."""
    last = messages[-1] if messages else None
    if last and last["role"] == "assistant" and isinstance(last.get("content"), str):
        return last
    return None


def _stop_strings(stop: Union[str, list[str], None]) -> list[str]:
    return [stop] if isinstance(stop, str) else stop or []
'''


def run(path):
    return subprocess.run([sys.executable, '-I', str(PATCHER), str(path)],
                          capture_output=True, text=True)


def load(path, name='recipe46706'):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod




def main():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        path = base / 'sglang/srt/entrypoints/openai/serving_deepseek_recipe.py'
        path.parent.mkdir(parents=True)
        path.write_text(FIXTURE)

        stock = load(path, 'recipe46706_stock')
        trailing = [{"role": "assistant", "content": "partial", "tool_calls": [{"id": 1}]}]
        assert stock._trailing_assistant_turn(trailing)["tool_calls"] == [{"id": 1}], \
            'fixture: stock keeps tool_calls'
        assert stock.DeepSeekRecipeChatEncodingAdapter("0.3")._default_effort == "0.3", \
            'fixture: stock leaves numeric-string default effort as string'
        assert stock.DeepSeekRecipeChatEncodingAdapter().encode_messages(
            stock.Req([{"role": "user", "content": None}]))["messages"][0]["content"] is None, \
            'fixture: stock passes null content through'

        first = run(base)
        assert first.returncode == 0, first.stderr
        text_after_first = path.read_text()
        second = run(base)
        assert second.returncode == 0 and 'already patched' in second.stdout, second.stdout
        assert path.read_text() == text_after_first, 'second run must not change the file'

        mod = load(path)
        # numeric-string effort -> the number it spells (request and default)
        assert mod._numeric_effort("30") == 30 and isinstance(mod._numeric_effort("30"), int)
        assert mod._numeric_effort("0.3") == 0.3
        assert mod._numeric_effort("high") == "high"
        assert mod._numeric_effort(None) is None
        assert mod.DeepSeekRecipeChatEncodingAdapter("0.3")._default_effort == 0.3
        assert mod.DeepSeekRecipeChatEncodingAdapter("30")._default_effort == 30
        # null content renders as ""
        fixed = mod._null_content_as_empty(
            [{"role": "user", "content": None}, {"role": "user", "content": "hi"}])
        assert fixed[0]["content"] == "" and fixed[1]["content"] == "hi"
        # trailing assistant turn loses tool_calls
        stripped = mod._trailing_assistant_turn(trailing)
        assert stripped == {"role": "assistant", "content": "partial"}, stripped
        # encode_messages: null content + tool-call turn write-back
        body = mod.DeepSeekRecipeChatEncodingAdapter().encode_messages(
            mod.Req([{"role": "user", "content": None}] + trailing, continue_final=True))
        assert body["messages"][0]["content"] == ""
        assert body["messages"][-1] == {"role": "assistant", "content": "partial"}, \
            body["messages"][-1]
        # continued_turn wraps the dump with the same two fixes
        cont = mod.DeepSeekRecipeChatEncodingAdapter().continued_turn(
            mod.Req(trailing + [{"role": "assistant", "content": None, "tool_calls": []}],
                continue_final=True))
        assert cont == {"role": "assistant", "content": ""}, cont
        # request-level numeric-string effort normalized before the none-check
        assert mod.DeepSeekRecipeChatEncodingAdapter().encode_effort(effort="0.3") == 0.3
        assert mod.DeepSeekRecipeChatEncodingAdapter().encode_effort(effort="none") is None
        assert 'import re' in text_after_first.splitlines()[2]

        # anchor drift fails loudly
        path.write_text(FIXTURE.replace('wire["messages"]', 'wire["messagesX"]'))
        drifted = run(base)
        assert drifted.returncode == 1 and 're-derive from sglang#46706' in drifted.stderr, \
            drifted.stderr

    print('test_patch_recipe_46706 PASS')


if __name__ == '__main__':
    main()
