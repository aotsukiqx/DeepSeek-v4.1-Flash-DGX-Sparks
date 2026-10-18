#!/usr/bin/env python3
"""runtime/patch_serving_chat_43006.py against a synthetic stand-in.

The stand-in reproduces only the anchored region of serving_chat's
_apply_jinja_template plus the minimal helpers it names; it is NOT SGLang's
file. Behavior check after patching: with tool_choice=none (tools=None on the
request path the anchor keys off), the system message carries no tools.
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / 'runtime' / 'patch_serving_chat_43006.py'

FIXTURE = '''class _Tool:
    def __init__(self, name):
        self.name = name
    def model_dump(self):
        return {"type": "function", "name": self.name}

class _Req:
    def __init__(self, tools):
        self.tools = tools

def dsv41_tool_payload(tool):
    return {"type": "function", "name": tool.name}

chat_encoding = type("m", (), {"dsv41_tool_payload": dsv41_tool_payload})

def _apply_jinja_template(messages, request, tools, is_dsv41=True):
    if True:
        if True:
            # An empty system message hosts the request tools; dsv41 renders a
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
    return messages
'''


def run(path):
    return subprocess.run([sys.executable, '-I', str(PATCHER), str(path)],
                          capture_output=True, text=True)


def load(path):
    spec = importlib.util.spec_from_file_location('sc43006', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'sc.py'
        path.write_text(FIXTURE)
        stock = load(path)
        msgs = [{"role": "user", "content": "hi"}]
        out = stock._apply_jinja_template(list(msgs), stock._Req([stock._Tool("f")]), tools=True)
        assert out[0].get("tools"), 'fixture: stock hosts tools'
        first = run(path)
        assert first.returncode == 0, first.stderr
        text_after_first = path.read_text()
        second = run(path)
        assert second.returncode == 0 and 'already patched' in second.stdout, second.stdout
        assert path.read_text() == text_after_first, 'second run must not change the file'

        mod = load(path)
        # tool_choice=none path: the request carries tools but the engine passes
        # tools=None down (request_tools becomes None) -> no tool hosting.
        out = mod._apply_jinja_template(list(msgs), mod._Req([mod._Tool("f")]), tools=None)
        assert "tools" not in out[0] and out[0]["role"] == "user", out
        # tool_choice auto: tools resolved -> hosted as before.
        out = mod._apply_jinja_template(list(msgs), mod._Req([mod._Tool("f")]), tools=True)
        assert out[0].get("tools") == [{"type": "function", "name": "f"}], out

        bad = Path(d) / 'bad.py'
        bad.write_text('def other():\n    pass\n')
        assert run(bad).returncode != 0, 'a missing anchor must fail loudly'
    print('ok: 43006 backport applies, is idempotent, fails on a missing anchor')
    return 0


if __name__ == '__main__':
    sys.exit(main())
