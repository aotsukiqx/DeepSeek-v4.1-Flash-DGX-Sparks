#!/usr/bin/env python3
"""runtime/patch_response_parser_43941.py against a synthetic stand-in.

Reproduces only the anchored region of chat_parsing.response_parser._coerce
plus the helpers it names; NOT SGLang's file. Behavior check after patching:
"3.0" coerces to integer 3 (stock drops the whole argument set), "3.5" is
rejected exactly as before.
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / 'runtime' / 'patch_response_parser_43941.py'

FIXTURE = '''def _coerce(raw, types):
    for type_name in types:
        try:
            if type_name == "integer":
                return int(raw)
            if type_name == "number":
                number = float(raw)
                return number
        except ValueError:
            continue
    return None
'''


def run(path):
    return subprocess.run([sys.executable, '-I', str(PATCHER), str(path)],
                          capture_output=True, text=True)


def load(path):
    spec = importlib.util.spec_from_file_location('rp43941', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'rp.py'
        path.write_text(FIXTURE)
        stock = load(path)
        assert stock._coerce("3", ("integer",)) == 3
        assert stock._coerce("3.0", ("integer",)) is None, 'fixture: stock drops 3.0'
        first = run(path)
        assert first.returncode == 0, first.stderr
        text_after_first = path.read_text()
        second = run(path)
        assert second.returncode == 0 and 'already patched' in second.stdout, second.stdout
        assert path.read_text() == text_after_first, 'second run must not change the file'

        mod = load(path)
        assert mod._coerce("3", ("integer",)) == 3
        assert mod._coerce("3.0", ("integer",)) == 3, 'whole-number float must coerce'
        assert mod._coerce("3.5", ("integer", "number")) == 3.5, 'fractional falls through to number'
        assert mod._coerce("x", ("integer",)) is None

        bad = Path(d) / 'bad.py'
        bad.write_text('def other():\n    pass\n')
        assert run(bad).returncode != 0, 'a missing anchor must fail loudly'
    print('ok: 43941 backport applies, is idempotent, fails on a missing anchor')
    return 0


if __name__ == '__main__':
    sys.exit(main())
