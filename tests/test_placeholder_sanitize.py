#!/usr/bin/env python3
"""Host-side tests for the recipe-line placeholder sanitize adapter."""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'adapter'))
import placeholder_sanitize as ps


class FakeRenderer:
    pre_normalize = staticmethod(lambda body: body)


def install(fake=None):
    mod = fake or FakeRenderer()
    ps.install(mod)
    return mod


class PlaceholderSanitize(unittest.TestCase):
    def setUp(self):
        self.saved = ps.os.environ.get('DSV41_PLACEHOLDER_SANITIZE')
        ps.os.environ.pop('DSV41_PLACEHOLDER_SANITIZE', None)

    def tearDown(self):
        ps.os.environ.pop('DSV41_PLACEHOLDER_SANITIZE', None)
        if self.saved is not None:
            ps.os.environ['DSV41_PLACEHOLDER_SANITIZE'] = self.saved

    def test_text_fields_sanitized(self):
        mod = install()
        body = {'messages': [
            {'role': 'user', 'content': 'log line ' + ps.PLACEHOLDER + ' end'},
            {'role': 'assistant', 'content': 'ok',
             'reasoning_content': 'thought ' + ps.PLACEHOLDER},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'nested ' + ps.PLACEHOLDER},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + ps.PLACEHOLDER}},
            ]},
        ]}
        out = mod.pre_normalize(body)
        texts = [m['content'] for m in out['messages']]
        self.assertIn(ps.REPLACEMENT, texts[0])
        self.assertNotIn(ps.PLACEHOLDER, texts[0])
        self.assertNotIn(ps.PLACEHOLDER, out['messages'][1]['reasoning_content'])
        self.assertNotIn(ps.PLACEHOLDER, texts[2][0]['text'])
        # image data is text-in-dict too; sanitize is conservative and only touches
        # content/reasoning_content at the message level plus nested text — the
        # data URL here sits under image_url and must survive intact for real
        # images that do not contain the literal token.
        self.assertIn('data:image/png;base64,', texts[2][1]['image_url']['url'])

    def test_idempotent_install(self):
        mod = install()
        first = mod.pre_normalize
        ps.install(mod)
        self.assertIs(mod.pre_normalize, first)

    def test_disabled_keeps_original(self):
        ps.os.environ['DSV41_PLACEHOLDER_SANITIZE'] = '0'
        mod = FakeRenderer()
        before = mod.pre_normalize
        ps.install(mod)
        self.assertIs(mod.pre_normalize, before)

    def test_missing_target_is_tolerated(self):
        mod = SimpleNamespace()  # no pre_normalize
        ps.install(mod)  # must not raise


if __name__ == '__main__':
    unittest.main()
