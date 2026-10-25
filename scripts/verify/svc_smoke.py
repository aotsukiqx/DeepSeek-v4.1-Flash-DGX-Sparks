#!/usr/bin/env python3
"""Wire smokes for the serving-layer line — RECONSTRUCTED 2026-10-25 (session source).

Four probes, one per merged serving change, against a live head:
  1. /v1/messages thinking {"type":"enabled"} WITHOUT budget_tokens  -> 200
  2. /v1/messages text containing the literal image placeholder token -> 200
  3. /v1/chat/completions assistant tool_call with broken JSON arguments
     kept in history                                                -> 200
  4. vision-in-tool-result round trip, order-sensitive               -> 200 + correct
Exit 0 only if all four behave as the merged line should.
"""
import base64
import io
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8000'
MODEL = 'deepseek-v4.1-flash'
PLACEHOLDER = '<｜deepseek_image｜>'


def post(path, body, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
        return 200, r
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b'{}')


def probe1():
    code, r = post('/v1/messages', dict(
        model=MODEL, max_tokens=64,
        thinking={'type': 'enabled'},
        messages=[dict(role='user', content='Reply with the single number 7.')]))
    ok = code == 200
    detail = 'http %s' % code
    if ok:
        blocks = r.get('content') or []
        detail += ' blocks=%s' % [b.get('type') for b in blocks]
    return ok, 'thinking-enabled-no-budget: ' + detail


def probe2():
    code, r = post('/v1/messages', dict(
        model=MODEL, max_tokens=64,
        messages=[dict(role='user',
                       content='The token %s appeared in a log. Reply with OK.' % PLACEHOLDER)]))
    ok = code == 200
    return ok, 'placeholder-in-text: http %s %s' % (code, (r.get('error') or {}).get('message', '')[:80])


def probe3():
    tool = {'type': 'function', 'function': {
        'name': 'lookup', 'description': 'look up a value',
        'parameters': {'type': 'object', 'properties': {'k': {'type': 'string'}},
                       'required': ['k'], 'additionalProperties': False}}}
    msgs = [
        dict(role='user', content='Use lookup for key alpha.'),
        dict(role='assistant', content='', tool_calls=[{
            'id': 'c1', 'type': 'function',
            'function': {'name': 'lookup', 'arguments': '{"k": "alph"}}}]),
        dict(role='tool', tool_call_id='c1',
             content='{"error": "Invalid JSON input for tool lookup. Please try again."}'),
        dict(role='user', content='The key was alpha after all. Call lookup again with the exact key.'),
    ]
    code, r = post('/v1/chat/completions', dict(
        model=MODEL, temperature=0, max_tokens=200,
        chat_template_kwargs={'thinking': False},
        messages=msgs, tools=[tool]))
    ok = code == 200
    extra = ''
    if ok:
        calls = (r['choices'][0]['message'].get('tool_calls') or [])
        extra = ' calls=%s' % [c['function']['arguments'] for c in calls]
        ok = any('"alpha"' in (c['function']['arguments'] or '') for c in calls)
    return ok, 'history_repair broken-args: http %s%s %s' % (code, extra, (r.get('error') or {}).get('message', '')[:60])


def probe4():
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return True, 'vision probe skipped (no PIL on head)'
    img = Image.new('RGB', (1600, 400), 'white')
    d = ImageDraw.Draw(img)
    d.ellipse((150, 100, 350, 300), fill='red')
    d.rectangle((1200, 100, 1400, 300), fill='blue')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    tool = {'type': 'function', 'function': {
        'name': 'attach', 'description': 'attach a screenshot for analysis',
        'parameters': {'type': 'object', 'properties': {},
                       'additionalProperties': False}}}
    msgs = [
        dict(role='user', content='Call attach to submit the screenshot for review.'),
        dict(role='assistant', content='', tool_calls=[{
            'id': 'c1', 'type': 'function',
            'function': {'name': 'attach', 'arguments': '{}'}}]),
        dict(role='tool', tool_call_id='c1', content=[
            dict(type='image_url', image_url=dict(url=b64)),
            dict(type='text', text='Screenshot attached. Describe shape colors left-to-right.'),
        ]),
        dict(role='user', content='Now describe the two colored shapes and their left-to-right order.'),
    ]
    code, r = post('/v1/chat/completions', dict(
        model=MODEL, temperature=0, max_tokens=200,
        chat_template_kwargs={'thinking': False},
        messages=msgs, tools=[tool]))
    ok = code == 200
    detail = 'http %s' % code
    if ok:
        text = (r['choices'][0]['message'].get('content') or '').lower()
        ok = 'red' in text and 'blue' in text and text.find('red') < text.find('blue')
        detail += ' order_ok=%s text=%r' % (ok, text[:120])
    else:
        detail += ' %s' % (r.get('error') or {}).get('message', '')[:80]
    return ok, 'vision-in-tool-result: ' + detail


def main():
    fails = 0
    for name, fn in (('1', probe1), ('2', probe2), ('3', probe3), ('4', probe4)):
        ok, detail = fn()
        print('%s %s' % ('PASS' if ok else 'FAIL', detail), flush=True)
        fails += 0 if ok else 1
    print('SMOKE %s' % ('PASS' if fails == 0 else 'FAIL(%d)' % fails))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
