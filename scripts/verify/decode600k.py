#!/usr/bin/env python3
"""Single-stream decode @600k, net of TTFT — RECONSTRUCTED 2026-10-25.

Same generator as probe600k.py. Prewarm: one non-stream 600k call (cold prefill);
then a streamed call on the same text (prefix-cached prefill, TTFT ~2s) whose
decode-only phase gives decode_tok_s_net.
Usage: decode600k.py <seed>
"""
import json, random, sys, time, urllib.request

BASE = "http://127.0.0.1:8000/v1/chat/completions"
WORDS = ("archive ventilation catalogued quarterly the maintenance log recorded routine "
         "inspections of heating pipes researchers folders by date region and subject "
         "afternoon light faded slowly rain streaked the tall windows committee postponed "
         "its decision until the next quarterly meeting harbor beacons automated ledger").split()

def gen_text(seed, target_tokens):
    rng = random.Random(seed); out = []
    n = int(target_tokens * 1000 / 1.1)
    while len(out) < n:
        if rng.random() < 0.08: out.append(f"Marker {rng.randint(1000,9999)}:")
        out.extend(rng.choice(WORDS) for _ in range(rng.randint(4, 14)))
        out.append(".")
    return " ".join(out)

seed = int(sys.argv[1])
text = gen_text(seed, 600)
prompt = text + "\n\nWrite a detailed 500-word technical essay about lighthouses."

def call(mt, stream):
    body = {"model": "deepseek-v4.1-flash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mt, "temperature": 0, "stream": stream,
            "chat_template_kwargs": {"thinking": False}}
    if stream: body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=5400)
    if not stream:
        d = json.load(resp)
        return time.perf_counter() - t0, d["usage"]
    ctoks = 0; ttft = None
    for line in resp:
        if not line.startswith(b"data: "): continue
        payload = line[6:].strip()
        if payload == b"[DONE]": break
        try: ev = json.loads(payload)
        except Exception: continue
        d = ev.get("choices") or []
        if d:
            piece = d[0].get("delta", {}).get("content")
            if piece:
                if ttft is None: ttft = time.perf_counter() - t0
                ctoks += 1
        if ev.get("usage") and ttft is not None:
            ctoks = ev["usage"]["completion_tokens"]
    total = time.perf_counter() - t0
    return total, ttft, ctoks

prewarm_wall, usage = call(55, stream=False)
total, ttft, ctoks = call(500, stream=True)
decode_s = total - ttft
print(json.dumps({"seed": seed, "prewarm_s": round(prewarm_wall, 1),
                  "ptoks": usage["prompt_tokens"], "ttft_s": round(ttft, 2),
                  "ctoks": ctoks, "decode_s": round(decode_s, 1),
                  "decode_tok_s_net": round(ctoks / decode_s, 1)}), flush=True)
