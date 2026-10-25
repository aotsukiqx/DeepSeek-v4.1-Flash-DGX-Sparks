#!/usr/bin/env python3
"""H0b: decode throughput vs concurrent long-context load.

For each tier N in 1,2,4,6: launch N concurrent 600k sessions; each first does a
non-stream prefill (staggered so prefills serialize naturally), then all stream a
500-token decode together. Report per-stream net decode tok/s per tier.
Usage: decode_matrix.py
"""
import json, random, sys, threading, time, urllib.request

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

def one(seed, results, idx, barrier):
    prompt = gen_text(seed, 600) + "\n\nWrite a detailed 500-word technical essay about lighthouses."
    def post(mt, stream):
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
            json.load(resp); return time.perf_counter() - t0
        ttft = None; ctoks = 0
        for line in resp:
            if not line.startswith(b"data: "): continue
            p = line[6:].strip()
            if p == b"[DONE]": break
            try: ev = json.loads(p)
            except Exception: continue
            ch = ev.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content") and ttft is None:
                ttft = time.perf_counter() - t0
            if ev.get("usage") and ttft is not None:
                ctoks = ev["usage"]["completion_tokens"]
        total = time.perf_counter() - t0
        return ttft, ctoks, total - ttft
    pre = post(55, stream=False)
    barrier.wait()
    ttft, ctoks, dec = post(500, stream=True)
    results[idx] = (pre, ttft, ctoks, dec)

for n in (1, 2, 4, 6):
    results = [None] * n
    barrier = threading.Barrier(n)
    ths = [threading.Thread(target=one, args=(200 + i, results, i, barrier)) for i in range(n)]
    for t in ths: t.start()
    for t in ths: t.join()
    rates = [r[2] / r[3] for r in results]
    agg = sum(r[2] for r in results) / max(r[3] for r in results)
    print(json.dumps({"tier": n, "per_stream_net": [round(x, 1) for x in rates],
                      "per_stream_net_median": round(sorted(rates)[len(rates)//2], 1),
                      "aggregate_net": round(agg, 1)}), flush=True)
print("MATRIX-DONE")
