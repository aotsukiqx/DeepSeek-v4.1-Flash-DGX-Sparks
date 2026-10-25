#!/usr/bin/env python3
"""600k probes — RECONSTRUCTED 2026-10-25 (originals aged out of /tmp).

Generator is byte-identical to the original decode600k.py/probe600k.py lineage
(12-word WORDS, 8% Marker lines, 4-14-word blocks, 1.1 tok/word target), so
per-seed prompt sizes match the historical runs (~597.3k for seed 1..3).

Usage:
  probe600k.py cold <seed>          single 600k prefill, non-stream
  probe600k.py conc <n> <seedbase>  <n> concurrent cold prefills, aggregate tok/s
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

def post(seed, mt=55, stream=False):
    body = {"model": "deepseek-v4.1-flash",
            "messages": [{"role": "user",
                          "content": gen_text(seed, 600) +
                          "\n\nWrite a detailed 500-word technical essay about lighthouses."}],
            "max_tokens": mt, "temperature": 0, "stream": stream,
            "chat_template_kwargs": {"thinking": False}}
    if stream: body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=5400))
    return time.perf_counter() - t0, d["usage"]

def cold(seed):
    wall, usage = post(seed)
    ptoks = usage["prompt_tokens"]
    print(json.dumps({"mode": "cold", "seed": seed, "ptoks": ptoks,
                      "wall_s": round(wall, 1),
                      "prefill_tok_s": round(ptoks / wall, 1)}), flush=True)

def conc(n, seedbase):
    results = [None] * n
    def work(i):
        s = seedbase + i
        wall, usage = post(s)
        results[i] = (wall, usage["prompt_tokens"])
    t0 = time.perf_counter()
    ths = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in ths: t.start()
    for t in ths: t.join()
    makespan = time.perf_counter() - t0
    total = sum(r[1] for r in results)
    print(json.dumps({"mode": "conc", "n": n, "seedbase": seedbase,
                      "total_ptoks": total, "makespan_s": round(makespan, 1),
                      "aggregate_tok_s": round(total / makespan, 1),
                      "per_req_wall_s": [round(r[0], 1) for r in results]}), flush=True)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "cold":
        cold(int(sys.argv[2]))
    elif mode == "conc":
        conc(int(sys.argv[2]), int(sys.argv[3]))
    else:
        print(__doc__); sys.exit(2)
