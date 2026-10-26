#!/usr/bin/env python3
"""v4 -> v5: add input-scattered prefill flags to EXTRA_SGLANG_ARGS in /tmp/env.mig."""
p = "/tmp/env.mig"
s = open(p).read()
FLAG = " --enable-attn-tp-input-scattered --moe-dense-tp-size 4"
out, hit = [], 0
for line in s.splitlines(keepends=True):
    if line.startswith("EXTRA_SGLANG_ARGS=") and hit == 0:
        hit = 1
        body = line.rstrip("\n")
        assert body.endswith('"'), "no close quote"
        if "--enable-attn-tp-input-scattered" not in body:
            line = body[:-1] + FLAG + '"\n'
    out.append(line)
assert hit == 1
open(p, "w").write("".join(out))
print("env.mig v5 written (input-scattered flags added)")
