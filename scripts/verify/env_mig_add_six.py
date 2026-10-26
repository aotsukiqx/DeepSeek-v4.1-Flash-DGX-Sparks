#!/usr/bin/env python3
"""Add the six production-proven adapter vars to /tmp/env.mig's EXTRA_CONTAINER_ENV."""
import sys

p = "/tmp/env.mig"
s = open(p).read()
ADD = [
    "DSV41_VERIFY_CAP=conf:0.1", "DSV41_BLOCK_VERIFY=1", "DSV41_DRAFT_TAU=0.7",
    "DSV41_WO_A_W8=1", "DSV41_WO_A_W8_DROP=1", "DSV41_WO_A_W8_MID=1",
    "DSV41_AUTOTUNE_KEEP=1", "DSV41_FOLDED_FENCE=1",
    # Pin gate: reactivated adapter paths (draft_tau / folded_fence / verify_cap
    # executor wraps) stay frozen on 620c76f2 production; active on the mig arm.
    "DSV41_PIN7=1",
]
PRE = 'EXTRA_CONTAINER_ENV="'
out, hit = [], 0
for line in s.splitlines(keepends=True):
    if line.startswith(PRE) and hit == 0:
        hit = 1
        body = line.rstrip("\n")
        assert body.endswith('"'), "line does not end with a close quote"
        toks = [t for t in body[len(PRE):-1].split() if t not in ADD]
        line = PRE + " ".join(toks + ADD) + '"\n'
    out.append(line)
assert hit == 1, "EXTRA_CONTAINER_ENV line not found"
open(p, "w").write("".join(out))
print("env.mig v3 written cleanly")
