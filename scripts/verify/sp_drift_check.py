"""prefill_sp anchor drift check against the candidate tree (in-container, CPU-only).

Replicates prefill_sp._src_hash/_resolve against the mig image's sglang modules and
reports MATCH/DRIFT/MISSING for all 23 anchors. Run:
  docker run --rm --entrypoint /bin/bash -e CUDA_VISIBLE_DEVICES= \
    -v /tmp/sp_drift_check.py:/tmp/sp_drift_check.py dsv41-4x-spark:canary-mig \
    -c "python3 /tmp/sp_drift_check.py"
"""
import sys, hashlib, inspect

sys.path.insert(0, "/opt/dsv41/adapter")
import prefill_sp as ps

from sglang.srt.models import deepseek_v4 as v4
from sglang.srt.models import deepseek_v2 as v2
from sglang.srt.layers import engram, linear
from sglang.kernels.ops.layernorm import hc_combine_norm as hcn

M = dict(v4=v4, v2=v2, engram=engram, hcn=hcn, linear=linear)

def resolve(name):
    mod, attr = name.split(".", 1)
    obj = M[mod]
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj

def src_hash(obj):
    obj = getattr(obj, "fn", obj)
    obj = inspect.unwrap(obj)
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()[:16]

tables = [("MAIN", ps._EXPECTED), ("FP8", ps._EXPECTED_FP8), ("TAIL", ps._EXPECTED_TAIL)]
n_ok = n_drift = n_missing = 0
for label, tbl in tables:
    print(f"=== {label} ===", flush=True)
    for k, want in sorted(tbl.items()):
        try:
            got = src_hash(resolve(k))
        except AttributeError as e:
            print(f"MISSING {k}: {e}")
            n_missing += 1
            continue
        if got == want:
            n_ok += 1
            print(f"MATCH  {k}")
        else:
            n_drift += 1
            print(f"DRIFT  {k}: want={want} got={got}")
print(f"\nTOTAL: {n_ok} match, {n_drift} drift, {n_missing} missing")
