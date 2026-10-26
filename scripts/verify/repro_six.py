#!/usr/bin/env python3
"""Reproduce the six-adapter duplicate-invariant boot crash in 30s (imports only).

Usage (in canary-mig image):
  python3 repro_six.py all         # all six adapters on
  python3 repro_six.py minus VERIFY_CAP   # all but one
"""
import os
import sys

SIX = {
    "VERIFY_CAP": "conf:0.1", "BLOCK_VERIFY": "1", "DRAFT_TAU": "0.7",
    "WO_A_W8": "1", "AUTOTUNE_KEEP": "1", "FOLDED_FENCE": "1",
}

mode = sys.argv[1] if len(sys.argv) > 1 else "all"
drop = mode[len("minus "):] if mode.startswith("minus ") else None
for k, v in SIX.items():
    if k != drop:
        os.environ[f"DSV41_{k}"] = v
os.environ["DSV41_WO_A_W8_DROP"] = "1"
os.environ["DSV41_WO_A_W8_MID"] = "1"
os.environ["DSV41_LAUNCHER"] = "tp4"
os.environ.setdefault("DSV41_ENGRAM_PREFETCH", "0")

sys.path.insert(0, "/opt/dsv41/adapter")
import sitecustomize  # noqa: F401  (installs the EngramLoader)

print(f"[repro] mode={mode} drop={drop}", flush=True)

# Phase 1: the model-load-time chain (what ran before the crash point)
import sglang.srt.models.deepseek_v4_dspark  # noqa: F401
print("[repro] deepseek_v4_dspark imported (hooks ran)", flush=True)

# Phase 2: the scheduler draft-worker chain that crashed
import sglang.srt.speculative.dspark_components.dspark_worker_v2  # noqa: F401
print("[repro] dspark_worker_v2 imported OK — no duplicate invariant", flush=True)

from sglang.srt.utils.invariants import _REGISTRY  # type: ignore
planner = sys.modules.get("sglang.srt.speculative.dspark_components.dspark_planner")
print(f"[repro] planner in sys.modules: {planner is not None}, "
      f"__name__={getattr(planner, '__name__', None)!r}, "
      f"invariant registered: {'dspark.verify_len_budget' in _REGISTRY}", flush=True)
