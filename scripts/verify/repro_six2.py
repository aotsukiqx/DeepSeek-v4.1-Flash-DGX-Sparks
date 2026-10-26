#!/usr/bin/env python3
"""Faithful repro v2: trace Invariant registrations during the real import chain.

Patches invariants.Invariant to log each registration (name + caller file), then
imports the scheduler-side chain in the same order the scheduler process does,
with all six adapters enabled. Duplicates become visible with their sources.
"""
import os
import sys
import traceback

SIX = {
    "VERIFY_CAP": "conf:0.1", "BLOCK_VERIFY": "1", "DRAFT_TAU": "0.7",
    "WO_A_W8": "1", "AUTOTUNE_KEEP": "1", "FOLDED_FENCE": "1",
}
for k, v in SIX.items():
    os.environ[f"DSV41_{k}"] = v
os.environ["DSV41_WO_A_W8_DROP"] = "1"
os.environ["DSV41_WO_A_W8_MID"] = "1"
os.environ["DSV41_LAUNCHER"] = "tp4"
os.environ.setdefault("DSV41_ENGRAM_PREFETCH", "0")

sys.path.insert(0, "/opt/dsv41/adapter")
import sitecustomize  # noqa: F401

# Trace the registry BEFORE anything registers
import sglang.srt.utils.invariants as inv
_orig_init = inv.Invariant.__init__
REG = []

def traced_init(self, name, bucket, prop, **kw):
    src = traceback.extract_stack()[-3]
    REG.append((name, src.filename, src.lineno))
    print(f"[inv] {name} <- {src.filename}:{src.lineno}", flush=True)
    _orig_init(self, name, bucket, prop, **kw)

inv.Invariant.__init__ = traced_init

# The scheduler process chain (same order as run_scheduler_process):
import sglang.srt.managers.scheduler  # noqa: F401
print("[repro2] managers.scheduler imported", flush=True)

# Model load phase imports (what load_model triggers):
import sglang.srt.models.deepseek_v4  # noqa: F401
import sglang.srt.models.deepseek_v4_dspark  # noqa: F401
print("[repro2] model modules imported", flush=True)

# Draft worker init chain (the crash site):
import sglang.srt.speculative.dspark_components.dspark_worker_v2  # noqa: F401
print("[repro2] dspark_worker_v2 imported OK", flush=True)

names = [n for n, _, _ in REG]
dups = {n for n in names if names.count(n) > 1}
print(f"[repro2] registrations={len(names)} duplicates={dups or 'NONE'}", flush=True)
