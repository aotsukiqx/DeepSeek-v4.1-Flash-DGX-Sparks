# Config Arm: prefill graphs + verify/draft family ON (zero-code optimization)

> Expected: +5-12% cold prefill & decode recovery without any re-port work.
> Cost: one boot window (~25 min) with auto-rollback.
> Depends on: mig image (canary-mig) already boots successfully.

## What to change

Only the `.env` (staged as `/tmp/env.mig` on spark-eb1a). Add these to `EXTRA_CONTAINER_ENV`:

```
# Prefill CUDA graphs (new pin supports them; old pin disabled due to incompatibility)
# No specific env var needed — they should be ON by default on 7d996300.
# If boot log doesn't show "Capture prefill graph" lines, investigate --enable-prefill-cuda-graph flag.

# Verify/draft family (all passed install on 7d996300, were turned OFF for first boot)
DSV41_VERIFY_CAP=conf:0.1
DSV41_BLOCK_VERIFY=1
DSV41_DRAFT_TAU=0.7
DSV41_WO_A_W8=1
DSV41_DRAFT_HEAD_FP8=1
DSV41_DRAFT_HEAD_FP8_IMPL=tp4
DSV41_AUTOTUNE_KEEP=1
DSV41_FOLDED_FENCE=1
DSV41_L2_PREFETCH=1
```

## What NOT to change (keep from first measurement)

```
EP_SIZE=1
DSV41_MOE_B12X_NEXT=1 (+ GRAPH_BS + TUNE_ROWS)
DSV41_FAST_LOAD=1 + TP_SLICE=on + INFLIGHT_GB=6
DSV41_PREFILL_EMPTY_CACHE_EVERY=4
DSV41_PREFILL_SP=0 (still OFF — needs structural re-port)
DSV41_HC_FUSED=0 (still OFF — needs kernel path update)
```

## Verification protocol

1. Boot (window script handles stop→serve→health-wait→proof-gate)
2. Boot gates: all green (same as first measurement)
3. Wire smokes: 4/4 PASS
4. **Key check**: DSpark accept len ≥ 3.2 (verify_cap/block_verify ON should maintain this)
5. Probes: cold 600k ×2 (discard + measured) + decode ×2 + conc6
6. Compare vs:
   - First mig measurement: cold 3953, decode 37.8, conc6 4021
   - Production baseline: cold 4101, decode 44.0, conc6 3884
7. qeval paired vs production baseline (BROKE=0)

## Success criteria

| Metric | Target | Meaning |
|---|---|---|
| cold 600k | ≥4100 | At or above production baseline |
| decode | ≥40 | Recovering from 37.8 |
| conc6 | ≥4000 | Maintaining upstream gain |
| accept len | ≥3.2 | DSpark working correctly with verify family |

## Rollback

Window script auto-rolls back. No code changes to undo.

## Risks

- Some of the verify/draft adapters may hit drift at SERVE time (not import time) —
  their needles were checked at the import level but serve-time behavior differs.
- If any adapter crashes at boot, turn it off individually and retry.
- prefill CUDA graphs may consume extra memory during capture — watch for OOM.
