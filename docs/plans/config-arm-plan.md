# Plan: Config-Arm Optimization (Immediate, Zero Code Changes)

> Goal: recover 5-12% on the mig arm by enabling adapters and features that are
> already compatible but currently env-OFF. One boot window (~25 min).

## What to Enable

### 1. Prefill CUDA Graphs
The new pin (7d996300) supports breakable prefill CUDA graphs; the old pin (620c76f2)
disabled them due to "incompatible with DeepSeek-V4, heavy capture-pool memory pressure"
(docs/600k-migration-20261011.md:393). The upstream wave (#37958 shared-read default-on,
#43733 replay-at-any-context, #45565 MLA prefill capture to 8192) has matured them.

**Expected**: +5-10% cold prefill (py-spy: ~10% host wall is CUTLASS-DSL marshaling that
graphs absorb).

**Action**: Add to EXTRA_SGLANG_ARGS in `/tmp/env.mig`:
```
--enable-prefill-cuda-graph
```

**Risk**: Capture-pool memory pressure on 128GB UMA (the original disable reason).
Mitigation: if boot OOMs, remove the flag and retry.

### 2. verify_cap + block_verify
These installed OK on the candidate (import-level, no drift-gate raise). They were
turned OFF in the arm env out of caution.

**Expected**: ~+2% decode (verify window control improves DSpark acceptance).

**Action**: Add to EXTRA_CONTAINER_ENV in `/tmp/env.mig`:
```
DSV41_VERIFY_CAP=conf:0.1 DSV41_BLOCK_VERIFY=1
```

**Risk**: The drift-gate may fire at serve time (install passed ≠ serve-time pass).
If boot refuses: remove these two vars.

### 3. draft_tau + wo_a_w8
Also installed OK at import level.

**Expected**: ~+2% decode (draft path optimization).

**Action**: Add to EXTRA_CONTAINER_ENV:
```
DSV41_DRAFT_TAU=0.7 DSV41_WO_A_W8=1
```

### 4. autotune_keep + folded_fence
Import-level pass, but these have drift-gates that may fire at serve.

**Expected**: ~+1% combined.

**Action**: Add:
```
DSV41_AUTOTUNE_KEEP=1 DSV41_FOLDED_FENCE=1
```

## Execution

1. Generate new `/tmp/env.mig` with all additions
2. Run `scripts/verify/mig_window.sh` (one boot + probes + rollback)
3. Compare vs arm-1 baseline: cold 3953, decode 37.8, conc6 4021
4. Success = any metric improves ≥2%

## Appendix: env.mig Generation Script

```python
# Run on Mac, then scp to spark-eb1a:/tmp/env.mig
# Based on production .env.tp4 with mig-arm deltas
import re, subprocess
prod = subprocess.run(['ssh','-n','spark-eb1a','cat ~/nvmodels/DeepSeek-v4.1-Flash-DGX-Sparks/.env.tp4'],
                      capture_output=True, text=True).stdout
OFF = ('DSV41_MOE_B12X_NEXT','DSV41_FAST_LOAD','DSV41_PREFILL_SP','DSV41_HC_FUSED',
       'DSV41_VERIFY_CAP','DSV41_BLOCK_VERIFY','DSV41_FOLDED_FENCE','DSV41_AUTOTUNE_KEEP',
       'DSV41_DRAFT_TAU','DSV41_WO_A_W8','DSV41_DRAFT_HEAD_FP8','DSV41_L2_PREFETCH',
       'DSV41_DRAFT_MAIN_PROJ_SPLIT','DSV41_DUAL_GRAPH','DSV41_COPY_DRAFTS',
       'DSV41_REPLICATED_SPLIT','DSV41_ROCE_GATHER','DSPARK_STS_TABLE')
out=[]; seen=False
for l in prod.splitlines(keepends=True):
    if l.startswith('IMAGE='): l='IMAGE=dsv41-4x-spark:canary-mig\n'
    elif l.startswith('BUILD_DOCKERFILE='):
        l=('BUILD_DOCKERFILE=Dockerfile.canary-mig\n'
           'BUILD_ARGS="--build-arg PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple '
           '--build-arg RUSTUP_DIST_SERVER= --build-arg CRATES_MIRROR="\n')
    elif l.startswith('EP_SIZE='): l='EP_SIZE=1\n'
    elif any(l.startswith(f) for f in OFF): continue
    elif l.startswith('EXTRA_CONTAINER_ENV="') and not seen:
        seen=True
        body=l[len('EXTRA_CONTAINER_ENV="'):].rstrip('\n')[:-1]
        toks=[t for t in body.split() if not any(t.startswith(f) for f in OFF)]
        keep=[
            # Core (proven on mig)
            'DSV41_PREFILL_EMPTY_CACHE_EVERY=4',
            'DSV41_ENGRAM_PREFETCH=1', 'DSV41_SHARED_PAD_K=1',
            'DSV41_FAST_LOAD=1', 'DSV41_FAST_LOAD_TP_SLICE=on',
            'DSV41_FAST_LOAD_INFLIGHT_GB=6', 'DSV41_FAST_LOAD_N_EXPERTS=384',
            'DSV41_FAST_LOAD_SLAB_CAP_GB=12',
            # MoE (b12x_next with needle fix)
            'DSV41_MOE_B12X_NEXT=1',
            'DSV41_MOE_B12X_NEXT_GRAPH_BS=1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32',
            'DSV41_MOE_B12X_NEXT_TUNE_ROWS=6:5+6',
            # CONFIG ARM ADDITIONS (this plan's changes):
            'DSV41_VERIFY_CAP=conf:0.1', 'DSV41_BLOCK_VERIFY=1',
            'DSV41_DRAFT_TAU=0.7', 'DSV41_WO_A_W8=1',
            'DSV41_AUTOTUNE_KEEP=1', 'DSV41_FOLDED_FENCE=1',
        ]
        l='EXTRA_CONTAINER_ENV="'+' '.join(keep+toks)+'"\n'
    elif l.startswith('EXTRA_SGLANG_ARGS='):
        # Add prefill CUDA graphs flag
        if '--enable-prefill-cuda-graph' not in l:
            l = l.rstrip('\n').rstrip('"') + ' --enable-prefill-cuda-graph"\n'
    out.append(l)
open('/tmp/env.mig','w').write(''.join(out))
```
