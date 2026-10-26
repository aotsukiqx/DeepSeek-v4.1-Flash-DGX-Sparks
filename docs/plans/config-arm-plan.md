# Plan: Config-Arm Optimization (Superseded → See Status)

> Status 2026-10-26 08:00: this plan's original three items were revised after offline
> verification. Current authoritative status: `docs/PROJECT_STATE.md` §8.

## Verification Outcome (2026-10-26)

| Original item | Verdict | Why |
|---|---|---|
| Enable prefill CUDA graphs | **NOT AVAILABLE** | DSV4 auto-disable rule: `PREFILL_CP_BCG_SUPPORTED_COMBINATIONS` = {(ZIGZAG,TRTLLM_MHA),(INTERLEAVE,DSA)}; `dsv4` backend not whitelisted. Flag vocabulary also changed: `--cuda-graph-backend-{decode,prefill}={full,breakable,disabled}`. |
| Enable verify_cap + block_verify + draft family | **Symbol-verified OK, deferred** | All six adapters (verify_cap/block_verify/draft_tau/wo_a_w8/autotune_keep/folded_result_fence) have their required symbols on 7d996300. BUT none ever ran in production (all default OFF) — new experiment, not recovery. Second window. |
| (new) `--enable-prefill-cp --cp-strategy interleave` | **PRIMARY BET, launched 07:52** | Native context-parallel prefill works for DSV4 (`is_dsa_enable_prefill_cp()` True for V4). Upstream equivalent of prefill_sp (+18% hist.). env.mig v2 = v1 + CP flags ONLY; backup `/tmp/env.mig.v1-nocp`. |

## Symbol Verification Record (2026-10-26, in-image)

```
dspark_accept.py:        AcceptSampling(32) _accept_sampling_core(94) SoftmaxTemp(278)  ✅
dspark_draft_sampler.py: DsparkDraftSampler.stage_sampling_params                        ✅
dsv4_wo_a.py:            wo_a_bf16_small_batch(_mxfp8) _wo_a_reduce _quantize_partial    ✅
flashinfer_autotune.py:  _autotune_cache_digest flashinfer_autotune_context              ✅
dspark_verify.py:        TargetVerifyExecutor.accept_and_finalize(165) AcceptOuts(547)   ✅
                         AcceptOuts fields == FIELDS exactly (cap_trim/commit absorbed)  ✅
deepseek_v4_dspark.py:   DeepseekV4ForCausalLMDSpark + _apply_wo_a_bf16_matmul           ✅
```

## Decision tree after CP window

- cold ≥ 4400: CP replaces prefill_sp → full A/B + qeval
- cold 3953-4400: keep CP; layer hc_fused re-port next
- cold < 3953 or boot fail: restore v1 env (`cp /tmp/env.mig.v1-nocp /tmp/env.mig`); prefill_sp re-port required

## Appendix: env.mig generation script

Still valid for regenerating the v1 base (adapters OFF, EP1, mig image); add
`--enable-prefill-cp --cp-strategy interleave` to EXTRA_SGLANG_ARGS for v2:

```python
# Run on Mac, then scp to spark-eb1a:/tmp/env.mig
import subprocess
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
            'DSV41_PREFILL_EMPTY_CACHE_EVERY=4',
            'DSV41_ENGRAM_PREFETCH=1', 'DSV41_SHARED_PAD_K=1',
            'DSV41_FAST_LOAD=1', 'DSV41_FAST_LOAD_TP_SLICE=on',
            'DSV41_FAST_LOAD_INFLIGHT_GB=6', 'DSV41_FAST_LOAD_N_EXPERTS=384',
            'DSV41_FAST_LOAD_SLAB_CAP_GB=12',
            'DSV41_MOE_B12X_NEXT=1',
            'DSV41_MOE_B12X_NEXT_GRAPH_BS=1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32',
            'DSV41_MOE_B12X_NEXT_TUNE_ROWS=6:5+6',
        ]
        l='EXTRA_CONTAINER_ENV="'+' '.join(keep+toks)+'"\n'
    elif l.startswith('EXTRA_SGLANG_ARGS='):
        # v2: append CP flags
        if '--enable-prefill-cp' not in l:
            l = l.rstrip('\n').rstrip('"') + ' --enable-prefill-cp --cp-strategy interleave"\n'
    out.append(l)
open('/tmp/env.mig','w').write(''.join(out))
```
