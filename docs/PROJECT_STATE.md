# PROJECT STATE — DeepSeek-V4.1-Flash DGX Spark TP4 Fleet

> **Single source of truth.** This document must survive context compaction and enable
> any future session to resume exactly where we left off. Last updated: 2026-10-26 08:00.

## 1. Fleet & Repository

| Item | Value |
|---|---|
| Fleet | head `spark-eb1a` (192.168.2.23), workers `spark-a89f/141e/2fe7` (192.168.2.20-22) |
| Hardware | 4× NVIDIA DGX Spark (GB10, SM120, 128GB unified memory, 200G RoCE switched fabric) |
| Model | DeepSeek-V4.1-Flash-1017 (48 shards, config.json byte-identical to V4.1-Flash) |
| Cluster repo | `spark-eb1a:~/nvmodels/DeepSeek-v4.1-Flash-DGX-Sparks` (origin: MiaAI-Lab main) |
| Mac working tree | `/Volumes/LLM2/Projects/DeepSeek-v4.1-Flash-DGX-Sparks` (origin: aotsukiqx fork) |
| Deploy method | Mac commits → `git push origin main` → `git bundle create /tmp/x.bundle main` → `scp` to head → `git fetch + merge --ff-only` → **ALWAYS verify with `git log --oneline -1` + grep the change** |

## 2. Two Code Lines (running simultaneously)

### Production line (LIVE, stable, verified)
- **Image**: `dsv41-4x-spark:canary-roce` (bundle#3, code `1c36c98`)
- **sglang pin**: `620c76f2` on base `nightly-dev-cu13-20261011@4793eb40`
- **Config**: EP1, TP4, b12x_next ON, prefill_sp ON, hc_fused ON
- **KV pool**: MAX_TOTAL_TOKENS=6700000, MRR=32, chunk=4096, MEM_FRACTION_STATIC=0.80
- **H0a adopted**: `DSV41_PREFILL_EMPTY_CACHE_EVERY=4` (empty_cache every 4th chunk)
- **RoCE**: ON (SGLANG_ROCE_ALLREDUCE=1, +17% conc6 asset, R2-proven)
- **Rollback points**: tags `pre-bundle3`, `pre-bundle2`, `pre-serving`, `pre-660c` + `.env.tp4.pre-*` backups on all 4 nodes

### Migration arm (measured once, NOT adopted)
- **Image**: `dsv41-4x-spark:canary-mig` (Dockerfile.canary-mig)
- **sglang pin**: `7d996300` (2026-10-24, settled after MegaGate churn)
- **Config**: EP1, TP4, b12x_next ON (1-line needle fix), prefill_sp OFF, hc_fused OFF
- **Extra wheels**: flashinfer 0.7.2, deep-gemm 0.2.3.post2, kernel 0.4.10, flash-mla 1.0.0, **deepseek-recipe**
- **Boot requires**: Dockerfile.canary-mig + `/tmp/env.mig` (EP1 config with adapter OFF set)

## 3. Latest Measurements (2026-10-26)

| Metric | Production (baseline) | Migration arm | Delta | Root cause |
|---|---:|---:|---:|---|
| cold 600k tok/s | 4101-4121 | 3953.4 | **-3.7%** | prefill_sp OFF (-18% hist.) partially compensated by upstream wave |
| decode net tok/s | 44.0 | 37.8 | **-14%** | prefill_sp OFF (~-8%), hc_fused OFF (~-2%), ~-4% unattributed (see §8a) |
| conc6 aggregate | 3884 | 4021 | **+3.5%** | upstream scheduler improvement |
| smoke | 4/4 PASS | 4/4 PASS | ✓ | all serving-layer fixes work on both lines |
| qeval | 53/55 BROKE=0 | (not compared) | — | needs paired run |

**Decision table verdict**: -3.7% → "migration + re-port" path. But see §8: native
prefill-CP may replace the re-port.

## 4. Migration Boot Blockers — ALL RESOLVED (9 total)

The mig arm required 9 fixes across 8 boot attempts (W7-W11 + EP1a-c + final):

| # | Blocker | Fix | Commit |
|---|---|---|---|
| 1 | import chain failures | sitecustomize hook chain clean (32/32) | multiple |
| 2 | `flash_mla` missing | `sgl-flash-mla==1.0.0` in Dockerfile pip layer | `21780e6` |
| 3 | EP2 shape assert (576 not ×128) | **EP1 path** (b12x_next handles 576 natively) | — |
| 4 | engram `column_bounds` kwarg | kwargs-tolerant init (bitwise-equivalent split) | `2c0d9a3` |
| 5 | engram `peer` attr | `self.peer = None` in init | `7516e87` |
| 6 | fast_load EP2 slicing 60x slow | `DSV41_FAST_LOAD_TP_SLICE=on` (env-only knob) | — |
| 7 | slab pile-up (39.7GB anon) | `DSV41_FAST_LOAD_SLAB_CAP_GB=12` backpressure in `_new_slab()` | `c6b314d` |
| 8 | b12x_next needle drift | `_weight_layout_method(self)` substrings (multi-line-safe) | `bd6b744` |
| 9 | **`deepseek-recipe` missing** | `pip install deepseek-recipe` — misdiagnosed as "OOM" for 7 boots | `edaf31c` |

Also: engram `staged_by_scheduler` + `_sort_host_reads` attrs → `0094702`.

**Key lesson**: the "-9 SIGKILL / OOM" errors in W9-W11 were ALL the missing `deepseek-recipe`
package (detokenizer exit was cleanup, not the OS killer). 7 boot attempts × ~25 min ≈ 3 hours
spent on a pip install. Check pip deps in the Dockerfile FIRST.

## 5. Key Files & Their Roles

### Adapters (`adapter/`)
| File | Function | Mig arm status |
|---|---|---|
| `moe_b12x_next.py` | SM120 MXFP4 MoE on b12x (raced plans) | ✅ ON (needle fix) |
| `prefill_sp.py` | Prefill sequence parallel (+18% cold) | ❌ OFF — native CP replaces it (see §8) |
| `hc_fused.py` | HC mix stats fusion (+4.3% cold) | ❌ OFF (module moved) |
| `fast_load.py` | Eager weight reads with pacing | ✅ ON (slab cap + TP_SLICE=on) |
| `engram_backend.py` | NVMe-resident Engram tables | ✅ ON (3 attr fixes) |
| `engram_prefetch.py` | Engram row prefetch on side stream | ✅ ON |
| `prefill_empty_cache.py` | Host memory reclaim during long prefills | ✅ ON (EVERY=4) |
| `loop_abort.py` | Repetition loop detection & abort | ✅ ON |
| `history_repair.py` | Broken tool-call args → repaired history | ✅ ON |
| `placeholder_sanitize.py` | Image placeholder token → `[image]` on recipe line | ✅ ON |
| `encoding_compat.py` | Thinking alias, max_tokens cap, anthropic protocol | ✅ ON |
| `block_verify.py` | Block verification for sampled rows | ❌ mig OFF — prod ON (=1), symbol-verified on 7d996 |
| `verify_cap.py` | Per-request verify length via acceptance cutoff | ❌ mig OFF — prod ON (=conf:0.1), symbol-verified |
| `folded_result_fence.py` | Folded results cloned off persistent buffers | ❌ mig OFF — prod ON (=1), FIELDS match upstream |
| `autotune_keep.py` | FlashInfer autotune cache across boots | ❌ mig OFF — prod ON (=1), symbol-verified |
| `draft_tau.py` | Draft proposal temperature | ❌ mig OFF — prod ON (=0.7), symbol-verified |
| `wo_a_w8.py` | FP8 twin for verify/draft wo_a weights | ❌ mig OFF — prod ON (=1+DROP+MID), symbol-verified |
| `draft_head_fp8.py` / `_tp4.py` | Draft LM head from FP8 copy | ❌ OFF |
| `sitecustomize.py` | Import hook that installs all adapters | ✅ active |

**IMPORTANT (2026-10-26 correction)**: the six "verify/draft family" adapters
(verify_cap/block_verify/draft_tau/wo_a_w8/autotune_keep/folded_result_fence) were
**never enabled in production either** — all default OFF, absent from .env.tp4. Treat
them as new experiments, not as "recovery of lost production features".

### Dockerfiles
| File | Purpose |
|---|---|
| `Dockerfile` | Base (3-Spark line, dev-dsv41 image) |
| `Dockerfile.canary` | 4-Spark canary (non-RoCE) |
| `Dockerfile.canary-roce` | **Production** (sglang-canary row, RoCEnante overlay) |
| `Dockerfile.canary-mig` | **Migration arm** (sglang-mig row = 7d996300, 5 pip wheels, no old patches) |

### Manifest (`runtime/sources.manifest`)
- `sglang-canary` row: `620c76f2` (production)
- `sglang-mig` row: `7d996300` (migration arm, sha256 `5d648297...`)

### Serving patches (`runtime/`)
| File | Target | Status on mig |
|---|---|---|
| `patch_encoding_dsv41.py` | placeholder sanitize | **absorbed by placeholder_sanitize adapter** |
| `patch_encoding_dsv41_44751.py` | tool-result image order | **architecture-replaced** (recipe handles it) |
| `patch_serving_chat_43006.py` | tool_choice=none | **absorbed upstream** |
| `patch_response_parser_43941.py` | integer tool args "3.0" | **absorbed upstream** |

### Verification tools (`scripts/verify/`)
| Tool | Function |
|---|---|
| `probe600k.py` | Cold 600k prefill + conc6 aggregate probes |
| `decode600k.py` | Single-stream decode @600k, net-of-TTFT |
| `decode_matrix.py` | Decode throughput vs concurrency tier |
| `svc_smoke.py` | 4-probe serving smoke (thinking/placeholder/history/vision) |
| `mig_window.sh` | Full mig boot window (build→serve→gates→probes→auto-rollback) |
| `serve_watchdog.sh` | Auto-recover after node reboots |
| `boot_hygiene.py` | Rustup reaper + orphan checker |
| `qeval.py` + `qeval_tasks.py` | 55-task paired quality evaluation (McNemar) |

## 6. Operational Procedures

### Boot production line
```bash
ssh spark-eb1a 'cd ~/nvmodels/DeepSeek-v4.1-Flash-DGX-Sparks && ./start-tp4.sh stop; \
  nohup python3 scripts/verify/boot_hygiene.py reap --minutes 16 & \
  nohup ./start-tp4.sh serve &'
# Wait ~2.5 min for health
```

### Boot mig arm via measurement window (preferred — auto-rollback)
```bash
# Prereqs: /tmp/env.mig staged (see docs/plans/config-arm-plan.md §appendix),
# /tmp/mig_window.sh present, /tmp/dsv41-maintenance touched during window.
ssh spark-eb1a 'L=$(wc -l < /tmp/mig-window.log); echo "LOG_BASELINE=$L" > /tmp/window-marker.txt; \
  touch /tmp/dsv41-maintenance; \
  setsid nohup bash /tmp/mig_window.sh > /tmp/mig-window-stdout.log 2>&1 < /dev/null &'
# Monitor: tail -n +$((LOG_BASELINE+1)) /tmp/mig-window.log | grep -E '^\['
# Window: PHASE0 build (fast if cached) → PHASE1 serve (~18 min) → PHASE2 (~26 min) → rollback
```

### Measurement probes (traffic-gated)
```bash
# Wait for 45s POST gap, then:
python3 /tmp/probe600k.py cold <seed>      # single cold prefill (discard first after boot)
python3 /tmp/probe600k.py conc 6 <seed>    # 6-concurrent aggregate
python3 /tmp/decode600k.py <seed>           # single decode net-of-TTFT
```

### Post-crash recovery (proven 4+ times)
```bash
./start-tp4.sh stop
for h in spark-a89f spark-141e spark-2fe7; do
  ssh -n $h 'docker ps -aq --filter name=dsv41 | xargs -r docker rm -f; pkill -9 -f launch_server'
done
sleep 30  # drain swap
nohup python3 scripts/verify/boot_hygiene.py reap --minutes 16 &
nohup ./start-tp4.sh serve &
```

## 7. Hard-Won Operational Rules

1. **Mig boot = ~25 min production outage.** Only attempt with offline-verified fixes. Check pip deps in Dockerfile first (deepseek-recipe cost 7 boots).
2. **GB10 UMA**: NEVER prewarm page cache (OOM). Swap activation = thrash (59-82s/shard). Recovery = stop→drain→serve.
3. **`/tmp` is untrusted** (systemd-tmpfiles ages it out — `/tmp/cand7` died this way). All probes/scripts live in `scripts/verify/`; deploy via `cp` to `/tmp` before use.
4. **Bundle relays**: always verify with `git log --oneline -1` + grep after merge. Several "successful" relays ran in temp shells and never landed.
5. **`pkill` patterns must not self-match** (use `[.]` in regex). Remote `.env` edits must be line-based (comment lines contain decoy matches).
6. **Maintenance marker**: `touch /tmp/dsv41-maintenance` during every experiment window; `rm` it after. The watchdog respects it.
7. **Desktop services**: masked via `systemctl --user mask` on all 4 nodes (reboot-surviving). If they return after reboot: `systemctl --user stop pipewire pipewire-pulse wireplumber snap.snapd-desktop-integration...`
8. **Node reboots happen** (2026-10-25 12:07 cluster-wide event). Watchdog auto-recovers. Check `last reboot` for timing.
9. **Candidate source access** (2026-10-26): `/tmp/cand7` aged out; the only candidate copy now lives in the mig image: `ssh spark-eb1a "docker run --rm --entrypoint /bin/bash dsv41-4x-spark:canary-mig -c 'grep ... /sgl-workspace/sglang/python/sglang/...'"`.

## 8. Next Steps (Prioritized — updated 2026-10-26 08:00)

### Native prefill-CP: CLOSED (rejected by model, 2026-10-26 08:00)
`--enable-prefill-cp --cp-strategy interleave` → boot exit 1 at 70s:
```
deepseek_v4.py:4641 ValueError: V4.1 vision supports TP/EP/DP without CP or PP;
MoE A2A requires CUDA MegaMoE
```
CP engaged at topology level (log prefix became ATTN_CP0), then the V4.1 vision model
constructor rejected it. Adapters armed fine before the raise — failure is purely the
CP flag. env.mig restored to v1 (backup `/tmp/env.mig.v1-nocp`). First launch also hit
a transient worker-DNS SERVFAIL (spark-a89f systemd-resolved on registry-1.docker.io;
retry succeeded).

Prefill CUDA graphs likewise unavailable: `PREFILL_CP_BCG_SUPPORTED_COMBINATIONS` =
{(ZIGZAG,TRTLLM_MHA),(INTERLEAVE,DSA)}; our `dsv4` backend is not whitelisted
(c4-indexer scratch OOMs the capture pool). Flag vocabulary changed on 7d996300:
`--cuda-graph-backend-{decode,prefill}={full,breakable,disabled}`.

### CRITICAL PATH: prefill_sp re-port — Phase A+B DONE (commits d34dfd4, 6592aee)
Phase A: classification (7 match / 8 port / 4 redirect / 2 gone-on-CUDA); CUDA runtime
path identified — **Model._forward_layers_hc_pre_from_prev is STILL the CUDA driving
loop** (V4.1 config class defaults hc_pre_from_prev_sublayer=True; config.json alone
misleads); per-layer seam = CudaDecoderLayer.forward (37 lines) + op_*.
Phase B: dual-pin hash tables (28-anchor 7d996300 set + 620c legacy, probe-selected —
production rebuild safe); CUDA-safe install() (comm/shard refuse with Phase C pointer);
verified in-image BOTH pins (28/28 MATCH; 620c probe PASS; CPU tests ok).
**Resume at Phase C** (rewrite _sp_forward_layers to new per-layer signature; SP wraps
at op_gather_a/b + MoeTransport + op_combine/op_mhc; then validation window).
⚠️ Alias gotcha: `DeepseekV4DecoderLayer` resolves to Platform class when CUDA is
hidden — always run checks CUDA-visible.

### Then
- hc_fused re-port: `docs/plans/hc_fused-report-plan.md` (+4.3% cold, half day)
- **Six-adapter mig window COMPLETE 2026-10-26 11:37 (4 crash-fix iterations)**:
  cold 3994.3 (+1.0% vs v1 3953), conc6 4033.0 (+0.3%), decode flat (36.8 avg),
  smoke 4/4, qeval 71/75. **Flat because verify_cap is INERT** on 7d996300 — its
  planner hook point (`compute_confidence_tensor`) was absorbed into the budget
  machinery; without conf capture the verify trimming (the decode contributor)
  never engages. Unlock = small verify_cap re-port onto the new planner's
  confidence path (confidence_budget_prepare & co).
  Crash chain root-caused & fixed (commits 6b5abd6..793125f): (1) wo_a_w8 kernel
  path moved (gemm/dsv4_wo_a) — the old-path fallback einsum-bridge detonated via
  (2) sitecustomize hook exceptions evicting engine modules (importlib removes
  failed-import modules while registrations persist → duplicate-invariant);
  sitecustomize now NEVER propagates hook exceptions; (3) verify_cap planner hook
  hasattr-guarded (loud INERT note); (4) block_verify execute needs **kw passthrough
  (7d996300 passes block_verification kwarg). mig_window.sh rollback now captures
  worker logs FIRST (before start-tp4.sh stop deletes containers).
- Full A/B + qeval paired → adopt if net positive; if adopted update `.env.tp4.example`, `docs/tp4.md`, tag rollback points

## 8a. Decode-Gap Attribution (RE-VERIFIED 2026-10-26, full env evidence)
⚠️ Two wrong turns this session, both from `^`-anchored greps missing the quoted
EXTRA_CONTAINER_ENV line (the decoy-line trap this repo's skill warns about).
**Final evidence**: production .env.tp4 (== /tmp/env.prod.live, diff empty) runs the
FULL adapter stack — VERIFY_CAP=conf:0.1, BLOCK_VERIFY=1, DRAFT_TAU=0.7,
WO_A_W8=1(+DROP=1+MID=1), AUTOTUNE_KEEP=1, FOLDED_FENCE=1, DRAFT_HEAD_FP8=1(tp4),
DRAFT_MAIN_PROJ_SPLIT=1, L2_PREFETCH=1, PREFILL_SP=1(shard)+FP8=1, HC_FUSED=1.
So mig decode 37.8 vs prod 44.0 (-14%) = prefill_sp OFF (~-8%) + hc_fused OFF (~-2%)
**+ six-adapter family OFF (~-4%)** — the original attribution stands; the six
adapters ARE production-proven. To audit env vars in future: `grep -o
'DSV41_[A-Z_0-9]*=[^ "]*' .env.tp4 | sort -u` (never `^`-anchor).

## 9. Historical Context (for reference)

- `docs/pin-migration-plan-20261024.md` — Full migration evaluation (dry-run, hash drift, decision table)
- `docs/serving-robustness-20261018.md` — Serving-layer bundles #1-#3 (all merged into production)
- `docs/incident-20261025-reboot-desktop.md` — Cluster reboot + desktop noise + W7-W11 boot campaign + final measurement
- `docs/600k-migration-20261011.md` — Original 620c76f2 migration (prefill_sp, hc_fused ports)
- `docs/tp4.md` — Production "what runs" reference (may be slightly stale)
- `docs/plans/` — config-arm / hc_fused / prefill_sp implementation plans

## 10. EP2 vs EP1 Theory (2026-10-25 analysis)

EP2 is a **workaround** (stock CUTLASS rejects 576 at EP1), not a perf choice:
- **Decode**: neutral-to-negative (3 sync points vs 1 per MoE layer; load imbalance at small batch; more kernel launches; RoCEnante needs re-tuning for TP2 groups)
- **Prefill**: small-positive (wider N=1152 GEMMs; all-to-all < all-reduce bytes)
- **Memory**: zero net change
- **Conclusion**: EP1 with b12x_next is the correct path (b12x handles 576 natively)
