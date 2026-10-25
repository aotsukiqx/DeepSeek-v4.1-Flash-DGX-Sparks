# PROJECT STATE — DeepSeek-V4.1-Flash DGX Spark TP4 Fleet

> **Single source of truth.** This document must survive context compaction and enable
> any future session to resume exactly where we left off. Last updated: 2026-10-26 07:00.

## 1. Fleet & Repository

| Item | Value |
|---|---|
| Fleet | head `spark-eb1a` (192.168.2.23), workers `spark-a89f/141e/2fe7` (192.168.2.20-22) |
| Hardware | 4× NVIDIA DGX Spark (GB10, SM120, 128GB unified memory, 200G RoCE switched fabric) |
| Model | DeepSeek-V4.1-Flash-1017 (48 shards, config.json byte-identical to V4.1-Flash) |
| Cluster repo | `spark-eb1a:~/nvmodels/DeepSeek-v4.1-Flash-DGX-Sparks` (origin: MiaAI-Lab main) |
| Mac working tree | `/Volumes/LLM2/Projects/DeepSeek-v4.1-Flash-DGX-Sparks` (origin: aotsukiqx fork) |
| Deploy method | Mac commits → `git push origin main` → `git bundle create /tmp/x.bundle main` → `scp` to head → `git fetch + merge --ff-only` → **ALWAYS verify with `git log --oneline -1` + grep the change** |
| Current repo HEAD | `6dd33ac` (Mac and cluster in sync) |

## 2. Two Code Lines (running simultaneously)

### Production line (LIVE, stable, verified)
- **Image**: `dsv41-4x-spark:canary-roce` (bundle#3, code `1c36c98`)
- **sglang pin**: `620c76f2` on base `nightly-dev-cu13-20261011@4793eb40`
- **Config**: EP1, TP4, b12x_next ON, prefill_sp ON, hc_fused ON, full adapter set
- **KV pool**: MAX_TOTAL_TOKENS=6700000, MRR=32, chunk=4096, MEM_FRACTION_STATIC=0.80
- **H0a adopted**: `DSV41_PREFILL_EMPTY_CACHE_EVERY=4` (empty_cache every 4th chunk)
- **RoCE**: ON (SGLANG_ROCE_ALLREDUCE=1, +17% conc6 asset, R2-proven)
- **Rollback points**: tags `pre-bundle3`, `pre-bundle2`, `pre-serving`, `pre-620c` + `.env.tp4.pre-*` backups on all 4 nodes

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
| decode net tok/s | 44.0 | 37.8 | **-14%** | prefill_sp OFF (~-8%), hc_fused OFF (~-2%), verify/draft family OFF (~-4%) |
| conc6 aggregate | 3884 | 4021 | **+3.5%** | upstream scheduler improvement |
| smoke | 4/4 PASS | 4/4 PASS | ✓ | all serving-layer fixes work on both lines |
| qeval | 53/55 BROKE=0 | (not compared) | — | needs paired run |

**Decision table verdict**: -3.7% → "migration + re-port" path (prefill_sp re-port should push above baseline).

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
| 7 | slab pile-up (39.7GB anon) | `DSV41_FAST_LOAD_SLAB_CAP_GB=12` backpressure | `c6b314d` |
| 8 | b12x_next needle drift | `_weight_layout_method(self)` substrings (multi-line-safe) | `bd6b744` |
| 9 | **`deepseek-recipe` missing** | `pip install deepseek-recipe` — this was misdiagnosed as "OOM" for 7 boots | `edaf31c` |

Also: engram `staged_by_scheduler` + `_sort_host_reads` attrs → `0094702`

**Key lesson**: the "-9 SIGKILL / OOM" errors in W9-W11 were ALL the missing `deepseek-recipe` package. The detokenizer exit was cleanup, not the OS killer. 7 boot attempts × ~25 min = ~3 hours spent on a pip install.

## 5. Key Files & Their Roles

### Adapters (`adapter/`)
| File | Function | Mig arm status |
|---|---|---|
| `moe_b12x_next.py` | SM120 MXFP4 MoE on b12x (raced plans) | ✅ ON (needle fix) |
| `prefill_sp.py` | Prefill sequence parallel (+18% cold) | ❌ OFF (structural re-port needed) |
| `hc_fused.py` | HC mix stats fusion (+4.3% cold) | ❌ OFF (module moved) |
| `fast_load.py` | Eager weight reads with pacing | ✅ ON (slab cap + TP_SLICE=on) |
| `engram_backend.py` | NVMe-resident Engram tables | ✅ ON (3 attr fixes) |
| `engram_prefetch.py` | Engram row prefetch on side stream | ✅ ON |
| `prefill_empty_cache.py` | Host memory reclaim during long prefills | ✅ ON (EVERY=4) |
| `loop_abort.py` | Repetition loop detection & abort | ✅ ON |
| `history_repair.py` | Broken tool-call args → repaired history | ✅ ON |
| `placeholder_sanitize.py` | Image placeholder token → `[image]` on recipe line | ✅ ON (new for mig) |
| `encoding_compat.py` | Thinking alias, max_tokens cap, anthropic protocol | ✅ ON |
| `block_verify.py` | Block verification for sampled rows | ❌ OFF (drift) |
| `verify_cap.py` | Per-request verify length via acceptance cutoff | ❌ OFF (drift) |
| `folded_result_fence.py` | Folded results cloned off persistent buffers | ❌ OFF (drift) |
| `autotune_keep.py` | FlashInfer autotune cache across boots | ❌ OFF (drift) |
| `draft_tau.py` | Draft proposal temperature | ❌ OFF |
| `wo_a_w8.py` | FP8 twin for verify/draft wo_a weights | ❌ OFF |
| `draft_head_fp8.py` / `_tp4.py` | Draft LM head from FP8 copy | ❌ OFF |
| `sitecustomize.py` | Import hook that installs all adapters | ✅ active |

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

### Boot mig arm (requires `/tmp/env.mig`)
```bash
# /tmp/env.mig is generated from production .env.tp4 with:
#   IMAGE=canary-mig, BUILD_DOCKERFILE=Dockerfile.canary-mig, EP_SIZE=1
#   b12x_next ON, fast_load ON+TP_SLICE=on+SLAB_CAP, prefill_sp/hc_fused/verify-family OFF
# Generate: see docs/plans/config-arm-plan.md §appendix
ssh spark-eb1a 'touch /tmp/dsv41-maintenance; \
  cd ~/nvmodels/DeepSeek-v4.1-Flash-DGX-Sparks && \
  cp .env.tp4 /tmp/env.prod.live && cp /tmp/env.mig .env.tp4 && \
  ./start-tp4.sh build && \
  ./start-tp4.sh stop && \
  nohup python3 scripts/verify/boot_hygiene.py reap --minutes 30 & \
  nohup ./start-tp4.sh serve &'
# ~18 min to health (48-shard load + init)
# ALWAYS: touch /tmp/dsv41-maintenance during window; rm it after
# ALWAYS: cp /tmp/env.prod.live back + serve to rollback
```

### Measurement probes (traffic-gated)
```bash
# Wait for 45s POST gap, then:
python3 /tmp/probe600k.py cold <seed>      # single cold prefill
python3 /tmp/probe600k.py conc 6 <seed>    # 6-concurrent aggregate
python3 /tmp/decode600k.py <seed>           # single decode net-of-TTFT
# ALWAYS: discard first cold run after boot; post-check probe window for foreign batches
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

1. **Mig boot = ~25 min production outage.** Only attempt with offline-verified fixes. Always check pip deps in Dockerfile first (deepseek-recipe cost 7 boots).
2. **GB10 UMA**: NEVER prewarm page cache (OOM). Swap activation = thrash (59-82s/shard). Recovery = stop→drain→serve.
3. **`/tmp` is untrusted** (systemd-tmpfiles ages it out). All probes/scripts live in `scripts/verify/`; deploy via `cp` to `/tmp` before use.
4. **Bundle relays**: always verify with `git log --oneline -1` + grep after merge. Several "successful" relays ran in temp shells and never landed.
5. **`pkill` patterns must not self-match** (use `[.]` in regex). Remote `.env` edits must be line-based (comment lines contain decoy matches).
6. **Maintenance marker**: `touch /tmp/dsv41-maintenance` during every experiment window; `rm` it after. The watchdog respects it.
7. **Desktop services**: masked via `systemctl --user mask` on all 4 nodes (reboot-surviving). If they return: `systemctl --user stop pipewire pipewire-pulse wireplumber snap.snapd-desktop-integration...`
8. **Node reboots happen** (2026-10-25 12:07 cluster-wide event). Watchdog auto-recovers. Check `last reboot` for timing.

## 8. Next Steps (Prioritized)

### Immediate (config-level, zero code changes)
See `docs/plans/config-arm-plan.md`
- Enable prefill CUDA graphs on mig arm
- Enable verify_cap + block_verify + draft family on mig arm
- Expected: +5-12% recovery, one boot window

### Short-term (half day)
See `docs/plans/hc_fused-report-plan.md`
- hc_fused re-port to `kernels/ops/gemm/hc_mix.py`
- Expected: +4.3% cold prefill

### Main effort (2-4 sessions)
See `docs/plans/prefill_sp-report-plan.md`
- prefill_sp structural re-port to `op_*` API family
- Expected: +18% cold prefill + ~8% decode
- This is the critical path to mig arm exceeding production baseline

### After all three
- Full A/B with qeval paired + 600k + conc6 → adopt if net positive
- If adopted: update `.env.tp4.example`, `docs/tp4.md`, tag rollback points

## 9. Historical Context (for reference)

- `docs/pin-migration-plan-20261024.md` — Full migration evaluation (dry-run, hash drift, decision table)
- `docs/serving-robustness-20261018.md` — Serving-layer bundles #1-#3 (all merged into production)
- `docs/incident-20261025-reboot-desktop.md` — Cluster reboot + desktop noise + W7-W11 boot campaign + final measurement
- `docs/600k-migration-20261011.md` — Original 620c76f2 migration (prefill_sp, hc_fused ports)
- `docs/tp4.md` — Production "what runs" reference (may be slightly stale)

## 10. EP2 vs EP1 Theory (2026-10-25 analysis)

EP2 is a **workaround** (stock CUTLASS rejects 576 at EP1), not a perf choice:
- **Decode**: neutral-to-negative (3 sync points vs 1 per MoE layer; load imbalance at small batch; more kernel launches; RoCEnante needs re-tuning for TP2 groups)
- **Prefill**: small-positive (wider N=1152 GEMMs; all-to-all < all-reduce bytes)
- **Memory**: zero net change
- **Conclusion**: EP1 with b12x_next is the correct path (b12x handles 576 natively)
