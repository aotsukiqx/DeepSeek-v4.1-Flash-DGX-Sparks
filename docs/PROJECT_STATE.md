# PROJECT STATE — DeepSeek-V4.1-Flash DGX Spark TP4 Fleet

> **Single source of truth.** This document must survive context compaction and enable
> any future session to resume exactly where we left off. Last updated: 2026-10-28 20:35.

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

### Production line (LIVE since 21:55 2026-10-27 — mig v4 + bp1 backports)
- **Image**: `dsv41-4x-spark:canary-mig-bp1` (sglang **7d996300** + bp1 source
  patches #46706/py + #46404 + #41275, Dockerfile.canary-mig patch chain)
- **Config**: unchanged from v4 (EP1, TP4, b12x_next ON, six adapters,
  DSV41_PIN7=1, prefill_sp OFF, hc_fused absorbed; env identical to pre-bp1)
- **bp1 evidence (2026-10-27 window)**: qeval paired BROKE=0 (55 tasks,
  p=1.000, median 73.7→75.3 tok/s); smoke 4/4 + trailing-toolcall-turn,
  assistant-null-content+tool_calls, numeric-effort-0.5, sampled request all
  PASS; decode600k **45.6** (gate ≥42); cold 600k seed5 **3977.8** (v4
  baseline 3978.8, -0.03%); conc6 contaminated by live traffic (4828, batching
  inversion — re-measure quiet during soak)
- **Known capacity change (#41275, expected)**: `max_total_num_tokens`
  6336256 → **6169856** (-166400 = 2600 pages): explicit cap re-validated
  through the patched auto path — the reserved-page overcommit upstream
  removes. NOT the predicted no-op; mechanism understood, accepted
- **Cutover procedure (proven)**: stop → clean workers → **drain ≥45s MANDATORY** →
  serve → 200 (env unchanged; no env copy needed for image-only flips)

### Rollback line (pre-bp1, one step back) — the ONLY staged rollback
- **Image**: `dsv41-4x-spark:canary-mig-pre-bp1` on all 4 nodes (v4 FINAL
  without bp1); git tag `pre-bp1`; env unchanged (`/tmp/env.pre-bp1` backup ==
  live .env.tp4)
- **Procedure (image-only flip)**: stop → drain ≥45s →
  `docker tag dsv41-4x-spark:canary-mig-pre-bp1 dsv41-4x-spark:canary-mig` on
  all 4 nodes → serve.
- **bp1 soak CLEARED 2026-10-28 20:20 (23h, user-authorized early)**: smoke
  4/4, decode spot 43.7 (≥42), RestartCount 0, 0 crash/OOM/traceback in 23h
  logs, no quality complaints. conc6 clean measurement NOT obtainable under
  live traffic (4789 @ ~57 prefill/3m — window-conditioned, both runs
  consistent, direction up = no regression signal; treat conc6 on this fleet
  as window-conditioned going forward). Pre-adoption rollback set RETIRED
  (canary-roce image ×4 deleted, /tmp/env.pre-adoption-620c removed; git tag
  pre-adoption-v4 kept as history; old canary-roce-pre-* tags left for disk
  history).

## 3. Latest Measurements (2026-10-26 13:20, four-way)

| Metric | Production | mig v1 | mig v3 | **mig v4 FINAL** | v4 vs prod |
|---|---:|---:|---:|---:|---:|
| cold 600k tok/s (seed3) | 4101-4121 | 3953.4 | 3994.3 | **3978.8** | -3.0% |
| decode net tok/s (avg) | 44.0 | 37.05 | 36.8 | **43.9** | **-0.2% ✓ closed** |
| conc6 aggregate | 3884 | 4021 | 4033 | **4042** | **+4.1%** |
| smoke | 4/4 | 4/4 | 4/4 | 4/4 | ✓ |
| qeval | 53/55 | — | 71/75 | 71/75 (median 76.6 tok/s) | ✓ |

**v4 config** (env.mig + DSV41_PIN7=1): six adapters + verify_cap STATIC-PATH re-port
(begin_static_step writes conf-derived cutoffs into verify_lens_buf after the engine's
full-stride reset; conf from DraftBlockProposer.propose wrap) + shadowed-elif fix
(draft_tau / folded_fence / verify_cap executor wraps NEVER installed on any pin
before — production's 44.0 was achieved without them; now active on mig, frozen on
620c behind DSV41_PIN7) + draft_tau DraftParams re-port.

**Decode gap CLOSED** (43.9 vs 44.0). Remaining: cold -3.0% — prefill_sp Phase C
(+18% hist.) is the last unlock. conc6 already beats production by +4.1%.

### Input-scattered shortcut: DEAD END (2026-10-26 14:19)
`--enable-attn-tp-input-scattered` + `--moe-dense-tp-size 4` (the eligibility
predicate requires dense FFN not fully-DP): flag resolved to 4, adapter logged
"not enabled while other conditions are not met", and the boot crashed at SWA
replay (`raw_positions - swa_replay_start`: 131 vs 128 row mismatch — interaction
with the SPARK_PREFILL_TP_SPLIT row-splitting already in our env). Upstream
territory; not retried. The remaining cold gap belongs to prefill_sp Phase C
(manual wrapper port; plan fully staged) or is accepted at adoption time.

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
| `prefill_sp.py` | Prefill sequence parallel (+18% cold) | ❌ OFF — Phase A+B done, Phase C pending (§8) |
| `hc_fused.py` | HC mix stats fusion (+4.3% cold) | ❌ OFF (module moved; re-port plan ready) |
| `fast_load.py` | Eager weight reads with pacing | ✅ ON (slab cap + TP_SLICE=on) |
| `engram_backend.py` | NVMe-resident Engram tables | ✅ ON (3 attr fixes) |
| `engram_prefetch.py` | Engram row prefetch on side stream | ✅ ON |
| `prefill_empty_cache.py` | Host memory reclaim during long prefills | ✅ ON (EVERY=4) |
| `loop_abort.py` | Repetition loop detection & abort | ✅ ON |
| `history_repair.py` | Broken tool-call args → repaired history | ✅ ON |
| `placeholder_sanitize.py` | Image placeholder token → `[image]` on recipe line | ✅ ON |
| `encoding_compat.py` | Thinking alias, max_tokens cap, anthropic protocol | ✅ ON |
| `block_verify.py` | Block verification for sampled rows | ✅ boot-green (kwargs fix 793125f) |
| `verify_cap.py` | Per-request verify length via acceptance cutoff | ⚠️ boots INERT — planner hook point absorbed upstream; re-port = decode unlock |
| `folded_result_fence.py` | Folded results cloned off persistent buffers | ✅ boot-green (FIELDS match) |
| `autotune_keep.py` | FlashInfer autotune cache across boots | ✅ boot-green |
| `draft_tau.py` | Draft proposal temperature | ✅ boot-green |
| `wo_a_w8.py` | FP8 twin for verify/draft wo_a weights | ✅ boot-green (kernel path fix 6b5abd6) |
| `draft_head_fp8.py` / `_tp4.py` | Draft LM head from FP8 copy | ❌ OFF (not symbol-verified on 7d996) |
| `sitecustomize.py` | Import hook that installs all adapters | ✅ hardened: hook exceptions NEVER propagate (module-eviction guard) |


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
| `sp_drift_check.py` | prefill_sp anchor drift check (run in mig image, CUDA-visible!) |
| `repro_six*.py` | six-adapter import-order crash repros (lilicorr trigger included) |
| `env_mig_add_six.py` | line-safe env.mig editor for the six-adapter vars |
| `patch_mig_window*.py` | mig_window.sh patchers (worker-log capture in rollback) |

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
10. **Adapter hooks must NEVER raise** (2026-10-26 lesson): an exception escaping a
    meta-path loader's exec_module makes importlib EVICT the engine module while global
    registrations (invariants) persist — the later re-import dies on "duplicate
    invariant", 13 min away from the root cause. sitecustomize is hardened (print +
    continue); keep every new hook hasattr-guarded, and give wrapped engine functions
    `**kw` passthrough (upstream adds kwargs across pins).
11. **Env audit**: never `^`-anchor greps for env vars — they live inside the quoted
    EXTRA_CONTAINER_ENV line. Use `grep -o 'DSV41_[A-Z_0-9]*=[^ "]*' .env.tp4 | sort -u`.
12. **In-image source checks must run CUDA-visible** (`--gpus all`):
    `DeepseekV4DecoderLayer` aliases to the WRONG (Platform) class when CUDA is hidden.
13. **Worker crash evidence**: capture `docker logs dsv41-worker` BEFORE
    `start-tp4.sh stop` (stop deletes worker containers). mig_window.sh does this now.
14. **Config-class vs config.json**: pin config CLASSES default flags that json omits
    (e.g. `configs/deepseek_v41.py` sets hc_pre_from_prev_sublayer=True). Never conclude
    from raw json alone.

## 8. Next Steps (Prioritized — updated 2026-10-26 12:10)

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

### prefill_sp re-port — Phase A+B DONE (commits d34dfd4, 6592aee; priority 2 above)
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

### Priority queue (REVISED 2026-10-28 20:35 — campaign closed, steady state)
1. **bp1 ADOPTED + soak CLEARED (2026-10-27 21:55 → 10-28 20:20)** (live,
   tag `canary-mig-bp1`; plan: docs/plans/backport-20261027-plan.md):
   #46706(py)+#46404+#41275. Gates: qeval BROKE=0 (55 tasks, p=1.0), smoke
   4/4 + 4 new probes, decode 45.6/43.7, cold 3978 (-0.03% vs v4),
   23h zero-crash soak. Accepted deviation: max_total_num_tokens
   6336256→6169856 (-2.6%, #41275 reserved-page correction via cap
   re-validation). **Backport campaign CLOSED** — upstream 10-24..10-27
   sweep fully adjudicated (W1 adopted, W2 no-path, rest pin-migration).
2. **W2 (#45794 BF16 indexer score width): CLOSED — no path (2026-10-28)**.
   In-image probe on canary-mig-bp1 (docker exec, CUDA-visible): capability
   (12,1), deep_gemm 0.2.3.post2 HAS `fp8_fp4_mqa_logits` →
   `_use_deep_gemm_prefill()` True; `dense_blocks.publish_prefill` hard-branches
   to `_deep_gemm_publish_prefill`/`score_tiles`. The torch
   `prefill_requests`/`_score_chunks` surface #45794 optimizes is DEAD CODE on
   this fleet (it runs only pre-SM100 / without the fp4 kernel). Do not
   backport; revisit only if a future pin/deep-gemm drops the fp4 kernel.
3. **prefill_sp Phase C: CLOSED (2026-10-28)** — the motivation (cold -3.0%
   vs old prod) is GONE: bp1 cold 3977.8 = v4 parity, no real-traffic
   cold/TTFT complaints. Re-open only with fresh evidence of prefill pain
   (plan stays warm: docs/plans/prefill_sp-report-plan.md §3).
4. **hc_fused re-port: SKIPPED — win absorbed upstream** (verified 2026-10-26):
   `deepseek_v4_mhc.py:370 _mix_stats_impl` — "The default split-K/Sinkhorn fusion
   preserves batch invariance", gated only on `x.is_cuda and use_fused_stats and
   bf16`; kernels live in `kernels/ops/layernorm/mhc.py` (sinkhorn{,_bf16x3,
   _deepgemm}). Re-port yield ≈ 0.
5. Hygiene after bp1 soak clears: retire the pre-adoption rollback set
   (keep `canary-mig-pre-bp1` one more cycle), update `docs/tp4.md` +
   `.env.tp4.example` if env semantics changed (bp1: none).

### Upstream audit 2026-10-29 (MiaAI-Lab merged; sglang 10-27 20:19→10-29 swept)
6. **MiaAI-Lab main MERGED (a1bbbf4, live on head)**: 31 upstream commits
   (10-17 ported-PR batch). Most value was already ported in-tree (annotated
   "Ported from MiaAI-Lab …"); conflicts resolved as OURS (sitecustomize =
   strict superset; loop_abort named-exit REJECTED — re-adds post-hoc
   finished_len, its 2 tests dropped with rationale). Incoming: model default
   → V4.1-Flash-1017 @1bae0e1 (matches live), start-tp4-switchless.sh + docs,
   TP8/EP1+b12x_next guard, SPS knobs, base-Dockerfile placeholder row.
   Git-tree only — no image/boot change; next build picks it up.
7. **sglang sweep (179 new commits)**: perf payload is PIN-MIGRATION-ONLY —
   **Engram decode-lookup wave** #46917/#46925/#46955 (all-to-all lookup for
   decode+graphs, Triton reorder) + #47033 (in-place row reads) + #46893
   (--engram-config); all rebuild layers/engram.py post-pin → apply fails.
   #47148 MegaGate-in-prefill applies clean but is a NO-OP here
   (SGLANG_OPT_DEEPGEMM_MEGA_GATE default False; consumer = MegaMoE; our MoE
   is b12x_next). dspark host-lengths family: #46891 clean BUT #46897/#46777
   are its required follow-ups (dsv4 backend needs host lengths) → ride pin.
   #46840 sim-accept fix (bp1 #46404 follow-up): calibration knob only → pin.
8. **bp2 bundle candidates (apply-clean + executing path, NO window opened)**:
   #47116 reasoning-token clamp after stop truncation (DSpark verified blocks
   extend past stop → usage over-report; 4 lines), #46835 page-index pad
   skip-copy (2 lines, micro), #46994 int all-reduce off custom AR
   (defensive, attn-dp=1 here). None justifies a solo outage; stage as bp2
   when a window opens anyway. #46737 breakable-graph storage release: skip
   (decode graphs run the full backend).
9. **Next pin-migration trigger UPDATED**: Engram decode wave + dspark
   host-lengths family + mem-cache/kv-cache refactor wave (#46973-#46991) +
   spec ragged-verify rework (#34031) all stack on post-pin refactors; wait
   for upstream settle (wave still landing 10-28/29), then one migration
   captures everything.

### Retired shortcuts (all evidence-backed)
- Native prefill-CP: V4.1 vision model rejects CP.
- Prefill CUDA graphs: dsv4 backend not in the BCG whitelist.
- `--enable-attn-tp-input-scattered --moe-dense-tp-size 4`: SWA replay row crash.
- hc_fused re-port: absorbed upstream (above).

### Completed this campaign (2026-10-26)
- Six-adapter arm boot-green after 4 crash fixes (wo_a_w8 kernel path, sitecustomize
  hardening, verify_cap hasattr-guard, block_verify **kw — commits 6b5abd6..793125f).
  Measured: cold 3994.3 / decode 36.8 / conc6 4033 / smoke 4/4 / qeval 71/75 — see §3.
- Native prefill-CP CLOSED (§ above); prefill CUDA graphs unavailable.
- prefill_sp Phase A+B: dual-pin hash tables verified on BOTH images; 28/28 MATCH.
- Engine understanding: boundary/transport declarative system read and documented
  (deepseek_v4_boundary.py, layer_boundary/{layout,row_transport,stage}.py extracted
  to ~/dsv41-srcdiff/new/ on the Mac).

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
