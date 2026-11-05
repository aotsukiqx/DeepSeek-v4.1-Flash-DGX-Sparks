# PROJECT STATE — DeepSeek-V4.1-Flash DGX Spark TP4 Fleet

> **Single source of truth.** This document must survive context compaction and enable
> any future session to resume exactly where we left off. Last updated: 2026-11-05 20:30.

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

### Production line (LIVE since 2026-11-05 20:05 — BP5 ON TOP OF BP4/PIN MIGRATION)
- **Image**: `dsv41-4x-spark:canary-mig` = pin 8817f63c base + bp4 chain
  (#46088/#47949/#47684, commit cb7b13e) **+ bp5 (#47728, commit 5277b193)**:
  `runtime/bp5/bp5-pr47728-dspark-bs1-swa-slot.patch` (python-only, test hunk
  stripped) = dspark bs=1 SWA ring-slot staging fix (correctness:
  stage_commit `.contiguous()` + dspark_verify assert). **#47935 was EXCLUDED
  at pre-verify**: mxfp4 tables need ckpt `engram_embed_quant=mxfp4` (ours is
  fp8) AND our adapter replaces EngramEmbedding.__init__ with its own C
  row_store → zero runtime effect here; rides next pin.
- **Gates (2026-11-05 window, ~35 min outage)**: boot 11 min healthy, gamma=5
  /verify 6 unchanged; pool full 5836800→**6260992 (+7.3%, boot jitter, up)**;
  smoke 4/4; patch presence verified in running container; qeval vs
  pinmig-adopt baseline (before-bp5 run was wasted on a 405 — qeval --url
  needs the FULL `/v1/chat/completions` path, bare host = 405 all-fail):
  **PRIMARY 53/55 BROKE=0 FIXED=0 p=1.000**, prose 5/5, json +1 fixed
  (json_escape recovered), median single-stream **74.75→76.88 tok/s (+2.8%)**.
- **c1 (ab_bench, moderate traffic, 5 reps, gen-tp order clean)**:
  **prose 49.87 / code 114.65** vs bp4 clean-window 49.55/111.32 → code
  **+3.0%**, prose parity. SPEC_TP_SYNC=init quantification folded into this
  same-line measurement (a clean-window re-run may separate it; +3.0% here is
  the bp4+SYNC+bp5 composite vs bp4-alone baseline).
- **SPS table (2026-11-05)**: `state-tp4/dspark_sps.json` INSTALLED (7 probes,
  bs 1-32, steps/s 22.70→3.79, self-check ✓) — first table ever on this fleet.
  Profiler path: `SGLANG_DSPARK_ENABLE_SPS_RECORD=1 SGLANG_RAGGED_VERIFY_MODE=static
  SGLANG_SIMULATE_ACC_LEN=1.0 --max-batch-size 32`; importer crashes AFTER all
  rounds are written when run under traffic (bs=16 33% steady) AND in a clean
  window (bs=8 0/5517 "common steps" — dp_rank-0-only records vs multi-rank
  assumption on pure-TP dp_size=1); rounds.jsonl was complete both times —
  fit directly via `fit_profile()`. boot.py still REFUSES compact mode
  (#39173, log line present), so the planner stays verify-all; flip
  `DSPARK_SPS_ALLOW_ENGRAM=1` only after #39173 lands.
- **Window incidents**: first boot CRASHED at first Engram lookup —
  `EngramEmbedding._empty` renamed to `new_rows` upstream; fixed 3 call
  sites (engram_backend ×2, engram_prefetch ×1, commit 16a8e0d), rebuilt
  (4.5 min), second boot healthy. Total outage ≈ 50 min (11:15→12:05).
  Lesson: import audits do not catch CALL-time renamed methods — next
  migration, grep adapters for `self\._[a-z_]+\(` against the new class
  surface before booting.
- **Gates**: build self-tests ✓; boot gamma=5 / chunked 4096 / 0 errors;
  smoke 4/4 + sampled ✓; qeval vs bp2 baseline: primary p=1.000, **2
  DETERMINISTIC json flips** (json_count 11≠10, json_escape 27≠28 — model
  self-counting off-by-one under changed numerics; 3/3 runs stable, primary
  magnitude 53→51/55 inside the harness noise floor; per bp1 precedent,
  judged by magnitude); median single-stream 71.3→**74.75 tok/s (+4.9%)**;
  decode600k **43.1** (≥42; bp2 44.0); cold **3970.9** (parity); conc6
  **4795** (parity).
- **Deviation accepted**: DSV4 pool full 6221312→**5704448** (-8.3%) — the
  mem-cache/memory-pool refactor wave's new capacity pricing. 600k line has
  8.5× headroom; MAX_TOTAL_TOKENS=6700000 cap now clamps via auto-fit.
- **Config**: env unchanged (/tmp/env.pre-pinmig == live). Stable across
  2026-11-02/03 per user's stability-window plan (watchdog ACTIVE).

### Rollback ladder (bp5 live)
- **One step back**: `canary-mig-pre-bp5` ×4 (image-only flip) + git tag
  `pre-bp5` + /tmp/env.pre-bp5 (= bp4 image, SPEC_TP_SYNC env unchanged).
  Two steps: `canary-mig-pre-bp4` ×4 + `pre-bp4`. Grandfather:
  `canary-mig-bp2` ×4 + `pre-pinmig`.

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
8. **bp2 bundle ADOPTED 2026-10-30 (see §2)**: #47116 reasoning clamp +
   #46835 page-index pad skip + #46994 int-AR guard + #47149 amax8 grid +
   #46024 hc_combine splits, one window. #46737 breakable-graph release:
   skipped (decode graphs run the full backend).
9. **Next pin-migration trigger UPDATED**: Engram decode wave + dspark
   host-lengths family + mem-cache/kv-cache refactor wave (#46973-#46991) +
   spec ragged-verify rework (#34031) all stack on post-pin refactors; wait
   for upstream settle (wave still landing 10-28/29), then one migration
   captures everything.

### sglang sweep 2026-10-30 (86 new commits; MiaAI-Lab: 0 new)
10. **bp2 bundle EXPANDED then ADOPTED same day (2026-10-30, user-opened
    window; see §2)**: #47149 amax8_varlen grid fix
    (`v41_indexer/sparse_table.py:519` — our prefill path; CC10.7-class win
    not resolvable on 48 SMs, cold parity 3995.9) and #46024
    hc_combine_norm_mxfp8 batch-aware splits (`deepseek_v4_mhc.py:509`,
    decode-shaped batches; decode 44.0 = in-band) joined #47116/#46835/#46994.
    In-image pre-verify BEFORE the boot proved the JIT csrc compiles + runs
    on GB10 (once.h is vendored in-tree at kernels/jit/include — #41147).
    Lesson: qeval's default URL is worker-local :8888 — engine-only workers
    score all-0; from a worker always pass --url http://192.168.2.23:8000.
11. **No-ops / skips this window**: #46676 (dp prefill coordination guard —
    attn_dp=1), #47304 (EPLB sim routing), #44457 (triton MLA backend — we run
    dsv4), #47415/#47178 (apply-fail → pin; detokenizer flat-IPC + scheduler
    stats are host-overhead wins for the NEXT PIN). #47328 deep_select 3-CTA
    tier (needs batch>48 SMs; decode never, prefill maybe — optional A/B if a
    window opens anyway). #47115 fill_padded_rows (≥2048 narrow rows; b12x
    likely bypasses stock topk fill).

### sglang sweep 2026-11-01 (115 new commits; MiaAI-Lab: 0 new; 0 reverts)
12. **bp3 candidate (STAGED, no solo window)**: #38565 deterministic top-p/top-k
    renorm at tp_size>1 (new `layers/sampling_renorm.py` + sampler import swap;
    flashinfer is_deterministic variants, self-guarded vs flashinfer version,
    author claims no measurable cost; closes the rare TP-rank-wedge failure
    mode — we have never observed one). Apply-clean. Fold into the next window
    (pin migration or any flip), gate with decode A/B.
13. **Pin-migration list grew**: #47333 (ragged-verify eager fallback when the
    tier exceeds the captured grid), #47521 (abort retracted requests whose KV
    recompute cannot rebuild), #46827 (joint_lru Full+SWA eviction — revisit
    the --swa-full-tokens-ratio 0.065 decision AFTER migration). All
    apply-fail on the post-pin refactors, as expected.
14. **Pin-readiness**: Engram wave quiet since 10-28 (settled); mem-cache/
    memory-pool buffer-info wave STILL landing (10-30/31 trio #44990/#45019/
    #45020) and spec ragged-verify rework active (#34038/#47333) → NOT
    settled. Bonus verified: ALL 8 bp1+bp2 backports are absorbed by any pin
    ≥ 2026-10-31 (0 reverts) — at migration the whole Dockerfile patch chain
    DROPS; the work moves to adapter re-derivation instead (engram_backend/
    engram_prefetch wrap the upstream-REWRITTEN layers/engram.py; prefill_sp
    hash tables; b12x_next needles; 3 patch scripts carry re-derivation
    headers). Prep OFFLINE per dsv41-sglang-pin-migration skill (route
    decision + stack replay on a candidate tarball) — zero cluster cost.
    Skipped: #47387 (default-off SM90-targeted opt-in decode indexer).

### Pin-migration prep (route decision DONE 2026-11-01, candidate 8817f63c)
- Tarball sha256 `85a03032…`; **8/8 bp1+bp2 backports absorbed** (patch chain
  drops); #38565 (bp3) also absorbed — no patcher needed.
- Overlay replay: rocenante 20/21 hunks clean (1 reject = PyNcclCommunicator
  `__init__` `roce_enabled` kwarg, upstream signature moved — mechanical
  re-derive); `roce_tp4_adapt.py` 4/4 probes PASS on the patched tree;
  22/23 sitecustomize-hooked modules exist (encoding_dsv41 gone = expected,
  dormant hook); flash_mla_sm120 target exists.
- Wheel stack: candidate pins torch==2.14.1 (same generation) + CUDA 13 line
  → pip-layer path per skill (no base rebuild unless >2 blockers appear).
- REMAINING (critical path): (1) rocenante hunk re-derive; (2) adapter
  re-derivation — engram_backend/engram_prefetch vs the wave-rewritten
  engram.py (1388→1540 lines, 54→60 defs), prefill_sp tables, b12x_next
  needles, dspark-family adapters vs spec rework; in-image test replay;
  (3) new Dockerfile (drop patch chain); (4) flashinfer top-k
  is_deterministic probe (0.7.2 may predate #4831 — #38565 self-falls-back);
  (5) rust rebuild expected (rust tree moved).
- **Timeline**: settle gate = mem_cache quiet ≥2 days from 10-31 13:41 -0700
  → gate passes 11-03 ~05:00 +0800. Adapter work 11-01/02. **Earliest
  window 11-03 (CN daytime)** if adapters finish + waves stay quiet; else
  11-04. Window ≈ 2.5-3h (rust rebuild + JIT-cold boot + full gates +
  qeval; auto-rollback staged). Re-sweep + re-verify absorption the evening
  before the window (candidate may move; re-derive is cheap now that the
  overlay anchors cleanly).

### Stability window 11-02/03 CLEARED (audit 2026-11-03 17:16 +0800)
- **2.5 天零事件**: 4/4 节点 Up 2 days healthy、RestartCount=0、节点零重启
  (up 9d, last reboot 10-25)；head+3 workers 自 11-02 00:00 起日志 0 条
  fatal pattern (traceback/CUDA error/OOM/SIGKILL)、0 次 evict/retract、
  HTTP 4xx/5xx = 0、watchdog (pid 2115579) 0 恢复动作。1518 请求已服务。
- 今日 smoke 4/4 PASS + temperature=0 json 抽检正常。KV pool full=5704448
  与迁移时一致（零漂移）。swap 1.7-2.1G/16G 静态占用, si≈0 — boot 期
  fast_load slab 冷页外推, 非 thrash; 可观察项。
- **回滚梯子可以退役一档**: `canary-mig-bp2` ×4 已稳 2 天, 按惯例
  pre-adoption 一档 (bp1) 之前已删; 下次构建前可清理孤儿 tag/镜像。

### sglang sweep 2026-11-03 (180 commits, 8817f63c..fbe54087; MiaAI-Lab: 0 new)
15. **Tier-1 bp 候选（已全部离线验证）**: **#46088 decode mHC post+combine+
   RMSNorm 融合内核**（+**#47949** CTA-cluster 小 batch split, 依赖 46088
   文件 + `srt.utils.get_device_core_count` — **在 pin 镜像内 import 实测
   OK, 48 SMs**）+ **#47684** dense indexer logits 整 SM wave 取整
   (self-contained `_num_sms`) + **#42968** indexer TP shard（默认 OFF,
   `SGLANG_OPT_DSV4_INDEXER_TP_SHARD`, opt-in A/B 型; scoring.py 大改需
   独立验证）。四者对 pin 8817f63c **apply-clean, 堆叠 46088→47949 与
   42968→47684 均通过**。#46088 门控全命中: live 探针 is_blackwell=True
   (property, SM120 通行), mult=4, hidden=5120, pre_from_prev ✓; 六 adapter
   无一 hash-gate 这些文件 → 无 drift-guard 耦合。
16. **Skip with evidence**: #47512 sampler JIT (softmax 半边 capability
   `max_sm=(10,10)` **排除 SM120**; argmax 半边 CUDA-only 可用但单独价值
   减半 — pin 迁移时随新基线重新评估) ; #38719 router FP32 精度 fix 只改
   bailing/ernie/kimi_linear/mimo 模型文件, 不碰 deepseek — no path。
   #46478 DSpark FA drafts query-width fix: apply-clean, touch
   flashattention_backend.py; spec 路径, 需窗口内 wire probe 验证, 单独
   价值中等。#45129 anthropic thinking.display=updates (apply-clean;
   我们跑 anthropic 协议, 客户端兼容性 fix, 零 perf)。
17. **Pin 迁移 settle 门未过**: scheduler request-lists 族 #43900-#43909
   **今天还在落** (最后一个 11-03 00:53 -0800), memory-pool 侧车族
   (#47714/#47675/#40914/#40916) + imports 移除 re-export shims
   (#47990/#47992) 11-02 仍在落 → mem_cache/scheduler 重构 wave 未静默。
   维持原判: settle 观察重启, 最早 ~11-05; bp 包先行, pin 迁移等 quiet ≥2 天。
18. **下个窗口的 bp 包建议 (bp4)**: #46088+#47949 (同文件捆绑) +
   #47684 (self-contained) 为核心; #42968 默认 OFF 可同 image 预埋
   (flag off 不执行); 可选 #46478/#45129 (spec/兼容 fix, 零 perf)。
   窗口流程照 bp2 手册: retag 回滚点 ×4 → in-image pre-verify (JIT 触发
   mhc_post_combine_norm_decode 编译) → qeval before (worker URL) →
   build → boot → gates + decode A/B (46088 主刀 decode 路径)。
   **DONE 2026-11-04/05 — see §2 production line. bp4 ADOPTED.**

### Perf headroom audit 2026-11-03 (post-window)
19. **SPS verify-budget 调度**: 表已于 2026-11-05 profile 并安装
   (`state-tp4/dspark_sps.json`, 7 探针 bs 1-32, self-check ✓ — 首张表)。
   boot.py 仍因 **sgl#39173** 拒绝 compact 模式（boot log 有明确拒绝行），
   planner 维持 verify-all。启用条件: #39173 落地后 `DSPARK_SPS_ALLOW_ENGRAM=1`
   越权 + 验证 graph capture 不再断言。profiler 全流程与两个坑见 §2
   （traffic 污染 + dp_rank-0-only 记录 vs 多 rank 校验假设）。
   `--speculative-dspark-align-verify-tokens-to-graph-tier` 语义 =
   "把 verify 窗口填满到 forward 本就 pad 到的 graph tier"（免费验证）。
   **verify_cap 重移植仍被取代**: 上游 planner 是它的官方版。
   `#47512 softmax SM120 排除（pin 迁移时只评 argmax 半边）` — 归属 §16 skip 清单。
20. **旧头寸仍有效**: gamma=8 code +12.3%（10-04 实测，需 dual-graph W2
   或 compact 模式救 prose）；conf:0.1 阈值 refit 已证伪（凸成本曲线，
   0.1 即最优）；KV pool 非瓶颈（0 evict，8.5× headroom）；

### sglang sweep 2026-11-05 evening (79 commits, fd19cfb6..732d34b9; MiaAI-Lab: 0 new; 0 reverts)
21. **bp5 裁决（全部证据在手）**:
    - **ADOPT 候选**: **#47728** dspark bs=1 SWA ring slot staging
      correctness fix（生产面仅 stage_commit `.contiguous()` +1 行 +
      dspark_verify assert; apply-clean on pin; bs=1 单流是我们主力形态
      — 低风险高价值）。**#47935** Engram MXFP4 packed tables + gather
      E2M1（表内存减半: row_bytes dim→dim/2; apply-clean; engram.py
      hunks 在 _PeerShards/shard_range/EngramEmbedding.__init__ 区,
      adapter hook 面 init/new_rows/_owned_rows 未变 → 兼容, 但同文件
      大改 → bp5 预验时 boot log 必查 staged/peer 属性面）。
    - **NO-OP on our line**: #43786 wo_a inverse-RoPE FP8 融合 — live
      probe `wo_a_fp8=False`（引擎 dequant wo_a 到 bf16, fp8 twin 是我们
      adapter 的 Triton 路径; 上游 CUDA 融合路径 dormant）。#48467
      trtllm_gen sparse MLA multi-CTA counter — 我们跑 flash_mla_sm120,
      非 trtllm backend。#47201 kernel-plugin 绑定基建 — 非插件路径。
    - **Pin 迁移级（不 bp）**: #48424 deep-gemm 0.2.6 bump + MQA logits
      API 变化 + mxfp8_input scale layout — 栈升级型, 随下次 pin 走。
    - #41294 Hopper low-ratio compressor — Hopper-only。#47494 mamba
      tracking refresh — NPU 路径。#48001/#48283/#48327/#48003-#48005
      spec/mem_cache/scheduler 重构族 — settle 未到, 全部 pin-only。
22. **Settle 门**: 上次 scheduler/mem_cache 活动 11-04 21:54 (-0800) =
      11-05 13:54 +0800 → 若今晚起静默, 最早 11-07 判定。pin 候选 tip
      732d34b9 (11-05 18:07 +0800)。下次 sweep 直接从 732d34b9 起。
23. **bp5 窗口建议**: #47728 + #47935 一窗（apply-clean 双 patch 链,
      照 bp4 手册: exact-commit diff → runtime/bp5-pr*.patch →
      Dockerfile 链 → in-image pre-verify（重点: engram boot 属性面 +
      #47935 后 packed 表内存对账）→ retag ×4 → qeval before/after +
      smoke + Engram hit_rate/内存对比）。Engram 表内存减半对 128GB
      UMA 是实际红利（每层 ~1-2GiB host cache → 同预算 rows 翻倍或
      cache ways 提升）。

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
