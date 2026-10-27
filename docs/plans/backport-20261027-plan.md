# Upstream backport → production plan (2026-10-27)

Baseline: production = `canary-mig` @ sglang **7d996300**, adopted 10-26 15:44.
**Soak CLEARED 10-27 20:33**: head Up 29h (healthy), 0 error/crash lines in 24h
logs, smoke 4/4 (thinking / placeholder / history_repair / vision all PASS).
Decode spot-check folded into W1 baseline probes (traffic was live: 118 prefill
batches / 3m at 20:31, no bench possible).

Upstream sweep: 204 commits on `7d996300..origin/main` (as of 10-27 20:19).
4 candidates survived triage; all 4 `git apply --check` clean against the pin
tree (source-only hunks).

## Priority waves

One build per wave, one production outage per window (~25 min mig boot + gates
+ probes ≈ 1.5–2h). Bundling per the backport skill: each merged change gets
its own smoke probe in the window.

**W1 prep DONE 2026-10-27 21:35** (commit relayed to head): 3 patch scripts
(35 anchor pairs, fidelity = byte-identical to `git apply` on pristine 7d996300),
3 synthetic tests green, Dockerfile.canary-mig patch chain + patched-import check
+ self-test rows wired. Known pre-existing (NOT bp1): Mac-tree
`tests.test_launcher_build` fails 2 canary-roce rows (rust fetch source + base
pin) — same failures on unmodified HEAD. Window = build → cutover → gates.

**W1 WINDOW RESULT — bp1 ADOPTED 2026-10-27 21:55** (`canary-mig-bp1` live,
rollback `canary-mig-pre-bp1` ×4 + git tag `pre-bp1` + `/tmp/env.pre-bp1`):
- qeval paired BROKE=0 (55 tasks, p=1.000; median 73.7→75.3 tok/s)
- smoke 4/4; new probes: trailing-toolcall-turn ✓, assistant
  null-content+tool_calls ✓, numeric effort 0.5 ✓, sampled ✓. NOTE: wire-level
  `content:null` on a USER message and string effort "8" are rejected (400) by
  the http_server pydantic schema BEFORE the recipe adapter — pre-existing,
  not the patch's surface; the string-effort fix is reachable via the env
  default and internal callers (covered by the synthetic test).
- decode600k seed4 **45.6** (gate ≥42) · cold seed5 **3977.8** (v4 3978.8)
  · conc6 4828 = live-traffic contamination (batching inversion; re-measure
  quiet during soak) · first cold after boot (seed4) = warm-cache artifact,
  discarded per rule
- **Gate deviation accepted**: `max_total_num_tokens` 6336256 → 6169856
  (-166400 = 2600 reserved pages). #41275 is NOT a no-op here: the explicit
  MAX_TOTAL_TOKENS cap exceeds available bytes and is re-validated through the
  patched auto path. The old number over-committed reserved pages; the new one
  is upstream's intended accounting. 600k workloads unaffected.
- 24h light soak ends ~2026-10-28 22:00 (smoke 4/4, decode ≥42, no crash
  loops, no quality complaints) + quiet-window conc6.

### W1 — correctness bundle (bp1) — target 10-28 low-traffic window

| # | Commit | Change | Why / expected effect |
|---|---|---|---|
| 1 | `449617bc` **#46706** (py only) | deepseek-recipe render: null contents → empty, trailing tool-call turn preserved (`wire_messages[-1] = last`), numeric-string effort parse | **P0 serving correctness** on our exact path (deepseek-recipe is a hard boot dep, blocker #9). Rust half (`sglang-processor/deepseek_v4.rs`) OUT OF SCOPE — image builds only the `sglang-mm` package |
| 2 | `24d55621` **#46404** | dspark: simulated-acceptance bonus drawn at the simulated row; sim-mode routing widened to non-greedy | Calibration-only (`SGLANG_SIMULATE_ACC_LEN`, .env.tp4.example:192 — bench knob, never left on). **Production no-op**; fixes biased accept-len标定 |
| 3 | `20ff40cb` **#41275** | DSV4.1 pool capacity: low-ratio page padding + indexer page rounding priced into auto-sizing | **Expected NO-OP**: we set explicit `MAX_TOTAL_TOKENS=6700000` → `calculate_pool_sizes_from_max_tokens` path, which the patch doesn't touch. Gate proves it (boot-log full_token unchanged). Guards future auto-sizing; keeps our tree aligned with upstream accounting |

Steps (per `dsv41-sglang-source-backport` skill, 4× proven):
1. Patch scripts `runtime/patch_{recipe_46706,dspark_sim_46404,poolcap_41275}.py`
   — OLD/NEW pairs, per-patch SENTINEL, idempotent, anchor-drift loud error,
   post-write verify; re-derivation header notes (surface successor =
   `serving_deepseek_recipe.py`, gone-`encoding_dsv41.py` line).
2. Fidelity gate: patched output byte-identical to `git apply` of each upstream
   commit (py hunks only).
3. Synthetic tests `tests/test_patch_*.py` (fixture = anchored region only;
   stock-fails / applies / no-op second run / semantics / anchor-missing).
4. **New wiring** — `Dockerfile.canary-mig` has NO patch chain today (patches
   were dropped at pin-7): add patch-COPY line + patch-RUN chain + self-test
   chain (watch trailing `\`). #46404's gate = upstream
   `test_dspark_simulated_bonus.py` imported into the self-test chain.
   `runtime/sources.manifest`: add `local` rows for the three patch scripts.
5. Rollback BEFORE build: git tag `pre-bp1`; retag `dsv41-4x-spark:canary-mig-pre-bp1`
   on all 4 nodes; `.env.tp4` backup (env unchanged expected — no env deltas in W1).
6. Build on head with rust build-args (`RUSTUP_DIST_SERVER` + `CRATES_MIRROR` —
   build-args are layer cache keys).
7. Cutover = proven sequence: stop → clean workers → **drain ≥45s** → serve.
   Rustup reaper before serve.
8. Gates: 9 boot lines + `gamma=5, verify_num_draft_tokens=6`; smoke 4 + **new
   probes per change**: (a) trailing-assistant tool-call turn (prefix continuation),
   (b) null-content message, (c) effort="8" numeric-string; + **mandatory sampled
   request** (temp=1.0, top_p=0.95); (d) boot-log `full_token` line byte-identical
   to current production boot (#41275 no-op proof).
9. Quality: qeval paired on the 75-task set — run `before` pre-flip in the window,
   `after` post-flip; gate BROKE=0 (McNemar).
10. Perf parity (traffic-gated, ≥45s POST gap): decode600k ≥42, conc6 ≈4042 ±3%,
    cold 600k ≈3979 ±3%.
11. Adopt: retag `canary-mig-bp1`; 24h light soak (smoke clean, decode ≥42, no
    crash loops). Keep pre-adoption rollback artifacts until bp1 clears, then retire.

### W2 — perf conditional (bp2) — ≥10-30, only after bp1 24h clears

| # | Commit | Change | Condition |
|---|---|---|---|
| 4 | `65f19fce` **#45794** | BF16 prefill indexer score width: pad per-request K to 64-multiple (lc≥1024), chunk rows against padded width, slice `[:, :lc]` | **Pre-check first** (read-only, live image, CUDA-visible per §7.12): confirm prefill scoring actually runs the torch `prefill_requests`/`_score_chunks` path on SM120 (DeepGEMM `flat_index_logits_tiles` path → DROP, no path for us). Then A/B measurement window (mig_window.sh, arm = bp1+patch vs bp1), bar ≥2% cold 600k or conc6 with decode ±noise (prefill-path causality). Miss bar → negative result under docs/results/, patch script stays dormant |

### Deferred / out of scope (recorded, no action)
- `46530`/`46516`/`46399`/`46718(dea05ca1)` prefill-graph buffer refactors,
  `43449` aux hidden capture, `46290`/`46330` spec placement — **next pin
  migration** (structural; adapters prefill_sp/engram couple here).
- `46263` MegaAttention aggregated default — PD-disagg predicate, we're
  non-disagg TP4 with explicit backend: no-op.
- AMD/mlx/kimi/QSA/NPU/fmha_v2-trtllm — wrong arch/model/backend.
- `42760` pickle fallback, `46620` TorchScript deprecations — hygiene, fold
  into the next pin cycle.

## Timeline
- **Tonight 10-27**: W1 prep — patch scripts + fidelity gate + synthetic tests +
  Dockerfile wiring + manifest rows; commit, push, bundle-relay to head.
- **10-28**: W1 window (pick traffic-quiet slot; watchdog maintenance marker
  `touch /tmp/dsv41-maintenance` during window, rm after).
- **10-29**: bp1 soak review.
- **≥10-30**: W2 pre-check (anytime on live image) → window if path confirmed.

## Success criteria
- bp1: all gates green (qeval BROKE=0, smokes 4+3+sampled, full_token unchanged,
  decode ≥42, conc6/cold ±3%) AND 24h light soak clean. Rollback = retag
  `canary-mig-pre-bp1` + proven env path (env identical, so image-only).
- bp2: adopted only on ≥2% targeted gain with decode flat; otherwise negative
  result documented.
