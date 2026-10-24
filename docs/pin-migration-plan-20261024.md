# Pin-migration dry-run + campaign plan — 2026-10-24 (pre-campaign review)

Upstream re-sweep before the production-optimization campaign: MiaAI-Lab **zero new
commits in 4 days** (delta still the 31 of 10-17, all already triaged); sglang added
~400 commits (10-19 → 10-24), dominated by a DSV4.1 host-overhead elimination wave.

## Dry-run on candidate 7d996300 (10-24 04:33, post-MegaGate-revert)

| Step | Result |
|---|---|
| `sglang-rocenante.patch` (patch -p1 -F3) | **rc=0, 0 rejects** |
| `roce_tp4_adapt.py` (path-sedded) | **rc=0** — all four assert-count-1 drift probes PASS |
| `encoding_dsv41.py` | **GONE** — replaced by chat_encoding*.py + serving_deepseek_recipe.py + parser/deepseek_recipe_{renderer,output}.py (#45315/#45907/#46219) |
| our 3 serving patches | must be re-derived on the recipe architecture (or dropped if upstream absorbed the semantics — NOT yet verified either way) |
| dspark_components | **reorganized** (new dspark_block_accept_estimator / dspark_deferred_commit / dspark_kv_inject / dspark_observability; old verify file split) → block_verify/verify_cap/folded_fence adapter re-ports |
| parser | #40479 response-template adapters restructures response_parser.py → the staged #43941 patch re-derives on migration |

Upstream churn signal: MegaGate routing merged #46232 and **reverted #46242 within 8h** —
pin target must be a settled point, not HEAD.

## Campaign plan (order + gates)

**Window 1 — serving bundle #3 (tomorrow, ~40 min):** staged commit 1c36c98
(sglang#43941 whole-number-float tool args; script byte-identical to upstream patch,
tests green, Dockerfiles wired). Pipeline: bundle relay → build ×4 → reaper+serve →
boot gates → wire smokes (+ a tool-arg `"count": 3.0` probe) → qeval paired
(BROKE=0) → 600k parity probe. Rollback: pre-bundle3 tag + image retag.

**Window 2 — pin-migration evaluation (multi-session, per dsv41-sglang-pin-migration):**
target = settled commit AFTER the MegaGate revert + encoding-recipe dust settles.
Benefits mapped to our measured bottlenecks (py-spy 10-24): prefill CUDA graphs
default-on direction (#37958/#43733/#45565/#46229-31) kills the CUTLASS-DSL
marshaling ~10% + launch 2.7%; host-sync elimination set (#45223/#45247/#45275/
#45276/#45579/#45573); index-K plan caching (#45577/#45741); DSpark per-step trim
(#46176) + simulated-acceptance fusion (#41169); Engram peer-shard reads (#46180);
full-scan indexer verify scoring (#45251); `--prefill-decode-interval` gating (#44001
— needs arg_groups, addresses the concurrent small-prefill interleave pain).
Costs: prefill_sp structural re-port (mHC/MegaMoE/MegaGate churn), dspark adapter
re-ports, encoding patches re-derivation on deepseek-recipe, arg_groups/encoding-
registry surface, base-stack rebuild from the target's own Dockerfile (nightly base
generational gap protocol). After migration: separate experiment arm re-enabling
prefill graphs (current disable is an old decision; the wave's wins assume them).

**Standing rejections (unchanged):** #44001 standalone backport (flag absent at pin),
#42159 (closed on py-spy evidence), #46127 busy-counter (PD-idle accounting, not our
shape), SM100/AMD/diffusion/PD series.

## Window 1 EXECUTED — serving bundle #3 (2026-10-24 晚, commit 1c36c98 live)

sglang#43941 (whole-number floats for integer tool args) deployed. Full verify:
boot gates green; smokes 5/5 incl new probe (model tool-call args parse clean);
qeval paired BROKE=0 +1 fixed (json_count; jsons spark-a89f:~/qeval-*-bundle3-20261024.json);
600k cold 3751.1 tok/s clean window (first measured run 2398.5 was TRAFFIC-CONTAMINATED:
4 foreign ~4k prefills interleaved — wall 249s vs clean 159s; rerun verified clean),
decode net-of-TTFT 41.3/40.4 — parity. Rollback: pre-bundle3 tag + 4× canary-roce-pre-bundle3.

### Ops incidents hit during the window (runbook additions)

1. **`docker image prune` deletes digest-only bases**: the 10-22 cleanup removed the
   workers' pinned `dev-dsv41@sha256:3dbc...` base (untagged ⇒ dangling ⇒ pruned),
   while the mutable `dev-dsv41` TAG locally points at a DIFFERENT newer image
   (381b27f ≠ pinned 37939c26). Never prune bases kept only by digest ref; after any
   prune, `docker image inspect <digest>` must pass on all 4 nodes BEFORE a build.
2. **docker hub unreachable from workers (EOF on registry-1.docker.io)** — recovery
   proven: on head `docker save <digest-ref>` (31GB) → scp per worker → `docker load`
   restores CONTENT (37939c26) but NOT the digest reference (RepoDigests don't
   survive save/load). Then build with `BASE_IMAGE=lmsysorg/sglang:dev-dsv41` (tag
   form) to pass start.sh's inspect/pull gate — the Dockerfiles' hardcoded digest
   FROM resolves from the LOCAL store in BuildKit (verified), so build content stays
   exactly the pinned base.
3. Remote `nohup scp` inside `ssh -n` dies with the session — use a `setsid`-detached
   script on the relay host.

## Migration EVALUATION complete (2026-10-24 深夜, candidate 7d996300, tarball sha256 5d648297c35a869ee9b27c961d6ce6ed1b8f8b41c6cc605d2d7e9222967d77ed)

Upstream settled (12h quiet after the MegaGate merge+revert churn). Full replay on head:
rocenante patch **0 rejects**, roce_tp4_adapt probes **PASS**, flash_mla target exists.
In-image 24-test drift audit vs candidate tree: **24/24 PASS** (but tests use
SimpleNamespace fixtures — NOT a hash-gate clearance).

**Decisive hash measurement (in-image, inspect.getsource sha256[:16]):**
prefill_sp surface = **0 same / 9 drift / 2 missing** (`_hc_combine`/`_hc_mix_stats`
methods gone; the mHC/MegaMoE/MegaGate wave rewrote the entire wrapped surface).
hc_fused targets: module moved to `kernels/ops/gemm/hc_mix.py` (+`hc_mix_splitk.py`
split, new `mhc_mega`/`mhc` family) — re-port + first-call bit-identity revalidation.

**Re-port inventory (work breakdown, in order):**
1. prefill_sp structural re-port (largest item; the 9 drifted call sites are the
   exactness argument's spine).
2. hc_fused re-port to ops/gemm/hc_mix.py; evaluate the new mhc_mega family.
3. Serving 4 patches re-derived on deepseek-recipe (NONE absorbed upstream —
   verified: no tool_result/placeholder/tool_choice/is_integer semantics in the
   renderer/chat_encoding/response_parser).
4. sitecustomize target list: 15/16 survive at same paths (incl ALL 5 dspark
   anchored modules — additions, not removals); replace the encoding_dsv41 entry
   with the recipe renderer path; re-point moe_fused_gate.
5. Base: same torch 2.14.1 generation; 3 wheel bumps only (flashinfer 0.7.0.post1→
   0.7.2, sgl-deep-gemm 0.2.2→0.2.3.post2, sglang-kernel 0.4.9→0.4.10) — try the
   pip-layer-onto-proven-nightly path FIRST (watch the flashinfer cubin/jit-cache
   pairing), full Dockerfile rebuild only if >2 stacked blockers appear.
6. b12x_next vs the MegaMoE refactor (serve-time check; import-level passes today).

**Verdict: GO as a planned 2-4 session campaign** (①base+wheels bake ②prefill_sp+
hc_fused re-ports with bit-identity ③serving re-derivation + sitecustomize ④full
gates: 24-test, boot, qeval paired, 600k A/B, then the prefill-graphs experiment
arm). NOT a spontaneous flip. Everything above is zero-production-cost evidence;
no cluster state was touched beyond /tmp staging.
