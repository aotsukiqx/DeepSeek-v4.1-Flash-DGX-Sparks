# DSV41 TP4 production plan (2026-10-05, post-campaign master document)

Fleet: spark-eb1a head + 3 workers, serving DeepSeek-V4.1-Flash, TP4/EP1.
This is the umbrella plan; engineering detail for the mainline lives in
`docs/results/vcap-20260104/dual-graph-spec.md` (15+ rounds, all anchors).

## Current production state (verified healthy at writing)

- gamma=5 (DSPARK_BLOCK_SIZE=5), DSV41_VERIFY_CAP=conf:0.1, MRR=32 with fully
  raced graph tiers (adopted today: code agg +47% / prose +56% at c32, c1
  unchanged, qeval BROKE=0; KV pool 5.30M)
- b12x_next pinned at 1.5.0 (e4084d2e); the two plan pins (<=8 rows, M64) are
  mandatory (-47% if removed)
- Env-gated available: gamma=8 EXT line (DSPARK_BLOCK_SIZE=8 + the confidence
  shim, commits e1fb298..90d7ad4): code +12.3%, patterned long-context
  continuation likely +20-30% (accept 5.9 saturates gamma=5 at 492k), but
  novel-output prose -12.9% -> NOT flippable on mixed traffic
- Repo HEAD synced Mac<->cluster; rollback images tagged on all nodes

## Priorities (reviewed; one branch point)

P0 - TRAFFIC PROFILE CHECK (5 min, gates the biggest immediate lever):
  `docker logs dsv41-head | grep "Decode batch" | accept-len distribution` over
  real traffic. Dominant accept>=4 (patterned/continuation-heavy) -> flip
  gamma=8 TODAY (one env + reboot) for +20-30% on that class, accepting -13% on
  novel prose. Mixed/novel-dominant -> do nothing now, P2 is the path.
  CAVEAT: logs reset on container restart; needs live traffic. If traffic is
  off, skip to P2 and run P0 when it returns.

P1 - CLIENT DISCIPLINE (zero cost, parallel): multi-turn long-context clients
  must reuse prefixes (radix cache hit skips the whole ~105 s prefill at 512k).

P2 - ENGINEERING MAINLINE: dual-graph W2 (1-2 focused days). Makes gamma=8
  free for novel prose -> default-on for everyone. Status: W1 complete (both
  families captured, healthy boot, code +12.3% kept, commit f6973a81+); W2 =
  selection wiring. Exact next steps in dual-graph-spec.md:
    a. variants.select hookup (runner selects the narrow family per batch)
    b. replay metadata routing per family (L3 stores exist)
    c. capture_hook batch slicing (draft side native 8 rows, verify 6)
    d. verify_cap STRIDE family-coupling (rows//STRIDE remap)
    e. R1 oracle FIRST (dual-execution comparator + boot-time self-check;
       silent-corruption trap [4,8] documented - never trust shape-compatible
       success without it)
    f. FORCE=narrow|wide hook -> Oracle 2 gates (greedy repeat, accept dist,
       qeval BROKE=0, 190k needle) -> mixed soak -> adopt
  Open item folded in: gamma=8 prose 39.2-vs-49.9 needs the interleaved A/B
  (single-boot data was in the tactic band).

P3 - CONTEXT_LENGTH 1M -> ~532k (one boot, measure pool delta; low value,
  zero risk, do opportunistically with any P2 boot)

P4 - BACKLOG (statused, no action unless triggered):
  - Quant fusion (+~2% prose, kernel work) - prose grinding is otherwise
    closed (L2 sweep flat, race expansion flat, thresholds/pins/trees/c32-old
    all measured out)
  - Clock pinning ~2.3GHz (user policy decision; no sag measured, 9W load)
  - Display-reserve Engram layer (+1.8GiB, host change; only under KV pressure)
  - c48+ tiers (curve still rising at 32; needs real traffic justification)
  - gamma=8 dual-graph adoption ALSO unlocks per-workload gamma via EXT rule

## Session lesson index (measurement discipline)

- Idle-window gate: check `Prefill batch` count before any bench; engine gen-tp
  RISING while wall falls = contamination tell
- Cold-boot first tier run is garbage (Engram cache); discard
- Same-tactics boot pairs hold prose noise to ~0.6%; cross-boot band is 39-57
- Adapter-level knobs ride EXTRA_CONTAINER_ENV only (standalone .env lines are
  silently ignored); reaper window must cover full boot (rustup probe kills
  the HTTP frontend silently); never bench unraced graph tiers (the c32
  "knee" was plan quality); hotpatch must reach ALL FOUR containers
- Long-context acceptance is content-dependent (patterned 5.9 vs novel 2.0),
  not length-dependent

## Asset index

- Tools: scripts/verify/{vcap_capture,vcap_fit,ab_bench,tier_bench,
  boot_hygiene}.py (row-floats/log-path parameterized)
- Result docs under docs/results/vcap-20260104/: dual-graph-spec.md (mainline),
  concurrency-adoption.md, long-context-512k.md, gamma8-probe.md,
  l2-sweep.md, race-expansion.md, c32-and-clocks.md (overturned note),
  b12x150-upgrade.md, RESULTS.md (verify_cap negative)
- qeval baselines on spark-a89f:/tmp: qeval-before.json (production),
  qeval-c24/-c32r/-gamma8.json
- Long-probe generator pattern: /tmp/longprose.py (seeded varied prose,
  char-cap 2.0M ~ 492k tokens; never repeat-token filler)
