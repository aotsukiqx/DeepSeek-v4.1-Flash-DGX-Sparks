# Incident 2026-10-25: cluster-wide reboot + desktop-session CPU noise (2x perf halver)

## Timeline (all times local, UTC+8)
- 12:07: ALL FOUR NODES went down together (previous boots, up since 10-09, all ended
  12:07-12:08) — cluster-level power/infrastructure event; the engine's exit-247
  "crash" was a symptom. Nodes auto-powered-on at 12:15-12:16.
- 12:15: Ubuntu graphical session auto-login woke desktop services on a89f + 2fe7
  (snapd-desktop-integration + wireplumber + pipewire + systemd --user ≈ 1.25
  cores/node). Head/141e quiet. RoCE collectives SPIN on CPU (spin_limit 2e7), so
  the noise halved the whole TP4 engine: cold 600k 2088 tok/s (band 3585-3853),
  decode 19 (band 40-47), GPU clocks 73% with 44 W @ 96% "util" (spin-wait signature).
- 12:23: engine restored on the rebooted nodes (still degraded, unmeasured cause).
- 13:11: causal test — SIGSTOP desktop procs on a89f+2fe7 → cold 2088.6 → 3839.7
  INSTANTLY. Procs auto-respawned via systemd --user at 13:14.
- 13:16: PERMANENT FIX fleet-wide (no sudo needed):
  `systemctl --user stop pipewire pipewire-pulse wireplumber
  snap.snapd-desktop-integration.snapd-desktop-integration xdg-document-portal
  xdg-permission-store` + `systemctl --user mask <same>` — masks persist per-user
  across reboots. Post-fix cold 3884 (top of band).

## Contamination window (bounded): 12:15-13:15 on 2026-10-25 ONLY
Everything before 12:07 ran on the pre-reboot uptime (up since 10-09) — all
historical baselines (10-11 matrix, 10-17 A/B, 10-18..24 bundle verifications,
mig windows W1-W6 this morning) are UNCONTAMINATED. The mig-window load slowness
(54-120 s/shard) was real (cold cache / fast_load-EP2), not desktop noise.

## Hard rules added
- Bench preflight MUST include the fleet health check (see skill): desktop-proc CPU
  ≈ 0 on all nodes; uniform-slowdown signature = clocks<max + power low at high util.
- Probes live in the REPO now (scripts/verify/{probe600k,decode600k,svc_smoke}.py);
  /tmp versions age out (systemd-tmpfiles) — deploy via `cp` to /tmp on the head.
- The 12:07 synchronized reboot needs ops follow-up (PDU/rack power? watch for recurrence).

## R2 experiment result (2026-10-25 14:06-15:05, post-desktop-fix)

conc6 aggregate (6×600k, probe600k.py conc from scripts/verify/):
- RoCE ON (production config): **3883.8 tok/s** (top of the historical band)
- RoCE OFF (SGLANG_ROCE_ALLREDUCE stripped): **3209.4 tok/s** (**-17.4%**)
- cold single (roce off): 3754.6 (parity); decode 20.3 (traffic-confounded both arms)

**Verdict: the RoCEnante overlay is a +17% conc6 ASSET — fully exonerated for the
10-11 4652 vs 10-17 3838 gap.** That gap is not roce, not weights, not pool, not
desktop noise (pre-reboot era) — remaining explanation is measurement-era/stack
state; the mig validation window's conc6 probe (vs both 3884 today and 4652) is the
last open test of the scheduler-admission family.

Ops notes from the window: the scripted rollback's nohup hit a transient
"no such file" and failed (engine down 14:48-15:05, manual restore ~3 min);
the new serve watchdog was correctly held off by the maintenance marker.
Watchdog validated-by-design; markers are now part of every window protocol.

## H0a ADOPTED + H0b partial (2026-10-25 afternoon)

H0a `DSV41_PREFILL_EMPTY_CACHE_EVERY=4` (empty_cache on every 4th qualifying
forward; py-spy: per-chunk calls cost ~4% host): cold 600k **4101.4 / 4121.4**
(band top was 3884; +6%), decode net **44.0** (top of band). qeval ×2: PRIMARY
BROKE=0 both; one secondary flip = json_count (got 11 want 10), the same
borderline task that flipped in BOTH directions across knob-less boots this week —
stable within-boot, recorded as open observation (reverse-boot check folds into the
next natural reboot). Adopted in production env.

H0b decode-vs-concurrency matrix (scripts/verify/decode_matrix.py, live traffic):
tier1 41.7 | tier2 31.8-37.7 (agg 63.6) | tier4 21.1/3.0/21.7/3.0 | tier6 median
2.8 — tier>=4 streams COLLAPSE (3 tok/s class), but the window carried live
production traffic (pool-pressure retraction suspected). Clean-window re-run +
retract-log analysis queued; do not cite tier4/6 numbers yet.
