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
