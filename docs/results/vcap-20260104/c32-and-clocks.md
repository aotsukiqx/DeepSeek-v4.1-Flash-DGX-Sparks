# c32 tier + load clocks (2026-10-04 evening): both rejected/parked, production stays MRR=16

## c32 tier (MAX_RUNNING_REQUESTS=32, graphs to bs 32, b12x capacities to 32x6)

Operational note found on the way: `DSV41_MOE_B12X_NEXT_GRAPH_BS` is an adapter-level
knob and only takes effect inside `EXTRA_CONTAINER_ENV`; a standalone `.env.tp4` line is
silently ignored (the adapter warns at boot when the engine's capture list exceeds it).
Also: the first tier bench after any boot runs on a cold Engram cache - discard it
(the prod boot's cold prose_c16 read 122.6 vs 205 warm; only warm numbers compare).

Warm, same-caliber, same-boot-pair comparison:

| tier | production (MRR 16) | c32 arm | verdict |
|---|---:|---:|---|
| c1 prose / code | 53.3 / 118.2 | 54.5 / 113.3 | unchanged (boot noise band) |
| c16 prose agg | 204.9-206.6 | 210.8 | unchanged |
| c16 code agg | 724.6-734.1 | 714.2 | unchanged |
| c32 code agg | - | **667.9** | **-8 % vs c16** |
| KV pool | 6.44M | 5.67M | -12 % |

Code aggregate REGRESSES at 32 streams (the MoE/bandwidth saturates around c16);
prose aggregate rises (+58 %) but per-stream is already slow and the code loss plus the
KV loss dominate for a mixed fleet. **c32 rejected**; revisit only if real traffic shows
queueing at 16 (the config is one env change: MRR=32 + CUDA_GRAPH_MAX_BS_DECODE=32 +
EXTRA_CONTAINER_ENV graph-bs list, this file documents the cost).

## Load clocks (sampled on the head during warm c16 decode)

2197-2210 MHz at 8.5-9.5 W across the load window - **no thermal or power sag**; decode
is memory-bound with the compute side nearly idle. The 3003 MHz max SM clock is not
reached under this governor. Pinning clocks higher (community reports ~2300 MHz on GB10)
is a persistent-clock-setting policy decision, not a technical blocker, with modest
expected upside; parked for the user.

Artifacts: bench-c32-{c1,tiers}.json, bench-prod-c16{,-warm}.json, bench-prod3-c1.json,
/tmp/tier-warm2.json, /tmp/clocks2.log (head).
