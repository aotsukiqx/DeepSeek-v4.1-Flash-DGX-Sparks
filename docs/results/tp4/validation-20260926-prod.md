# TP4 production line validation (2026-09-26, this fleet)

Fresh switch to the upstream v2.1 production line (`Dockerfile.canary-roce`, EP_SIZE=1,
full adapter set; local: NFS_SHARE=0 local weights, f1 HCA pair, port 8000).
Boot 02:52-03:16 UTC (23 min first boot incl. autotune; DSV41_AUTOTUNE_KEEP=1 armed).

Cross-stack comparable metrics (engine log + openai_serving.py + code probes):

| metric | this fleet | upstream docs (switched fabric) |
|---|---|---|
| decode step time, prose c1 | ~32 ms | 32.4-33.6 ms |
| decode step time, code c1 | ~40 ms | 38.9-39.4 ms |
| accept len, code c1 | 4.7-5.2 | ~6 |
| cold prefill, ~139k unique text | 4973 tok/s | 4822-4970 @ ~120k |
| needle 32k x4 / 131k x1 | PASS | PASS |

Aggregate (openai_serving PROSE prompts, accept len 1.5-2.1 on literary prose):
c1 52.9 / c2 80.0 / c4 118.4 / c8 163.8 tok/s.
Code probes: c1 105-110 / c4 200-285 / c8 284 / c16 417 tok/s.

Known state: KV pool 6.75M tokens (fast-loader trade-off, upstream -3..-6%);
DSPARK_SPS_TABLE not profiled (verify-all schedule) — upstream reports ragged
verify crashes the Engram path on this model (docs/astra-new.md), do not enable.
RoCEnante active on f1 rails (ROCE_TP8_ROUTE bytes=983040 at bs16 capture,
DSV41_ROCE_GATHER route bytes=98304). CUDA graphs cover bs 1-16.
