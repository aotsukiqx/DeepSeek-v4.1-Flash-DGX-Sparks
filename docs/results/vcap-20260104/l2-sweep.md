# L2 prefetch budget sweep (2026-10-05): flat, default 6 MB confirmed

Clean four-arm sweep (idle fleet, warm boots, shared autotune tactics - the four
boots' prose spread was only 0.6 %, far tighter than the day's earlier boot band,
which validates the idle+same-tactics discipline):

| DSV41_L2_PREFETCH_MB | prose tok/s | code tok/s | gen-tp |
|---|---:|---:|---:|
| 4  | 49.55 | 120.0 | 67.3 |
| 6 (default) | 49.87 | 120.0 | 64.9 |
| 8  | 49.57 | 118.2 | 64.9 |
| 12 | 49.55 | 120.0 | 64.3 |

No arm moves prose beyond noise in either direction: the 6 MB default already covers
the useful inter-collective prefetch window (a larger budget cannot be consumed
between RoCE collectives; a smaller one leaves slack that costs nothing). Item closed
with a negative result; production unchanged.

Measurement lesson (added to the discipline): a first 4 MB round read prose 25.1 /
code 57.1 - halved - because live user traffic (up to 16 concurrent, 61 prefills)
shared the window; the engine's own gen-tp lines were HIGHER, the tell for
contamination. Every bench round now checks `Prefill batch` count over the last
minutes before trusting wall-clock numbers.

Artifacts: bench-l2-{base,4mb,4mb-clean,6mb-clean,8mb,12mb}.json.
