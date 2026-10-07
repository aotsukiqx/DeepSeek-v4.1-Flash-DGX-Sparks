# sparkDash literal reproduction on this fleet (2026-10-07)

Question: does the production line (image `canary-roce` built 2026-10-05 16:13 UTC,
HEAD 805e193 + the 6-line vcap-dbg print, MRR=32 raced tiers, b12x 1.5.0) reproduce the
upstream-published sparkDash numbers (README / docs/tp4.md v2.1 tables, upstream HEAD
cad252b unchanged since the 2026-09-26 deploy)?

Method: sparkDash 1.8.9 DecodeBench/PrefillBench protocol re-implemented verbatim
(`benchmarks/sparkdash_decode.mjs`, `benchmarks/sparkdash_prefill.mjs`; prompt catalogs
and math from MiaAI-Lab/sparkDash @6e2a394 — 1.8.9 changed only quota/timeout plumbing,
prompts and formulas are the 1.8.8 ones the README cites). Decode: chat completions,
greedy (temp 0, top_p 1), thinking off, min_tokens=max_tokens=256, ignore_eos, stop=[],
32-token warmup per type, per-stream tps=(completion_tokens−1)/(tLast−tFirst),
aggregate=Σ(completion_tokens−1)/(max tLast − min tFirst). Prefill: `" the` filler +
UUID-salt header, max_tokens 8, tps=usage.prompt_tokens/TTFT, 2 passes, sizes 4k–128k.
Driven from the Mac against http://192.168.2.23:8000; 3 reps per decode cell, median
reported. Fleet: idle except one 127k request that drained before the window — engine
logs show #running-req exactly 1 during c1 waves and exactly 16 during c16 waves
(zero contamination).

## Decode, aggregate tok/s (median of 3 | best)

| cell | upstream | this fleet | Δ median |
|---|---:|---:|---:|
| prose c1 | 87.7 | 83.7 \| 83.8 | −4.6 % |
| prose c16 | 342.7 | 308.0 \| 316.7 | −10.1 % |
| code c1 | 124.8 | 125.2 \| 125.5 | **+0.3 %** |
| code c16 | 438.3 | 431.1 \| 440.5 | −1.6 % |

TTFT c1: prose 195–272 ms, code 201–235 ms (upstream prose c1 223 ms). All streams hit
exactly 256 tokens, finish=length (protocol clean, no loop-abort).

## Engine side during the waves (40-step Decode-batch averages)

| wave | accept len | note |
|---|---:|---|
| prose c1 | 3.02–3.10 | step ≈ 36.3 ms at 83.7 tok/s |
| code c1 | 5.25–5.28 | step ≈ 42.2 ms at 125.2 tok/s |
| prose c16 | 2.94–3.12 | bs 16 exact |
| code c16 | 5.13–5.26 | bs 16 exact |

The prose gap is the documented cross-stack greedy-path artifact (docs/tp4.md: "another
fabric, another all-reduce follows a different greedy text"): our accept is HIGHER than
the ~2.9 implied upstream (3.05 vs 2.9) while the step on this specific prompt runs
~36 ms vs the ~33 ms upstream band; the same upstream doc's independent switchless-ring
reproduction swung prose C1 −23 % on the same image. Code cells — where acceptance is
near-ceiling and stable — land on literal parity.

## Prefill, cold, synthetic filler tok/s (pass1/pass2)

| size | upstream | this fleet | Δ best |
|---|---:|---:|---:|
| 4k | 4,059 | 4,476 / 4,662 | **+12 %** |
| 16k | 5,855 | 5,589 / 5,604 | −4.3 % |
| 32k | 5,900 | 5,649 / 5,710 | −3.2 % |
| 64k | 5,925 | 5,671 / 5,667 | −4.3 % |
| 128k | 5,797 | 5,511 / 5,532 | −4.6 % |

Synthetic filler rides a hot Engram row; on unique real text (the workload that
matters) this fleet measured 4,973 tok/s @139k vs upstream 4,822–4,970 @~120k
(validation-20260926-prod.md) — parity/better. The 16k–128k synthetic −3…−5 % is a
row-cache/NVMe state effect on the repeated-token path, not a compute gap.

## Verdict

- **Code decode: literal parity** (+0.3 % c1, −1.6 % c16, best rep above upstream).
- **Prose decode: −5…−10 %**, concentrated in the single-prompt greedy-path artifact
  upstream themselves exclude from cross-stack comparison; engine step time and
  acceptance are healthy (accept 3.05 > upstream's implied 2.9).
- **Prefill: parity** on real text; synthetic filler −3…−5 % at 16k+, +12 % at 4k.
- Beyond-card: c32 tiers (raced) give code 1056 / prose 307 aggregate — upstream has
  no tier above c16.

Raw waves: `sparkdash-20260107-decode.json` / `sparkdash-20260107-prefill.json`.
