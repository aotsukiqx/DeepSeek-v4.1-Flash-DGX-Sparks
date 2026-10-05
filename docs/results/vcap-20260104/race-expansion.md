# b12x race surface expansion (2026-10-05): flat, closed

TUNE_ROWS now accepts multiple row families per topk ("6:5+6", adapter commit
7a35a83f). Arm: both the verify (6xbs) and draft (5xbs) families raced at load.
Result vs the same-day clean baseline: prose 50.2 vs 49.87 (+0.7 %), code 120.0 =
120.0, gen-tp 65.1 vs 64.9 - noise. The heuristic's plan picks for the draft-row
capacities were already adequate, and/or the draft MoE slice (~10 % of the step)
is too small for plan differences to register at c1. Item closed negative;
production env restored. Prose menu after this: c32 by traffic profile (decision,
data: 16-concurrent bursts with queueing observed in live traffic today; prose
aggregate +53 % at c32 vs code -8 %), then quant fusion (~+2 %, kernel work).
