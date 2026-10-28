# Upstream issue draft: post-hoc `Req.finished_len` mutation desyncs pool accounting (invariant checker fires with negative residue)

**Target repo:** sgl-project/sglang (verified on main @ 5cc9c933, 2026-10-27;
first observed on 620c76f2 in production)
**Type:** robustness / undocumented contract
**Not yet filed** — draft prepared 2026-10-28 from a production incident
(2026-10-15, 49 min outage) on a 4× DGX Spark (GB10/SM120) TP4 deployment of
DeepSeek-V4.1-Flash with DSpark speculative decoding.

## Summary

`Req.finished_len` has an implicit contract: it may only be set **inside**
`update_finish_state`'s own flow (stop-string/regex via `_locate_str_stop_finished_len`,
`_cap_finished_len_at_max_new_tokens`). Engine-side consumers correctly honor
it (e.g. `BatchResultProcessor._visible_output_len`, speculative sampling-mask
row drops "past finished_len, as output_ids is trimmed there"). But nothing
prevents an extension from shrinking `finished_len` **after** the step's
bookkeeping has run on the longer length. When that happens, pool accounting
desyncs and the invariant checker
(`scheduler_components/invariant_checker.py::_check_pool_invariant`,
`available + evictable + protected + session_held + uncached == total`) fires
with **negative residue** (`total_accounted < total`) — i.e. over-released
tokens — and, where the checker is in hard-error mode, takes the scheduler
down.

## Reproduction pattern

Any wrapper of `Req.update_finish_state` that terminates a request early and
retroactively trims output, e.g. a repetition-loop detector (our production
adapter, and the MiaAI-Lab bdf657c lineage `FINISH_ABORT` loop-abort, both do
this):


```python
def update_finish_state(self, new_accepted_len=1):
    original(self, new_accepted_len)
    if self.finished():
        return
    hit = detect_repetition(self.output_ids)          # any retroactive decision
    if hit is None:
        return
    self.finished_reason = FINISH_MATCHED_STR(matched="repetition")
    self.finished_len = keep_one_cycle_copy(self.output_ids)   # ← post-hoc shrink
```

Under speculative decoding (multi-token accept steps) this reliably trips the
checker within a few finished requests: per-step token accounting (KV/radix
release paths, incremental detokenizer state such as `cur_decode_ids_len`)
computed against the FULL emitted length, while the finish-side consumers
derive from the shorter `finished_len`. Two readers, two truths.

Observed signature (2026-10-15 incident):

- checker message: `[token_to_kv_pool] total=..., available=..., evictable=...`
  with `total_accounted - total < 0` (over-free direction)
- requests in flight at the time had `finished_len < len(output_ids)` set by
  the wrapper in the SAME scheduler iteration
- removing ONLY the post-hoc `finished_len` assignment (keeping the early
  finish) makes the violation disappear entirely

## Prior art checked (not duplicates)

- **#31599** (closed, fixed by #31569): the adjacent ordering bug in the same
  function — `max_new_tokens` cap evaluated before the stop checks masks an
  in-budget stop and leaks post-stop tokens **to the client**. Engine-internal,
  finish-reason selection; no pool accounting involvement, no extension angle.
  Our report is the complement: an **extension** mutating `finished_len`
  **after** the flow desyncs **pool accounting** (checker over-free).
- Pool-leak family — #39838 (session_held sessions), #35270 (radix
  `evictable_size()` double-count), #38840 (shared boundary page double-free):
  all distinct root causes; none involve `finished_len` or post-hoc mutation.
- No upstream repetition/loop-detector feature or issue exists as of
  2026-10-28 (search: `repetition loop detect`, `post-hoc`, `output_ids trim`).

## Suggested directions

1. **Document/enforce the contract**: one line in `Req.finished_len`'s owning
   section ("set only inside `update_finish_state`; post-hoc mutation voids
   pool accounting") — cheap, prevents recurrence.
2. **Offer a supported trim API** for extensions: a `Req.finish_with_trim(finished_len)`
   that performs the shrink inside the accounting flow (updating the same
   bookkeeping the stop-string path does), so early-finish helpers don't have
   to choose between leaking the loop tail and corrupting the pool.
3. Optionally, make the checker's report distinguish over-free (negative
   residue) from under-free — the direction is the debugging breadcrumb.

## Our mitigation (for the record)

The wrapper now finishes early WITHOUT touching `finished_len`
(`DSV41_LOOP_ABORT_TRUNCATE=0`): the request still terminates in ~128 tokens
instead of the completion cap, the client sees the loop tail. Zero recurrences
since 2026-10-15 across two pins (620c76f2, 7d996300) in 24/7 traffic.
