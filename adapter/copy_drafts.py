"""Prompt-lookup copy drafts: exact n-gram copies in front of the DFlash2 draft. Default OFF.

With DSV41_COPY_DRAFTS set, every step first searches each request's committed history
(origin_input_ids_unpadded + output_ids at propose time; no unverified drafts) for an
earlier occurrence of the trailing MATCH-gram and overwrites the LEADING rows of the
proposed draft block with the tokens that followed that occurrence:

  * k <= min(DSV41_COPY_MAX, gamma, followers available) rows are written in place into
    draft_tokens; the [bs, gamma] block shape, the anchors and proposal_block_ids are
    untouched, and requests without a match keep their DFlash2 rows bit-for-bit;
  * greedy requests: verification compares tokens, so an exact copy accepts and a wrong
    copy stops the accept and commits the target token - identical to any wrong draft;
  * sampled requests: the copy rows' corrected_logits are overwritten with a one-hot
    (1e4 at the copy token, 0 elsewhere), so q after SoftmaxTemp is exactly the copy
    token while the rejection step uses the untouched target p - the distribution the
    engine would see for a draft that is itself one-hot. Skipped when corrected_logits
    is None (greedy-only folded/eager paths, where accept compares tokens only);
  * the confidence of copy rows is clamped to 1.0 (the rows are exact history tokens,
    and the head sees out-of-distribution inputs): folded steps clamp
    proposal.confidence directly, eager steps clamp the planner's
    compute_confidence_tensor output through install_planner. This feeds scheduling
    only (verify_cap's live length), never the sampling path.

Every downstream read (worker's verify_ids_2d, grammar masks, the folded in-graph
accept) happens after propose returns, so writing the returned views/buffers in place
needs no change to folding or the graph layout. Installed after verify_cap, this wraps
outside it: verify_cap stores the confidence reference first, then the clamp happens on
the same tensor and is visible to it.

The history cache is rid -> (int64 tokens, origin len, output len), extended each step
by only the new output_ids tail (output_ids is append-only by contract; a shorter
sequence means the request was reset, so the entry is rebuilt); rids that left the
batch are dropped lazily.

Env:
  DSV41_COPY_DRAFTS  "1"/"on" enables (default off)
  DSV41_COPY_MATCH   match length, default 8
  DSV41_COPY_MAX     max copy rows per step, default 5 (clamped to gamma)
"""
import os

import numpy as np
import torch

ENABLED = os.environ.get("DSV41_COPY_DRAFTS", "").strip() not in ("", "0", "off", "false")


def _int_env(name, default):
    raw = os.environ.get(name, "").strip()
    try:
        return max(1, int(raw)) if raw else default
    except ValueError:
        return default


MATCH = _int_env("DSV41_COPY_MATCH", 8)
MAX_COPY = _int_env("DSV41_COPY_MAX", 5)

# rid -> (int64 ndarray of committed tokens, len(origin_input_ids_unpadded),
#         len(output_ids) when the array was built)
_state = {"hist": {}, "ncopy": None}


def _history(req):
    """Committed tokens for one request; cached per rid, extended by the new tail only."""
    origin = req.origin_input_ids_unpadded
    out = req.output_ids  # array("q"), append-only by contract
    o_len, u_len = len(origin), len(out)
    entry = _state["hist"].get(req.rid)
    if entry is None or entry[1] != o_len or entry[2] > u_len:
        # first sight, origin changed, or outputs shrank (request reset): rebuild
        hist = np.concatenate(
            [np.array(origin, dtype=np.int64), np.array(out, dtype=np.int64)]
        )
    elif entry[2] < u_len:
        hist = np.concatenate([entry[0], np.array(out[entry[2]:u_len], dtype=np.int64)])
    else:
        hist = entry[0]
    _state["hist"][req.rid] = (hist, o_len, u_len)
    return hist


def _find_copy(hist, n, max_k):
    """Rightmost exact n-gram match of the history tail before the tail itself."""
    size = hist.size
    if size < n + 1:
        return None
    m = size - n  # tail = hist[m:]; candidates j in [0, m) each have a follower at j+n
    tail = hist[m:]
    eq = hist[:m] == tail[0]
    for t in range(1, n):
        eq &= hist[t:t + m] == tail[t]
        if not eq.any():
            return None
    hits = np.nonzero(eq)[0]  # n == 1 skips the loop, so eq may be all False here
    if hits.size == 0:
        return None
    j = int(hits[-1])
    k = min(max_k, size - (j + n))
    return hist[j + n:j + n + k]


def _clamp_rows(conf, ncopy):
    """conf [bs, gamma] or [bs]: clamp the copy rows' confidence to 1.0."""
    if conf is None or ncopy is None or conf.dim() not in (1, 2) or conf.shape[0] != len(ncopy):
        return
    two_d = conf.dim() == 2
    for i, k in enumerate(ncopy):
        if k:
            (conf[i, :k] if two_d else conf[i]).fill_(1.0)


def install_draft(draft_module):
    """sglang.srt.speculative.dspark_components.dspark_draft: prepend exact copy rows."""
    if not ENABLED:
        return
    cls = draft_module.DraftBlockProposer
    if getattr(cls, "_dsv41_copy_drafts", False):
        return
    cls._dsv41_copy_drafts = True
    orig = cls.propose

    def propose(self, *a, **kw):
        out = orig(self, *a, **kw)
        dt = out.draft_block.draft_tokens
        gamma = dt.shape[-1]
        reqs = list(kw["batch"].reqs)
        bs = int(kw["bs"])
        if len(reqs) == bs:
            limit = min(MAX_COPY, gamma)
            copies = [None] * bs
            for i, req in enumerate(reqs):
                copies[i] = _find_copy(_history(req), MATCH, limit)
            live = {req.rid for req in reqs}
            for rid in [r for r in _state["hist"] if r not in live]:
                del _state["hist"][rid]
            cl = out.draft_block.corrected_logits  # [bs, gamma, V] or None
            if cl is not None and (cl.dim() != 3 or cl.shape[:2] != (bs, gamma)):
                cl = None  # drifted layout: accept compares tokens, so skip the one-hot
            rows = torch.arange(limit, device=dt.device)
            ncopy = [0] * bs
            for i, cp in enumerate(copies):
                if cp is None:
                    continue
                k = int(cp.size)
                ncopy[i] = k
                toks = torch.as_tensor(cp, dtype=dt.dtype, device=dt.device)
                dt[i, :k] = toks
                if cl is not None:
                    cl[i, :k, :] = 0.0
                    cl[i, rows[:k], toks.long()] = 1e4
            _state["ncopy"] = ncopy
            _clamp_rows(out.confidence, ncopy)
        else:
            _state["ncopy"] = None  # rows misaligned with reqs: leave the step alone
        return out

    cls.propose = propose
    print(f"[copy_drafts] armed: {MATCH}-gram copies, up to {MAX_COPY} rows in front "
          "of the DFlash2 draft", flush=True)


def install_planner(planner_module):
    """sglang.srt.speculative.dspark_components.dspark_planner: clamp copy-row confidence."""
    if not ENABLED:
        return
    cls = planner_module.DSparkVerifyPlanner
    if getattr(cls, "_dsv41_copy_drafts", False):
        return
    cls._dsv41_copy_drafts = True
    orig = cls.compute_confidence_tensor

    def compute_confidence_tensor(self, *a, **kw):
        out = orig(self, *a, **kw)
        ncopy = _state["ncopy"]
        _state["ncopy"] = None  # one shot: the eager path calls this once per step
        _clamp_rows(out, ncopy)
        return out

    cls.compute_confidence_tensor = compute_confidence_tensor
    print("[copy_drafts] planner hook armed: copy-row confidence clamped to 1.0", flush=True)
