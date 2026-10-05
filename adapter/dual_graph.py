"""DSV41_DUAL_GRAPH: two verify-graph families at widened blocks (W1 capture + W2 selection).

At a DSPARK_BLOCK_SIZE > 5 boot the target runner captures, per bs <= MAX_BS, a narrow
family (rows DSV41_DUAL_GRAPH_ROWS, default 6) under the attention-variant key
'dsv41_narrow6' alongside the stock wide (9-row) graphs. Selection (W2): when a step's
batch-max live verify length fits the narrow width (verify_cap's live buffer, i.e. no
EXT extension this step), the verify batch is built at width 6 - verify_ids_2d
truncated, verify_window sliced, verify_num_draft_tokens swapped - and the narrow graph
replays (stride/epilogue buffers/metadata all derive from the batch's actual width).
Steps with an EXT extension keep the wide family's measured +12.3%.

Env:
  DSV41_DUAL_GRAPH=1        master gate (off in production)
  DSV41_DUAL_GRAPH_ROWS=6   narrow family width
  DSV41_DUAL_GRAPH_MAX_BS=8 narrow family max batch size
  DSV41_DUAL_GRAPH_FORCE=narrow|wide   override selection (A/B; narrow still needs the
                             family captured for that bs; works without verify_cap)
  DSV41_DUAL_GRAPH_ORACLE=N R1 self-check: the first N narrow-eligible steps run BOTH
                             families and compare the narrow replay's logits against the
                             wide reference (argmax agreement + max |delta|); PASS line
                             printed after N steps. Costs a second verify forward.

Selection needs DSV41_VERIFY_CAP=conf:T (the live length is the decision signal);
without it only FORCE=narrow does anything.
"""
import copy as _copy
import os

import torch

ENABLED = os.environ.get("DSV41_DUAL_GRAPH", "0").strip() not in ("0", "", "off", "false")
NARROW_ROWS = int(os.environ.get("DSV41_DUAL_GRAPH_ROWS", "6") or 6)
MAX_BS = int(os.environ.get("DSV41_DUAL_GRAPH_MAX_BS", "8") or 8)
LABEL = "dsv41_narrow6"
FORCE = os.environ.get("DSV41_DUAL_GRAPH_FORCE", "").strip().lower()
ORACLE_N = int(os.environ.get("DSV41_DUAL_GRAPH_ORACLE", "0") or 0)

_NARROW_NOW = [False]          # inside a narrow capture pass
_NARROW_BS = set()             # bs values with a captured narrow family (this process)
_STEP_WIDTH = [0]              # width chosen for the step in flight (0 = wide)
_ORACLE_LOG = []               # (bs, argmax_agreement, max|dlogit|)
_EPILOGS = []                  # epilogue instances seen (family state machine)


def _backends(mr):
    """Attention backends carrying verify metadata (the capture/replay store)."""
    seen = []
    for attr in ("decode_attn_backend", "attn_backend"):
        b = getattr(mr, attr, None)
        if b is not None and hasattr(b, "speculative_num_draft_tokens"):
            seen.append(b)
    for b in getattr(mr, "decode_attn_backend_group", []) or []:
        if b is not None and hasattr(b, "speculative_num_draft_tokens"):
            seen.append(b)
    return seen


def _fresh_live(bs):
    """Make verify_cap's live buffer current for this step (idempotent with its own call)."""
    try:
        import verify_cap
        conf = verify_cap._state.get("conf")
        if conf is not None:
            verify_cap.set_live_from_confidence(conf, bs)
    except Exception:
        pass


def _decide_width(bs):
    """6 when this step should replay the narrow family, else 0 (wide)."""
    if not _NARROW_BS:
        return 0
    if FORCE == "wide":
        return 0
    if bs not in _NARROW_BS:
        return 0
    if FORCE == "narrow":
        return NARROW_ROWS
    try:
        import verify_cap
        live = verify_cap._state.get("live")
        if live is None:
            return 0
        # live = rows incl. anchor, in {2..6} or 9 (EXT). Fits the narrow family iff
        # no request extended past 5 drafts this step.
        if int(live[:bs].max().item()) > NARROW_ROWS:
            return 0
        return NARROW_ROWS
    except Exception:
        return 0


def _apply_epilogue_width(ep, width):
    """Family state machine (W1g design): families clone from constructor state only;
    every width-coupled value (stride, gamma, stride-wide buffers) is applied
    idempotently. Kept referenced so captured graphs bind the right storage."""
    if int(getattr(ep, "_dsv41_w", -1) or -1) == width:
        return
    BUFS = ("out_tokens_buf", "cap_trim_lens_buf")
    fams = ep.__dict__.setdefault("_dsv41_fams", {})
    if "wide" not in fams:
        fams["wide"] = {"stride": ep.stride, "gamma": getattr(ep, "gamma", None),
                        "bufs": {n: getattr(ep, n) for n in BUFS if hasattr(ep, n)}}
        ep._dsv41_wide_s = ep.stride
    fam = fams.get(width)
    if fam is None:
        w = fams["wide"]
        fam = fams[width] = {
            "stride": width, "gamma": None if w["gamma"] is None else width - 1,
            "bufs": {n: (b.new_zeros((b.shape[0], width) + tuple(b.shape[2:]))
                         if b.dim() > 1 else b)
                     for n, b in w["bufs"].items()}}
    ep.stride = fam["stride"]
    if fam["gamma"] is not None:
        ep.gamma = fam["gamma"]
    for n, b in fam["bufs"].items():
        setattr(ep, n, b)
    ep._dsv41_w = width


def _apply_step_width(ep, w):
    """Apply the decided step width (0 -> the wide family)."""
    _apply_epilogue_width(ep, w if w == NARROW_ROWS else int(getattr(ep, "_dsv41_wide_s", 9)))


def _apply_step_width(ep, w):
    """Apply the decided step width (0 -> the wide family's OWN state, not a clone:
    the wide captured graphs bind the constructor buffers)."""
    if w == NARROW_ROWS:
        _apply_epilogue_width(ep, w)
        return
    fams = ep.__dict__.get("_dsv41_fams")
    wide = fams.get("wide") if fams else None
    if wide is not None:
        ep.stride = wide["stride"]
        if wide["gamma"] is not None:
            ep.gamma = wide["gamma"]
        for n, b in wide["bufs"].items():
            setattr(ep, n, b)
        ep._dsv41_w = wide["stride"]
    # else never narrowed: constructor state is already wide

def _oracle_due():
    return ORACLE_N > 0 and len(_ORACLE_LOG) < ORACLE_N

def _oracle_report():
    worst_a = min(a for _, a, _ in _ORACLE_LOG)
    worst_d = max(d for _, _, d in _ORACLE_LOG)
    bss = sorted({b for b, _, _ in _ORACLE_LOG})
    ok = worst_a >= 1.0 and worst_d < 0.05
    print(f"[dual_graph] R1 oracle {'PASS' if ok else 'FAIL'}: {len(_ORACLE_LOG)} steps, "
          f"bs {bss}, worst argmax_agree={worst_a:.4f}, worst max|dlogit|={worst_d:.4f}",
          flush=True)


def install_verify_hook(module):
    """dspark_verify: epilogue family state follows the step width (replay) and the
    batch width (capture), via the W1g eager-family design."""
    if not ENABLED:
        return
    cls = getattr(module, "DsparkVerifyEpilogue", None)
    if cls is None or getattr(cls, "_dsv41_dual_graph", False):
        return
    cls._dsv41_dual_graph = True
    orig_bss = cls.begin_static_step
    orig_ra = cls.read_accept
    orig_ep = cls._static_epilogue

    def begin_static_step(self, bs, armed):
        if _NARROW_NOW[0]:
            _apply_epilogue_width(self, NARROW_ROWS)
            _STEP_WIDTH[0] = NARROW_ROWS
            return orig_bss(self, bs, armed)
        _fresh_live(bs)
        w = _decide_width(bs)
        _apply_step_width(self, w)
        if w == NARROW_ROWS:
            self._static_step_state = None     # dedup key does not know the width
        _STEP_WIDTH[0] = w
        return orig_bss(self, bs, armed)

    def read_accept(self, bs):
        if _NARROW_NOW[0]:
            _apply_epilogue_width(self, NARROW_ROWS)
        elif _STEP_WIDTH[0] == NARROW_ROWS:
            _apply_epilogue_width(self, NARROW_ROWS)
        else:
            _apply_step_width(self, 0)
        return orig_ra(self, bs)

    def _static_epilogue(self, out, forward_batch):
        bs = int(forward_batch.batch_size)
        width = int(forward_batch.input_ids.shape[0]) // max(bs, 1)
        if width > 0:
            _apply_epilogue_width(self, width)
        return orig_ep(self, out, forward_batch)

    cls.begin_static_step = begin_static_step
    cls.read_accept = read_accept
    cls._static_epilogue = _static_epilogue
    print("[dual_graph] epilogue families follow the step/batch width (selection armed)",
          flush=True)


def install_executor_hook(module):
    """dspark_verify.TargetVerifyExecutor: build the verify batch at the chosen width.

    Installed BEFORE verify_cap's run_non_compact wrap (sitecustomize order), so this
    inner wrap sees the live buffer already set for the step; it re-derives the same
    decision begin_static_step made (same inputs, deterministic, rank-0-broadcast live).
    """
    if not ENABLED:
        return
    ex = getattr(module, "TargetVerifyExecutor", None)
    if ex is None or getattr(ex, "_dsv41_dual_graph", False):
        return
    ex._dsv41_dual_graph = True
    orig = ex.run_non_compact

    def _narrow_window(vw, bs, wide_stride):
        # VerifyWindow is a frozen msgspec.Struct: build a fresh instance, never mutate
        p2d = getattr(vw, "positions_2d", None)
        vcl = getattr(vw, "verify_cache_loc", None)
        v2 = getattr(vw, "verify_cache_loc_2d", None)
        return type(vw)(
            positions_2d=p2d[:, :NARROW_ROWS].contiguous()
            if p2d is not None and p2d.dim() == 2 else p2d,
            verify_cache_loc=vcl.view(bs, wide_stride)[:, :NARROW_ROWS].reshape(-1).contiguous()
            if vcl is not None and vcl.dim() == 1 and vcl.numel() == bs * wide_stride else vcl,
            verify_cache_loc_2d=v2[:, :NARROW_ROWS].contiguous()
            if v2 is not None and v2.dim() == 2 else v2,
        )

    def run_non_compact(self, *, batch, draft_input, verify_ids_2d, **kw):
        if _NARROW_NOW[0]:
            return orig(self, batch=batch, draft_input=draft_input,
                        verify_ids_2d=verify_ids_2d, **kw)
        bs = int(verify_ids_2d.shape[0])
        _fresh_live(bs)
        w = _decide_width(bs)
        _STEP_WIDTH[0] = w
        wide_stride = int(verify_ids_2d.shape[1])
        # The worker's step TAIL also reads verify_num_draft_tokens (logprob
        # chain_stride, GenerationBatchResult.speculative_num_draft_tokens, which
        # the scheduler uses to unflatten out_tokens): set it for the WHOLE step
        # and let the next step's entry set its own - never restore in between.
        self.verify_num_draft_tokens = NARROW_ROWS if w == NARROW_ROWS else wide_stride
        if w != NARROW_ROWS:
            return orig(self, batch=batch, draft_input=draft_input,
                        verify_ids_2d=verify_ids_2d, **kw)
        wide_w = wide_stride
        vw = kw.get("verify_window")

        def narrow_call():
            # verify_num_draft_tokens counts ROWS INCL. the anchor (= stride; stock
            # gamma=5 -> 6): the engram layer and DFlashVerifyInput both derive
            # per-request width from it, so the narrow family passes 6, not 5
            self.verify_num_draft_tokens = NARROW_ROWS
            # the eager pre-graph metadata path (prepare_for_verify ->
            # backend.init_forward_metadata) runs OUTSIDE the runner wraps and reads
            # the backend's own spec width + metadata store: swap both here too
            mr = getattr(getattr(self, "target_worker", None), "model_runner", None)
            backends = _backends(mr) if mr is not None else []
            saved_spec = [(b, b.speculative_num_draft_tokens) for b in backends]
            saved_stores = []
            for b in backends:
                b.speculative_num_draft_tokens = NARROW_ROWS
                ns = getattr(b, "_dsv41_narrow_meta_store", None)
                if ns is not None:
                    saved_stores.append((b, b.cuda_graph_metadata_of_bucket_and_bs))
                    b.cuda_graph_metadata_of_bucket_and_bs = ns
            kw2 = dict(kw)
            if vw is not None:
                kw2["verify_window"] = _narrow_window(vw, bs, wide_stride)
            try:
                return orig(self, batch=batch, draft_input=draft_input,
                            verify_ids_2d=verify_ids_2d[:, :NARROW_ROWS].contiguous(), **kw2)
            finally:
                # verify_num_draft_tokens deliberately NOT restored: the worker's
                # step tail still reads it; the next step's entry resets it
                for b, v in saved_spec:
                    b.speculative_num_draft_tokens = v
                for b, s in saved_stores:
                    b.cuda_graph_metadata_of_bucket_and_bs = s

        if _oracle_due():
            # R1: wide reference first (full stock path), then narrow; the comparison
            # uses forward logits only - unaffected by the accept-length caps the
            # narrow family state puts into the shared verify_lens buffer.
            _STEP_WIDTH[0] = 0
            self.verify_num_draft_tokens = wide_stride   # wide reference must run wide
            ep = getattr(self, "verify_epilogue", None)
            if ep is not None:
                # disarm the in-graph commit inject: the reference's folded accept
                # must not write KV / advance the injector a second time (the
                # narrow run owns this step's commit). Forward logits are unaffected.
                ep.begin_static_step(bs, False)
            res_w = orig(self, batch=batch, draft_input=draft_input,
                         verify_ids_2d=verify_ids_2d, **kw)
            _STEP_WIDTH[0] = NARROW_ROWS
            if ep is not None:
                ep.begin_static_step(bs, True)
            res_n = narrow_call()
            lw = res_w.logits_output.next_token_logits
            ln = res_n.logits_output.next_token_logits
            k = min(lw.shape[0], ln.shape[0])
            agree = float((lw[:k].argmax(-1) == ln[:k].argmax(-1)).float().mean().item())
            md = float((lw[:k].float() - ln[:k].float()).abs().max().item())
            _ORACLE_LOG.append((bs, agree, md))
            print(f"[dual_graph] oracle step bs={bs} argmax_agree={agree:.4f} "
                  f"max|dlogit|={md:.4f}", flush=True)
            if len(_ORACLE_LOG) >= ORACLE_N:
                _oracle_report()
            if bool(getattr(res_n, "can_run_cuda_graph", True)):
                return res_n
            print("[dual_graph] narrow graph unavailable at runtime; wide rerun", flush=True)
            _STEP_WIDTH[0] = 0
            return orig(self, batch=batch, draft_input=draft_input,
                         verify_ids_2d=verify_ids_2d, **kw)
        res = narrow_call()
        if not bool(getattr(res, "can_run_cuda_graph", True)):
            # The narrow graph could not run (unexpected fallback): redo the step wide
            # so downstream shapes stay on the stock path.
            print("[dual_graph] narrow graph unavailable at runtime; wide rerun", flush=True)
            _STEP_WIDTH[0] = 0
            return orig(self, batch=batch, draft_input=draft_input,
                         verify_ids_2d=verify_ids_2d, **kw)
        return res

    ex.run_non_compact = run_non_compact
    print("[dual_graph] verify batch builds at the selected width", flush=True)


def install(module):
    if int(os.environ.get("DSPARK_BLOCK_SIZE", "5") or 5) + 1 <= NARROW_ROWS:
        print("[dual_graph] not a widened boot: family is the stock one", flush=True)
        return
    cls = module.DecodeCudaGraphRunner
    if getattr(cls, "_dsv41_dual_graph", False):
        return
    cls._dsv41_dual_graph = True
    orig = cls._capture_one_stream
    set_variant = module._set_capture_attention_variant
    patch_model = module.torch_compile_decoration.patch_model

    def _capture_one_stream(self, stream_idx=None):
        orig(self, stream_idx)
        wide = int(getattr(self, "captured_req_width", 0) or 0)
        if wide <= NARROW_ROWS:
            return
        # The same runner class serves the DRAFT worker too (its width is the draft
        # query layout): only the TARGET verify runner gets the narrow family - the
        # drafter stays at its native width inside every graph
        if getattr(self.model_runner, "is_draft_worker", False):
            return
        backends = _backends(self.model_runner)
        saved_spec = [(b, b.speculative_num_draft_tokens) for b in backends]
        saved_meta = [(b, getattr(b, "cuda_graph_metadata_of_bucket_and_bs", None))
                      for b in backends]
        # ONE fresh store for the whole narrow pass: per-bs slots under the existing
        # bucket keys; kept on the backend for replay-time routing (W2)
        for b, store in saved_meta:
            if store is not None:
                b.cuda_graph_metadata_of_bucket_and_bs = {
                    k: ({} if isinstance(v, dict) else v) for k, v in store.items()}
        try:
            for b, _ in saved_spec:
                b.speculative_num_draft_tokens = NARROW_ROWS
            try:
                import verify_cap
                verify_cap._state["stride_ovr"] = NARROW_ROWS
            except Exception:
                pass
            captured = []
            for bs in [b for b in self.capture_bs if b <= MAX_BS]:
                saved = self.captured_req_width
                self.captured_req_width = NARROW_ROWS
                set_variant(LABEL)
                try:
                    with patch_model(
                        self.model_runner.model,
                        bs in self.compile_bs,
                        num_tokens=bs * NARROW_ROWS,
                        tp_group=self.model_runner.tp_group,
                    ) as forward:
                        _NARROW_NOW[0] = True
                        try:
                            self.capture_one_shape(bs, forward, stream_idx, None, LABEL)
                        finally:
                            _NARROW_NOW[0] = False
                    captured.append(bs)
                finally:
                    self.captured_req_width = saved
                    set_variant(None)
            _NARROW_BS.update(captured)
            # keep the narrow metadata for replay; restore the wide stores
            for b, store in saved_meta:
                if store is not None:
                    b._dsv41_narrow_meta_store = b.cuda_graph_metadata_of_bucket_and_bs
                    b.cuda_graph_metadata_of_bucket_and_bs = store
            if captured:
                print(f"[dual_graph] captured narrow verify family rows={NARROW_ROWS} "
                      f"for bs {captured} (selection armed)", flush=True)
        finally:
            for b, v in saved_spec:
                b.speculative_num_draft_tokens = v
            try:
                import verify_cap
                verify_cap._state.pop("stride_ovr", None)
            except Exception:
                pass

    cls._capture_one_stream = _capture_one_stream

    # ---- W2 replay routing ----
    orig_rav = cls._resolve_attention_variant

    def _replay_narrow(self, forward_batch):
        if _STEP_WIDTH[0] != NARROW_ROWS or not _NARROW_BS:
            return False
        if getattr(self.model_runner, "is_draft_worker", False):
            return False
        fb = forward_batch
        try:
            if not fb.forward_mode.is_target_verify():
                return False
            bs = int(fb.batch_size or 0)
            if bs > MAX_BS or bs not in _NARROW_BS:
                return False
            ids = fb.input_ids
            return ids is not None and int(ids.shape[0]) == bs * NARROW_ROWS
        except Exception:
            return False

    def _resolve_attention_variant(self, forward_batch):
        if _replay_narrow(self, forward_batch):
            return LABEL
        return orig_rav(self, forward_batch)

    cls._resolve_attention_variant = _resolve_attention_variant

    def _with_narrow(self, forward_batch, fn, *a, **kw):
        if not _replay_narrow(self, forward_batch):
            return fn(*a, **kw)
        saved_w = self.captured_req_width
        self.captured_req_width = NARROW_ROWS      # the uniform-width replay invariant
        saved_stores = []
        for b in _backends(self.model_runner):
            ns = getattr(b, "_dsv41_narrow_meta_store", None)
            if ns is not None:
                saved_stores.append((b, b.cuda_graph_metadata_of_bucket_and_bs))
                b.cuda_graph_metadata_of_bucket_and_bs = ns
        try:
            return fn(*a, **kw)
        finally:
            self.captured_req_width = saved_w
            for b, s in saved_stores:
                b.cuda_graph_metadata_of_bucket_and_bs = s

    def _bind(orig_m):
        # bind now: a closure over the loop variable would late-bind every wrapper
        # to the last method (execute), recursing execute->load_batch->execute
        def _m(self, *a, **kw):
            fb = None
            for x in a:
                if getattr(x, "input_ids", None) is not None and hasattr(x, "forward_mode"):
                    fb = x
                    break
            if fb is None:
                # can_run_graph/load_batch/execute all take forward_batch positionally
                # (verified on the serving branch); a missing batch cannot be routed
                return orig_m(self, *a, **kw)
            return _with_narrow(self, fb, orig_m, self, *a, **kw)

        return _m

    for name in ("can_run_graph", "load_batch", "execute"):
        if hasattr(cls, name):
            setattr(cls, name, _bind(getattr(cls, name)))
    print(f"[dual_graph] armed: narrow rows={NARROW_ROWS} bs<={MAX_BS} force={FORCE or '-'} "
          f"oracle={ORACLE_N}", flush=True)


def install_sampler_hook(module):
    """dspark_draft_sampler: during a narrow capture's warmup the folded sampler derives
    bs from query_token_num (8) and views hidden through gamma (8) while the batch
    carries 6 rows per request."""
    if not ENABLED:
        return
    cls = getattr(module, "DsparkDraftSampler", None)
    if cls is None or getattr(cls, "_dsv41_dual_graph", False):
        return
    cls._dsv41_dual_graph = True
    orig = cls.__call__

    def __call__(self, hidden_states, input_ids):
        if not _NARROW_NOW[0] or int(getattr(self, "query_token_num", 0) or 0) <= NARROW_ROWS:
            return orig(self, hidden_states, input_ids)
        saved = (self.query_token_num, getattr(self, "gamma", None))
        self.query_token_num = NARROW_ROWS
        if saved[1] is not None:
            self.gamma = NARROW_ROWS - 1
        try:
            return orig(self, hidden_states, input_ids)
        finally:
            self.query_token_num = saved[0]
            if saved[1] is not None:
                self.gamma = saved[1]

    cls.__call__ = __call__
    print("[dual_graph] folded sampler follows the narrow captures", flush=True)
