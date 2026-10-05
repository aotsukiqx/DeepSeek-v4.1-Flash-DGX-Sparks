"""DSV41_DUAL_GRAPH: capture a narrow (6-row) verify graph family at widened blocks.

W1 of docs/results/vcap-20260104/dual-graph-spec.md: capture-only, nothing replays the
family yet (selection is W2). At a DSPARK_BLOCK_SIZE > 5 boot, after the stock capture
pass of DecodeCudaGraphRunner._capture_one_stream (still inside the caller's capture
session), a second family is captured for bs <= DSV41_DUAL_GRAPH_MAX_BS (8): verify
rows DSV41_DUAL_GRAPH_ROWS (6) under the attention-variant key 'dsv41_narrow6'. The
b12x MoE capacities for both row families come from DSV41_MOE_B12X_NEXT_ROWS in
moe_b12x_next (rows 6,8,9 at gamma=8).
"""
import os

import torch

ENABLED = os.environ.get("DSV41_DUAL_GRAPH", "0").strip() not in ("0", "", "off", "false")
NARROW_ROWS = int(os.environ.get("DSV41_DUAL_GRAPH_ROWS", "6") or 6)
MAX_BS = int(os.environ.get("DSV41_DUAL_GRAPH_MAX_BS", "8") or 8)
LABEL = "dsv41_narrow6"
_NARROW_NOW = [False]


def install_verify_hook(module):
    """sglang.srt.speculative.dspark_components.dspark_verify: during a narrow capture the
    static epilogue views input_ids through self.stride (gamma+1 = 9); the narrow family
    needs 6. Guarded by the capture flag; replay keeps the stock stride until W2 routes
    it per family."""
    if not ENABLED:
        return
    cls = getattr(module, "DsparkVerifyEpilogue", None)
    if cls is None or getattr(cls, "_dsv41_dual_graph", False):
        return
    cls._dsv41_dual_graph = True

    def _wrap(name):
        orig = getattr(cls, name)

        def method(self, *a, **kw):
            if not _NARROW_NOW[0] or int(getattr(self, "stride", 0) or 0) <= NARROW_ROWS:
                return orig(self, *a, **kw)
            saved = self.stride
            self.stride = NARROW_ROWS
            try:
                return orig(self, *a, **kw)
            finally:
                self.stride = saved

        setattr(cls, name, method)

    for name in ("_static_epilogue", "begin_static_step"):
        if hasattr(cls, name):
            _wrap(name)
    print("[dual_graph] verify epilogue stride follows the narrow captures", flush=True)


def install(module):
    """sglang.srt.model_executor.runner.decode_cuda_graph_runner."""
    if not ENABLED:
        return
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
        def _backends():
            mr = self.model_runner
            seen = []
            for attr in ("decode_attn_backend", "attn_backend"):
                b = getattr(mr, attr, None)
                if b is not None and hasattr(b, "speculative_num_draft_tokens"):
                    seen.append(b)
            for b in getattr(mr, "decode_attn_backend_group", []) or []:
                if b is not None and hasattr(b, "speculative_num_draft_tokens"):
                    seen.append(b)
            return seen

        captured = []
        for bs in [b for b in self.capture_bs if b <= MAX_BS]:
            saved = self.captured_req_width
            self.captured_req_width = NARROW_ROWS
            saved_spec = [(b, b.speculative_num_draft_tokens) for b in _backends()]
            saved_meta = []
            for b, _ in saved_spec:
                b.speculative_num_draft_tokens = NARROW_ROWS
                store = getattr(b, "cuda_graph_metadata_of_bucket_and_bs", None)
                if store is not None:
                    saved_meta.append((b, store))
                    # fresh per-bs slots under the existing bucket keys: the narrow
                    # family stores its own metadata; the wide entries stay untouched
                    b.cuda_graph_metadata_of_bucket_and_bs = {
                        k: ({} if isinstance(v, dict) else v) for k, v in store.items()}
            try:
                set_variant(LABEL)
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
                for b, v in saved_spec:
                    b.speculative_num_draft_tokens = v
                for b, store in saved_meta:
                    b.cuda_graph_metadata_of_bucket_and_bs = store
                set_variant(None)
        if captured:
            print(f"[dual_graph] captured narrow verify family rows={NARROW_ROWS} "
                  f"for bs {captured}", flush=True)

    cls._capture_one_stream = _capture_one_stream
    print(f"[dual_graph] armed: narrow family rows={NARROW_ROWS} bs<={MAX_BS} "
          f"(capture-only; selection is W2)", flush=True)
