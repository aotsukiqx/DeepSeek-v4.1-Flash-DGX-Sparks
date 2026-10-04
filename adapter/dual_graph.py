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
        captured = []
        for bs in [b for b in self.capture_bs if b <= MAX_BS]:
            saved = self.captured_req_width
            self.captured_req_width = NARROW_ROWS
            try:
                set_variant(LABEL)
                with patch_model(
                    self.model_runner.model,
                    bs in self.compile_bs,
                    num_tokens=bs * NARROW_ROWS,
                    tp_group=self.model_runner.tp_group,
                ) as forward:
                    self.capture_one_shape(bs, forward, stream_idx, None, LABEL)
                captured.append(bs)
            finally:
                self.captured_req_width = saved
                set_variant(None)
        if captured:
            print(f"[dual_graph] captured narrow verify family rows={NARROW_ROWS} "
                  f"for bs {captured}", flush=True)

    cls._capture_one_stream = _capture_one_stream
    print(f"[dual_graph] armed: narrow family rows={NARROW_ROWS} bs<={MAX_BS} "
          f"(capture-only; selection is W2)", flush=True)
