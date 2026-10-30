#!/usr/bin/env python3
"""Backport sglang#46994 (c2401692) onto sglang main@7d996300.

[Fix] custom-all-reduce: Keep integer all-reduces off the custom kernels.

The custom all-reduce kernels only sum fp32/fp16/bf16 (FloatTrait in
distributed/communicator.cuh); an integer tensor that reaches
should_custom_ar would crash the kernels. Guard both CustomAllreduce and
CustomAllReduceV2 by dtype (SUPPORTED_DTYPES), and take integer rows in
_dp_gather_via_all_reduce down the in-place NCCL path instead of the TP
custom AR. DEFENSIVE here (no int AR observed hitting custom AR on this
fleet; attn_dp=1 keeps the dp_attention hunk idle) -- belt for the
Engram-hash-id gather shapes.

Applies clean with `git apply` on the pristine 7d996300 files (verified
2026-10-30). PIN-MIGRATION NOTE: absorbed at any pin >= c2401692.

Idempotent. Exits non-zero on anchor drift (these scripts are the
build's drift probes). Optional argv[1] overrides the sglang python
tree root (default: the in-image path).
"""
import pathlib
import sys

DEFAULT_BASE = "/sgl-workspace/sglang/python"
PR = '46994'

FILES = [
    (
        '''sglang/kernels/ops/communication/all_reduce.py''',
        '''# The dtypes the kernels reduce: those `FloatTrait` covers in distributed/communicator.cuh.
''',
        [
            ('''        self.__ffi_init__()


''',
             '''        self.__ffi_init__()


# The dtypes the kernels reduce: those `FloatTrait` covers in distributed/communicator.cuh.
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


'''),
        ],
    ),
    (
        '''sglang/srt/distributed/device_communicators/custom_all_reduce.py''',
        '''    # The kernels throw on other dtypes (csrc/allreduce/custom_all_reduce.cu, .hip).
''',
        [
            ('''    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]
''',
             '''    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]
    # The kernels throw on other dtypes (csrc/allreduce/custom_all_reduce.cu, .hip).
    _SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
'''),
            ('''        if self.disabled:
            return False
''',
             '''        if self.disabled:
            return False
        if inp.dtype not in self._SUPPORTED_DTYPES:
            return False
'''),
        ],
    ),
    (
        '''sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py''',
        '''    SUPPORTED_DTYPES,
''',
        [
            ('''from sglang.kernels.ops.communication.all_reduce import (
''',
             '''from sglang.kernels.ops.communication.all_reduce import (
    SUPPORTED_DTYPES,
'''),
            ('''        if self.disabled:
            return False
''',
             '''        if self.disabled:
            return False
        if inp.dtype not in SUPPORTED_DTYPES:
            return False
'''),
        ],
    ),
    (
        '''sglang/srt/layers/dp_attention.py''',
        '''    elif not local_tokens.dtype.is_floating_point:
''',
        [
            ('''    # Input IDs are in int 32. We should use inplace_all_reduce for local case because of custom all reduce.
''',
             ''''''),
            ('''            op=torch.distributed.ReduceOp.SUM,
            group=torch.distributed.group.WORLD,
        )
''',
             '''            op=torch.distributed.ReduceOp.SUM,
            group=torch.distributed.group.WORLD,
        )
    elif not local_tokens.dtype.is_floating_point:
        # Integer rows (input ids, Engram hash ids) take the in-place NCCL path:
        # the custom all-reduce kernels sum floating-point dtypes only.
        from sglang.srt.distributed.parallel_state import inplace_all_reduce

        inplace_all_reduce(
            global_tokens, group_name=get_parallel().tp_group.unique_name
        )
'''),
            ('''        NUM_GPUS_PER_NODE = 8
        if (
            not local_tokens.dtype.is_floating_point
            and get_parallel().tp_size <= NUM_GPUS_PER_NODE
        ):
            from sglang.srt.distributed.parallel_state import inplace_all_reduce

            inplace_all_reduce(
                global_tokens, group_name=get_parallel().tp_group.unique_name
            )

        else:
            global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)
''',
             '''        global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)
'''),
        ],
    ),
]

def main() -> int:
    base = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE)
    rc = 0
    for rel, sentinel, patches in FILES:
        target = base / rel
        if not target.is_file():
            print("ERROR: target not found: %s (base image changed?)" % target,
                  file=sys.stderr)
            return 1
        src = target.read_text()
        if sentinel in src:
            print("already patched: %s" % target)
            continue
        for old, _ in patches:
            n = src.count(old)
            if n != 1:
                print("ERROR: %s: expected exactly 1 anchor match, found %d -- "
                      "base changed, re-derive from sglang#%s" % (target, n, PR),
                      file=sys.stderr)
                rc = 1
        if rc:
            return rc
        for old, new in patches:
            src = src.replace(old, new)
        if sentinel not in src or any(new not in src for _, new in patches):
            print("ERROR: post-patch verification failed: %s" % target,
                  file=sys.stderr)
            return 1
        target.write_text(src)
        print("patched %s (%d hunk%s)" % (target, len(patches),
              "" if len(patches) == 1 else "s"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
