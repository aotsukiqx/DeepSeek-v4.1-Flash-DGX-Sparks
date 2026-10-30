#!/usr/bin/env python3
"""runtime/patch_int_ar_guard_46994.py against synthetic stand-ins.

Reproduces only the anchor regions of sglang main@7d996300's
kernels/ops/communication/all_reduce.py (IPCManager tail),
distributed/device_communicators/custom_all_reduce.py (+v2) (should_custom_ar),
and layers/dp_attention.py (_dp_gather_via_all_reduce local branch); NOT
SGLang's files. torch is a fake namespace of dtype tag strings.

Behavior after patching: an integer dtype never reaches the custom all-reduce
kernels (should_custom_ar False), and the DP local gather takes the in-place
NCCL path for integer rows at ANY tp_size (stock capped it at 8 GPUs and sent
wider layouts down the custom-AR path).
"""
import importlib.util
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / 'runtime' / 'patch_int_ar_guard_46994.py'

FIX_AR = '''import torch

torch.bfloat16 = 'bf16'
torch.float16 = 'f16'
torch.float32 = 'f32'


def cache_once(fn):
    return fn


class IPCManager:
    def init(self):
        self.__ffi_init__()


def get_all_reduce_module(dtype, world_size):
    return ('module', dtype, world_size)
'''

FIX_V1 = '''import torch

torch.float32 = 'f32'
torch.float16 = 'f16'
torch.bfloat16 = 'bf16'


class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    def __init__(self, disabled=False):
        self.disabled = disabled

    def should_custom_ar(self, inp):
        if self.disabled:
            return False
        if inp.numel() * inp.element_size() % 16 != 0:
            return False
        return True
'''

FIX_V2 = '''from sglang.kernels.ops.communication.all_reduce import (
    AllReduceAlgo,
)


class CustomAllReduceV2:
    def __init__(self, disabled=False):
        self.disabled = disabled

    def should_custom_ar(self, inp):
        if self.disabled:
            return False
        if inp.numel() * inp.element_size() % 16 != 0:
            return False
        return True
'''

FIX_DP = '''import torch


def world_dp_gather_enabled():
    return WORLD_GATHER


_TP = type('TP', (), {})()


def get_parallel():
    return _TP


_TP_AR_CALLS = []


def tensor_model_parallel_all_reduce(t):
    _TP_AR_CALLS.append(t)
    return t


class _Dist:
    class ReduceOp:
        SUM = 'SUM'

    group = type('G', (), {'WORLD': 'WORLD'})()

    @staticmethod
    def all_reduce(t, op, group):
        AR_CALLS.append((t, op, group))


torch.distributed = _Dist


def _dp_gather_via_all_reduce(global_tokens, local_tokens):
    # Input IDs are in int 32. We should use inplace_all_reduce for local case because of custom all reduce.
    if world_dp_gather_enabled():
        torch.distributed.all_reduce(
            global_tokens,
            op=torch.distributed.ReduceOp.SUM,
            group=torch.distributed.group.WORLD,
        )
    else:
        NUM_GPUS_PER_NODE = 8
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
'''


class Inp:
    def __init__(self, dtype, n=256):
        self.dtype = dtype
        self._n = n

    def numel(self):
        return self._n

    def element_size(self):
        return 1


class Tok:
    def __init__(self, fp):
        self.dtype = types.SimpleNamespace(is_floating_point=fp)

    def __setitem__(self, key, value):
        pass


def install_stubs(inplace_calls):
    def mod(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    for name in ('sglang', 'sglang.srt', 'sglang.srt.distributed',
                 'sglang.kernels', 'sglang.kernels.ops',
                 'sglang.kernels.ops.communication'):
        mod(name)
    ps = mod('sglang.srt.distributed.parallel_state')
    ps.inplace_all_reduce = lambda t, group_name: inplace_calls.append((t, group_name))
    ar = mod('sglang.kernels.ops.communication.all_reduce')
    ar.AllReduceAlgo = object
    ar.SUPPORTED_DTYPES = ('bf16', 'f16', 'f32')


def run(base):
    return subprocess.run([sys.executable, '-I', str(PATCHER), str(base)],
                          capture_output=True, text=True)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        files = {
            'sglang/kernels/ops/communication/all_reduce.py': FIX_AR,
            'sglang/srt/distributed/device_communicators/custom_all_reduce.py': FIX_V1,
            'sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py': FIX_V2,
            'sglang/srt/layers/dp_attention.py': FIX_DP,
        }
        for rel, text in files.items():
            p = base / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)

        inplace_calls = []
        install_stubs(inplace_calls)

        v1 = load(base / 'sglang/srt/distributed/device_communicators/custom_all_reduce.py', 'v1_stock')
        ar = Inp('i64')  # int stand-in, size % 16 == 0
        assert v1.CustomAllreduce().should_custom_ar(ar) is True, \
            'fixture: stock lets an int dtype through to the kernels'

        dp = load(base / 'sglang/srt/layers/dp_attention.py', 'dp_stock')
        dp.AR_CALLS = []
        dp.WORLD_GATHER = False
        dp._TP.tp_size = 16
        dp._TP.tp_group = types.SimpleNamespace(unique_name='tp')
        gt = Tok(False)  # int rows, wider than the stock 8-GPU cap
        dp._dp_gather_via_all_reduce(gt, Tok(False))
        assert len(dp._TP_AR_CALLS) == 1, \
            'fixture: stock sends >8-GPU int rows down the custom-AR path'

        first = run(base)
        assert first.returncode == 0, first.stderr
        text_after = {rel: (base / rel).read_text() for rel in files}
        second = run(base)
        assert second.returncode == 0 and 'already patched' in second.stdout, second.stdout
        assert all((base / rel).read_text() == t for rel, t in text_after.items())

        armod = load(base / 'sglang/kernels/ops/communication/all_reduce.py', 'ar_patched')
        assert armod.SUPPORTED_DTYPES == ('bf16', 'f16', 'f32'), armod.SUPPORTED_DTYPES

        v1p = load(base / 'sglang/srt/distributed/device_communicators/custom_all_reduce.py', 'v1_patched')
        assert v1p.CustomAllreduce().should_custom_ar(ar) is False, \
            'patched: int dtype refused before the size checks'
        assert v1p.CustomAllreduce().should_custom_ar(Inp('bf16')) is True
        assert v1p.CustomAllreduce(disabled=True).should_custom_ar(
            Inp('bf16')) is False

        v2p = load(base / 'sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py', 'v2_patched')
        assert v2p.CustomAllReduceV2().should_custom_ar(ar) is False
        assert v2p.CustomAllReduceV2().should_custom_ar(Inp('f32')) is True

        dpp = load(base / 'sglang/srt/layers/dp_attention.py', 'dp_patched')
        dpp.AR_CALLS = []
        dpp.WORLD_GATHER = False
        dpp._TP.tp_size = 16
        dpp._TP.tp_group = types.SimpleNamespace(unique_name='tp')
        inplace_calls.clear()
        dpp._dp_gather_via_all_reduce(Tok(False), Tok(False))
        assert len(inplace_calls) == 1 and not dpp._TP_AR_CALLS, \
            'patched: int rows take the in-place NCCL path at any tp_size'
        dpp._TP_AR_CALLS.clear()
        inplace_calls.clear()
        dpp._dp_gather_via_all_reduce(Tok(True), Tok(True))
        assert len(dpp._TP_AR_CALLS) == 1 and not inplace_calls, \
            'patched: float rows still go to the TP all-reduce'
        dpp.WORLD_GATHER = True
        dpp._dp_gather_via_all_reduce(Tok(True), Tok(True))
        assert len(dpp.AR_CALLS) == 1, 'patched: world gather branch intact'

        # anchor drift fails loudly (comment line is an anchor)
        (base / 'sglang/srt/layers/dp_attention.py').write_text(
            FIX_DP.replace(
                '# Input IDs are in int 32. We should use inplace_all_reduce for local case because of custom all reduce.',
                '# drifted'))
        drifted = run(base)
        assert drifted.returncode == 1 and 're-derive from sglang#46994' in drifted.stderr, \
            drifted.stderr

    print('test_patch_int_ar_guard_46994 PASS')


if __name__ == '__main__':
    main()
