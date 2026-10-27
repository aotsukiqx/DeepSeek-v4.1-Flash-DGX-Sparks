#!/usr/bin/env python3
"""runtime/patch_dspark_sim_46404.py against synthetic stand-ins.

Reproduces only the anchored regions of the four files sglang#46404 touches
(static_accept_tail.py, dspark_sampling_accept_graph.py, dspark_verify.py,
dspark_worker_v2.py) plus stubs for the names they reference; NOT SGLang's
files. The sglang.* imports resolve via sys.modules stubs installed here.

Behavior checks after patching: the simulated-acceptance bonus is drawn at the
SIMULATED row (not the real num_correct row) in the torch reference tail, the
sampling-accept region stages the simulated bonus and passes target_logits=None
under simulation, build_accept_outs takes the simulated row's argmax when
target_logits is given, sample_simulated_bonus honors top-k/top-p, and the
worker routes sampled batches into simulation regardless of
SGLANG_SIMULATE_ACC_GREEDY while keeping the greedy fast-path conditions.
"""
import importlib.util
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / 'runtime' / 'patch_dspark_sim_46404.py'

FIX_TAIL = '''import torch
from typing import Optional


def _static_accept_tail_kernel(target_ptr, sim_ptr, seq_lens_ptr, T, BLOCK_T,
                               HAS_SIM, HAS_BONUS_IN, HAS_PARTS):
    row = 0
    offs = 0
    eq = offs == offs
    if eq:
        num_correct = tl.min(tl.where(eq, BLOCK_T, offs), 0)
    if HAS_PARTS:
        s = offs
    elif HAS_BONUS_IN:
        bonus = tl.load(target_ptr + row * T + num_correct).to(tl.int64)
    if HAS_SIM:
        correct = tl.load(sim_ptr).to(tl.int32)
    else:
        correct = num_correct.to(tl.int32)
    return correct + 1


def static_accept_tail_torch(candidates, target_predict, seq_lens,
                             simulated_correct_len=None, bonus_in=None, **kw):
    bs, T = candidates.shape
    gamma = T - 1
    target = target_predict.view(bs, T)
    matches = (candidates[:, 1:] == target[:, :gamma]).to(torch.int32)
    num_correct = matches.cumprod(dim=1).sum(dim=1).to(torch.int32)
    rows = torch.arange(bs, device=candidates.device)
    if bonus_in is not None:
        bonus = bonus_in[:bs].to(torch.int64)
    else:
        bonus = target[rows, num_correct.long()].to(torch.int64)
    if simulated_correct_len is not None:
        correct = simulated_correct_len.repeat(bs).to(torch.int32)
    else:
        correct = num_correct
    commit = correct + 1
    return {"correct": correct, "bonus": bonus, "commit": commit}
'''

FIX_GRAPH = '''"""Module docstring stand-in.
The caller routes only all-sampled steps without top-k / top-p, logit adjustments,
grammar or a simulated sampled bonus; everything else keeps the eager accept.
"""
import torch


def build_accept_outs(**kw):
    return kw


def sampling_accept_commit_lens(io, epilogue, bs, bonus, correct_len,
                                block_verification=None, rejection_noise=None):
    accept_outs = dict(
        correct_len=correct_len,
        block_verification=block_verification,
        rejection_noise=rejection_noise,
    )
    commit_lens = build_accept_outs(
        correct_len=correct_len,
        bonus=bonus,
        simulated_correct_len=epilogue.simulated_correct_len,
        target_logits=io.logits,
        out=epilogue.read_accept(bs),
    )
    return commit_lens, accept_outs
'''

FIX_VERIFY = '''import torch

from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments


class SpecTpSyncSite:
    DSPARK_ACCEPT_GREEDY = "greedy"
    DSPARK_ACCEPT_SAMPLE = "sample"
    DSPARK_ACCEPT_GRAPH = "graph"


def simulated_bonus_sample_graph(**kw):
    return torch.zeros(1, dtype=torch.int64)


def simulated_bonus_sample(**kw):
    raise NotImplementedError("torch path stub")


def sample_simulated_bonus(
    target_logits,
    correct_len,
    greedy_bonus,
    sampling_info,
    bs: int,
    verify_num_draft_tokens: int,
) -> torch.Tensor:
    """Temperature-sample every verify row and return the row at the (simulated)
    correct_len as the bonus token. Temperature only: callers exclude top-p/top-k/min-p."""
    bonus = simulated_bonus_sample(
        target_logits=target_logits,
        correct_len=correct_len,
        temperatures=sampling_info.temperatures,
        bs=bs,
        rows_per_request=verify_num_draft_tokens,
    )
    return bonus.to(greedy_bonus.dtype).view_as(greedy_bonus)


def build_accept_outs(
    correct_len,
    bonus,
    tp_sync,
    sync_site,
    simulated_correct_len=None,
    target_logits=None,
    simulate_bonus_sampling_info=None,
    verify_num_draft_tokens=6,
    **kw,
):
    bs = correct_len.shape[0]
    if simulated_correct_len is not None:
        # The accept runs as served; only its length is replaced.
        correct_len = simulated_correct_len.repeat(bs).to(correct_len.dtype)
        if simulate_bonus_sampling_info is not None:
            assert target_logits is not None
            bonus = sample_simulated_bonus(
                target_logits=target_logits,
                correct_len=correct_len,
                greedy_bonus=bonus,
                sampling_info=simulate_bonus_sampling_info,
                bs=bs,
                verify_num_draft_tokens=verify_num_draft_tokens,
            )
    return {"correct_len": correct_len, "bonus": bonus}


class TargetVerifyExecutor:
    def _rejection_noise(self):
        return None

    def pick_sync_site(self, accept_sampling_info, simulate_bonus_sampling_info):
        outs = dict(
            block_verification=None,
            rejection_noise=self._rejection_noise,
        )
        sync_site = (
            SpecTpSyncSite.DSPARK_ACCEPT_GREEDY
            if accept_sampling_info is None or accept_sampling_info.is_all_greedy
            else SpecTpSyncSite.DSPARK_ACCEPT_SAMPLE
        )
        return sync_site, outs


class DsparkVerifyEpilogue:
    def __init__(self, simulated_correct_len=None, simulate_fold=False):
        self.simulated_correct_len = simulated_correct_len
        self.simulate_fold = simulate_fold

    def _sim_bonus_args(self, logits, bs):
        return {}

    def tail_broadcast(self, logits, bs):
        bonus_in = None
        if True:
            bonus_in = self._sim_bonus(logits, bs)
        return bonus_in

    def _sim_bonus(self, logits, bs: int) -> torch.Tensor:
        return simulated_bonus_sample_graph(**self._sim_bonus_args(logits, bs))

    def accept_epilogue(self, logits, bs, bonus):
        if self.simulate_fold:
            bonus = self._sim_bonus(logits, bs).to(bonus.dtype).view_as(bonus)
        accept = build_accept_outs(
            correct_len=torch.zeros(bs, dtype=torch.int32),
            bonus=bonus,
            tp_sync=None,
            sync_site=SpecTpSyncSite.DSPARK_ACCEPT_GRAPH,
            simulated_correct_len=self.simulated_correct_len,
        )
        return accept
'''

FIX_WORKER = '''class Batch:
    def __init__(self, sampling_info, has_grammar=False):
        self.sampling_info = sampling_info
        self.has_grammar = has_grammar


def verify_logits_adjustments_are_noop(si):
    return True


class SamplingInfo:
    def __init__(self, is_all_greedy=False, is_any_greedy=False, top_k=False,
                 top_p=False, min_p=False, temperatures=(0.7,)):
        self.is_all_greedy = is_all_greedy
        self.is_any_greedy = is_any_greedy
        self.need_top_k_sampling = top_k
        self.need_top_p_sampling = top_p
        self.need_min_p_sampling = min_p
        self.temperatures = list(temperatures)


class Graph:
    def run(self, **kw):
        return "region-accept"


class DSparkWorkerV2:
    _simulate_acc_greedy = False
    _simulated_correct_len = None
    _verify_planner = None
    _sampling_accept_graph = None


    def route(self, batch):
        sampling_info = batch.sampling_info
        # Simulated acceptance overrides correct_len, so only the bonus token needs the
        # request's temperature: draft + accept run greedy, the bonus is still sampled.
        simulate_bonus_sampling_info = None
        if (
            self._simulate_acc_greedy
            and sampling_info is not None
            and not sampling_info.is_all_greedy
            and not batch.has_grammar
            and verify_logits_adjustments_are_noop(sampling_info)
            and not sampling_info.need_top_p_sampling
            and not sampling_info.need_top_k_sampling
            and not sampling_info.need_min_p_sampling
        ):
            simulate_bonus_sampling_info = sampling_info
            sampling_info = None
        return simulate_bonus_sampling_info, sampling_info

    def region_accept(self, epilogue, bs, batch, simulate_bonus_sampling_info,
                      sampling_info, logits_output):
        region_accept = None
        if (
            self._sampling_accept_graph is not None
            and self._sampling_graph_eligible(
                batch=batch,
                proposal_folded=True,
                sampling_info=sampling_info,
                simulate_bonus_sampling_info=simulate_bonus_sampling_info,
            )
        ):
            region_accept = self._sampling_accept_graph.run(
                bs=bs, target_logits=logits_output.next_token_logits
            )
        return region_accept




    def _sampling_graph_eligible(self, batch, proposal_folded, sampling_info,
                                 simulate_bonus_sampling_info):
        return (
            self._sampling_accept_graph is not None
            and proposal_folded
            and self._verify_planner.mode_value == "static"
            and sampling_info is not None
            and not sampling_info.is_any_greedy
            and not sampling_info.need_top_k_sampling
            and not sampling_info.need_top_p_sampling
            and verify_logits_adjustments_are_noop(sampling_info)
            and simulate_bonus_sampling_info is None
            and not batch.has_grammar
        )


def types_simple():
    class P:
        mode_value = "static"
    return P()
'''

FILES = [
    ('sglang/kernels/ops/speculative/dspark/static_accept_tail.py', FIX_TAIL),
    ('sglang/srt/speculative/dspark_components/dspark_sampling_accept_graph.py', FIX_GRAPH),
    ('sglang/srt/speculative/dspark_components/dspark_verify.py', FIX_VERIFY),
    ('sglang/srt/speculative/dspark_components/dspark_worker_v2.py', FIX_WORKER),
]


def install_sglang_stubs():
    def mod(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    for name in ("sglang", "sglang.srt", "sglang.srt.speculative"):
        mod(name)
    dflash = mod("sglang.srt.speculative.dflash_utils")
    dflash.apply_dflash_verify_logits_adjustments = lambda *a, **k: None

    def target_probs(*, next_token_logits, sampling_info, draft_token_num, bs, **kw):
        return torch.full((bs, 5), 0.2)

    dflash.build_speculative_verify_target_probs = target_probs


def run(base):
    return subprocess.run([sys.executable, '-I', str(PATCHER), str(base)],
                          capture_output=True, text=True)


def load(path, name):
    install_sglang_stubs()
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Epilogue:
    def __init__(self, simulated_correct_len):
        self.simulated_correct_len = simulated_correct_len
        self.begun = []

    def sim_bonus(self, logits, bs):
        return torch.arange(bs, dtype=torch.float32)

    def read_accept(self, bs):
        return None


class IO:
    def __init__(self):
        self.logits = torch.tensor([[1.0, 2.0, 3.0]])


def main():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        paths = {}
        for rel, text in FILES:
            p = base / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            paths[rel] = p

        # ---- stock behavior (simulated bonus at the WRONG row / no routing)
        tail = load(paths[FILES[0][0]], 'tail_stock')
        cands = torch.tensor([[7, 11, 12, 13, 14, 15]])
        tgt = torch.tensor([[11, 99, 33, 44, 55, 0]])  # draft[0] matches, draft[1] misses
        out = tail.static_accept_tail_torch(
            cands, tgt, torch.tensor([10]), simulated_correct_len=torch.tensor([3]))
        assert out["correct"].item() == 3 and out["commit"].item() == 4
        assert out["bonus"].item() == tgt[0, 1].item(), \
            'fixture: stock draws bonus at num_correct row'
        worker = load(paths[FILES[3][0]], 'worker_stock')
        w = worker.DSparkWorkerV2()
        w._simulated_correct_len = torch.tensor([2.0])
        si = worker.SamplingInfo(top_p=True)
        sim, kept = w.route(worker.Batch(si))
        assert sim is None, 'fixture: stock does not simulate sampled batches'

        # ---- patch all four files, then idempotency
        first = run(base)
        assert first.returncode == 0, first.stderr
        snapshot = {rel: p.read_text() for rel, p in paths.items()}
        second = run(base)
        assert second.returncode == 0 and second.stdout.count('already patched') == 4, \
            second.stdout
        assert {rel: p.read_text() for rel, p in paths.items()} == snapshot

        # ---- patched: torch tail draws the bonus at the simulated row
        tail2 = load(paths[FILES[0][0]], 'tail_patched')
        out2 = tail2.static_accept_tail_torch(
            cands, tgt, torch.tensor([10]), simulated_correct_len=torch.tensor([3]))
        assert out2["correct"].item() == 3
        assert out2["bonus"].item() == tgt[0, 3].item(), \
            'patched must draw the bonus at the simulated row'
        out3 = tail2.static_accept_tail_torch(cands, tgt, torch.tensor([10]))
        assert out3["bonus"].item() == tgt[0, 1].item(), 'no-sim path unchanged'

        # ---- patched: sampling accept region stages the simulated bonus
        graph2 = load(paths[FILES[1][0]], 'graph_patched')

        epi = Epilogue(torch.tensor([2]))
        commit, _ = graph2.sampling_accept_commit_lens(
            IO(), epi, 1, torch.zeros(1, dtype=torch.float32),
            torch.tensor([1], dtype=torch.int32))
        assert commit["target_logits"] is None, 'sim mode must not pass raw logits'
        assert torch.equal(commit["bonus"], torch.arange(1, dtype=torch.float32))
        epi0 = Epilogue(None)
        commit0, _ = graph2.sampling_accept_commit_lens(
            IO(), epi0, 1, torch.zeros(1, dtype=torch.float32),
            torch.tensor([1], dtype=torch.int32))
        assert commit0["target_logits"] is not None, 'non-sim keeps target_logits'

        # ---- patched: verify epilogue / build_accept_outs / top-k bonus
        verify2 = load(paths[FILES[2][0]], 'verify_patched')
        assert hasattr(verify2.DsparkVerifyEpilogue, "sim_bonus")
        logits = torch.tensor([[0.0, 9.0], [0.0, 9.0], [0.0, 1.0], [5.0, 0.0],
                               [0.0, 0.0], [0.0, 0.0]])
        got = verify2.build_accept_outs(
            correct_len=torch.tensor([1], dtype=torch.int32),
            bonus=torch.zeros(1, dtype=torch.int64),
            tp_sync=None, sync_site="g",
            simulated_correct_len=torch.tensor([3], dtype=torch.int32),
            target_logits=logits, verify_num_draft_tokens=6)
        assert got["bonus"].item() == 0, got["bonus"]  # row 3 argmax -> token 0
        got2 = verify2.build_accept_outs(
            correct_len=torch.tensor([1], dtype=torch.int32),
            bonus=torch.zeros(1, dtype=torch.int64),
            tp_sync=None, sync_site="g",
            simulated_correct_len=torch.tensor([1], dtype=torch.int32),
            target_logits=logits, verify_num_draft_tokens=6)
        assert got2["bonus"].item() == 1, got2["bonus"]  # row 1 argmax -> token 1
        ex = verify2.TargetVerifyExecutor()
        si_greedy = types.SimpleNamespace(is_all_greedy=True)
        site, _ = ex.pick_sync_site(si_greedy, si_greedy)
        assert site == "sample", 'simulated bonus with greedy accept still samples'
        site2, _ = ex.pick_sync_site(si_greedy, None)
        assert site2 == "greedy"
        si_sampled = types.SimpleNamespace(is_all_greedy=False)
        site3, _ = ex.pick_sync_site(si_sampled, None)
        assert site3 == "sample"
        class SI:
            temperatures = [0.7]
            need_top_k_sampling = True
            need_top_p_sampling = False
        torch.manual_seed(0)
        b = verify2.sample_simulated_bonus(
            target_logits=logits, correct_len=torch.tensor([1]),
            greedy_bonus=torch.zeros(1, dtype=torch.int64),
            sampling_info=SI(), bs=1, verify_num_draft_tokens=6)
        assert b.shape == (1,) and b.item() in (0, 1, 2, 3, 4)

        # ---- patched: worker routing + begin_sim_bonus staging + eligibility
        worker2 = load(paths[FILES[3][0]], 'worker_patched')
        w2 = worker2.DSparkWorkerV2()
        w2._simulated_correct_len = torch.tensor([2.0])
        si2 = worker2.SamplingInfo(top_p=True)
        sim2, kept2 = w2.route(worker2.Batch(si2))
        assert sim2 is si2 and kept2 is si2, \
            'simulated length + sampled batch must simulate (bonus from own temps)'
        w3 = worker2.DSparkWorkerV2()
        w3._simulated_correct_len = torch.tensor([2.0])
        w3._simulate_acc_greedy = True
        si3 = worker2.SamplingInfo()
        sim3, kept3 = w3.route(worker2.Batch(si3))
        assert sim3 is si3 and kept3 is None, 'greedy-eligible path still nulls sampling'
        w4 = worker2.DSparkWorkerV2()
        w4._sampling_accept_graph = worker2.Graph()
        w4._verify_planner = types.SimpleNamespace(mode_value="static")
        epi4 = types.SimpleNamespace(begun=[])
        epi4.begin_sim_bonus = lambda bs, temps: epi4.begun.append((bs, temps))
        si4 = worker2.SamplingInfo()  # sampled, no top-k/top-p: region-eligible
        ra = w4.region_accept(epi4, 1, worker2.Batch(si4), si4, si4,
                              types.SimpleNamespace(next_token_logits=None))
        assert ra == "region-accept" and len(epi4.begun) == 1, epi4.begun
        assert epi4.begun[0][1] == si4.temperatures[:1]
        assert w4._sampling_graph_eligible(
            batch=worker2.Batch(si4), proposal_folded=True, sampling_info=si4,
            simulate_bonus_sampling_info=si4), \
            'staged simulated bonus must keep the region eligible'
        assert not w4._sampling_graph_eligible(
            batch=worker2.Batch(si4), proposal_folded=True, sampling_info=si4,
            simulate_bonus_sampling_info=worker2.SamplingInfo()), \
            'a simulated bonus staged from a different sampling_info stays ineligible'

        # ---- anchor drift fails loudly on the first file
        drift = base / FILES[0][0]
        drift.write_text(FIX_TAIL.replace(
            'rows = torch.arange(bs, device=candidates.device)',
            'rows = torch.arange(bs, device=candidates.device)  # drifted'))
        bad = run(base)
        assert bad.returncode == 1 and 're-derive from sglang#46404' in bad.stderr, \
            bad.stderr

    print('test_patch_dspark_sim_46404 PASS')


if __name__ == '__main__':
    main()
