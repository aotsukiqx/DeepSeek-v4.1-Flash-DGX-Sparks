"""CPU tests for adapter/copy_drafts.py (DSV41_COPY_DRAFTS, default off).

Checks, numbered:
 1. an earlier exact MATCH-gram match overwrites the FRONT of draft_tokens in place
    (same tensor object), requests without a match keep every row;
 2. the rightmost occurrence wins, and the copy length is min(MAX_COPY, gamma,
    followers available) including overlapping occurrences;
 3. sampled copy rows' corrected_logits become exactly one-hot (1e4 at the copy
    token, 0 elsewhere) and all other rows keep their bits; a None corrected_logits
    (greedy-only paths) is skipped without touching the rewrite;
 4. confidence: folded proposal.confidence clamps the copy rows' leading columns to
    1.0 only; the planner hook clamps the eager compute_confidence_tensor output the
    same way, consumes its ncopy in one shot, and leaves mismatched shapes alone;
 5. history cache: rid-keyed, extended by only the new output_ids tail, rebuilt when
    outputs shrink (request reset), and rids that left the batch are dropped;
 6. misaligned rows (len(reqs) != bs) leave the whole step untouched and clear ncopy;
 7. install refuses drifted modules (AttributeError) and a second install is a no-op;
 8. with the env unset, install_draft/install_planner install nothing (subprocess).

Run: PYTHONPATH=adapter python tests/test_copy_drafts.py
"""
import os

os.environ["DSV41_COPY_DRAFTS"] = "1"
os.environ["DSV41_COPY_MATCH"] = "3"

import array
import subprocess
import sys
from types import SimpleNamespace

import torch

import copy_drafts as cd  # noqa: E402

GAMMA, VOCAB = 5, 128
# deterministic references for the fake proposer's corrected_logits / confidence
REF_CL = torch.randn(8, GAMMA, VOCAB, generator=torch.Generator().manual_seed(1234))
REF_CONF = 0.5 * torch.rand(8, GAMMA, generator=torch.Generator().manual_seed(1234))


def make_draft_module(with_logits=True, with_conf=True, gamma=GAMMA, dt_ref=None):
    class P:
        def propose(self, *, batch, bs, device=None, target_model=None,
                    sampling_info=None, **kw):
            dt = 1000 + torch.arange(1, bs * gamma + 1, dtype=torch.int64).view(bs, gamma)
            cl = REF_CL[:bs, :gamma].clone() if with_logits else None
            conf = REF_CONF[:bs, :gamma].clone() if with_conf else None
            if dt_ref is not None:
                dt_ref.append(dt)
            return SimpleNamespace(
                draft_block=SimpleNamespace(draft_tokens=dt, corrected_logits=cl),
                confidence=conf)

    return SimpleNamespace(DraftBlockProposer=P)


def make_planner_module(ref):
    class Planner:
        def compute_confidence_tensor(self, **kw):
            return ref.clone()

    return SimpleNamespace(DSparkVerifyPlanner=Planner), Planner


def req(rid, origin, out):
    return SimpleNamespace(rid=rid, origin_input_ids_unpadded=list(origin),
                           output_ids=array.array("q", out))


def run_step(p, reqs, bs=None):
    return p.propose(batch=SimpleNamespace(reqs=reqs), bs=len(reqs) if bs is None else bs,
                     draft_input=None, verify_window=None, device="cpu",
                     target_model=None, sampling_info=None)


def reset_state():
    cd._state["hist"].clear()
    cd._state["ncopy"] = None


def check_rewrite():
    # 1 + 2: front rows replaced in place, rightmost match, unmatched rows kept
    reset_state()
    hist_a = [10, 11, 12, 13, 10, 11, 12, 50, 51, 52, 60, 99, 10, 11, 12]
    hist_b = [1, 2, 3, 4, 5, 6, 7, 8]
    dt_ref = []
    mod = make_draft_module(dt_ref=dt_ref)
    cd.install_draft(mod)
    p = mod.DraftBlockProposer()
    out = run_step(p, [req("a", hist_a, []), req("b", hist_b, [])])
    dt = out.draft_block.draft_tokens
    assert dt is dt_ref[0], "draft_tokens must be rewritten in place"
    assert dt[0].tolist() == [50, 51, 52, 60, 99], dt[0].tolist()
    assert dt[1].tolist() == (1000 + torch.arange(6, 11)).tolist(), "no match: rows untouched"
    assert cd._state["ncopy"] == [5, 0]
    # gamma clamp
    reset_state()
    mod = make_draft_module(gamma=2)
    cd.install_draft(mod)
    out = mod.DraftBlockProposer().propose(
        batch=SimpleNamespace(reqs=[req("a", hist_a, [])]), bs=1, draft_input=None,
        verify_window=None, device="cpu", target_model=None, sampling_info=None)
    assert out.draft_block.draft_tokens[0].tolist() == [50, 51]
    assert cd._state["ncopy"] == [2]
    hist_e = [5, 5, 5, 1, 2, 3, 9, 1, 2, 3]  # followers end the history: k = 4
    reset_state()
    mod = make_draft_module()
    cd.install_draft(mod)
    out = run_step(mod.DraftBlockProposer(), [req("e", hist_e, [])])
    assert out.draft_block.draft_tokens[0].tolist()[:4] == [9, 1, 2, 3]
    assert cd._state["ncopy"] == [4]
    hist_f = [4, 4, 4, 4, 4]  # occurrence overlaps the tail: k = 1
    reset_state()
    mod = make_draft_module()
    cd.install_draft(mod)
    out = run_step(mod.DraftBlockProposer(), [req("f", hist_f, [])])
    assert out.draft_block.draft_tokens[0].tolist()[:1] == [4]
    assert cd._state["ncopy"] == [1]


def check_onehot():
    # 3: one-hot corrected_logits on copy rows only; None skipped
    reset_state()
    hist_e = [5, 5, 5, 1, 2, 3, 9, 1, 2, 3]  # k = 4: row 4 stays a DFlash2 row
    mod = make_draft_module()
    cd.install_draft(mod)
    out = run_step(mod.DraftBlockProposer(), [req("e", hist_e, [])])
    cl = out.draft_block.corrected_logits
    assert cl is not None
    copy_row = cl[0, 0]
    assert copy_row.tolist() == [1e4 if v == 9 else 0.0 for v in range(VOCAB)]
    assert cl[0, 1, 1] == 1e4 and cl[0, 1, 2] == 0.0 and cl[0, 3, 3] == 1e4
    assert torch.equal(cl[0, 4:], REF_CL[0, 4:]), "rows past the copy keep their bits"
    reset_state()
    mod = make_draft_module(with_logits=False)  # greedy-only path
    cd.install_draft(mod)
    out = run_step(mod.DraftBlockProposer(), [req("e", hist_e, [])])
    assert out.draft_block.corrected_logits is None
    assert out.draft_block.draft_tokens[0].tolist()[:4] == [9, 1, 2, 3]


def check_confidence():
    # 4: folded confidence clamp + planner hook (one shot, shape guard, 1D)
    reset_state()
    hist_a = [10, 11, 12, 13, 10, 11, 12, 50, 51, 52, 60, 99, 10, 11, 12]
    hist_b = [1, 2, 3, 4, 5, 6, 7, 8]
    mod = make_draft_module()
    cd.install_draft(mod)
    out = run_step(mod.DraftBlockProposer(), [req("a", hist_a, []), req("b", hist_b, [])])
    conf = out.confidence
    assert conf is not None
    assert conf[0].tolist() == [1.0] * GAMMA, "full-copy row clamps every column"
    assert torch.equal(conf[1], REF_CONF[1]), "no-match request keeps its confidence"
    reset_state()
    cd._state["ncopy"] = [2, 0]
    ref = REF_CONF[:2].clone()
    mod, planner = make_planner_module(ref)
    cd.install_planner(mod)
    got = planner().compute_confidence_tensor(draft_hidden=None, anchor_tokens=None,
                                              draft_tokens=None, confidence_tap=None)
    assert got[0, :2].tolist() == [1.0, 1.0]
    assert got[0, 2:].tolist() == ref[0, 2:].tolist()
    assert torch.equal(got[1], ref[1])
    assert cd._state["ncopy"] is None, "planner consumes ncopy in one shot"
    ref2 = REF_CONF[:2].clone()
    got2 = planner().compute_confidence_tensor()
    assert torch.equal(got2, ref2), "second call without a propose is a no-op"
    cd._state["ncopy"] = [1]
    got3 = planner().compute_confidence_tensor()  # shape [2] != len(ncopy)=1 -> untouched
    assert torch.equal(got3, ref2)
    cd._state["ncopy"] = [1, 0]  # 1D defensive branch
    one_d = torch.tensor([0.25, 0.25])
    mod1, planner1 = make_planner_module(one_d)
    cd.install_planner(mod1)
    got4 = planner1().compute_confidence_tensor()
    assert got4.tolist() == [1.0, 0.25]
    reset_state()


def check_history():
    # 5: incremental tail, reset rebuild, lazy eviction
    reset_state()
    mod = make_draft_module()
    cd.install_draft(mod)
    p = mod.DraftBlockProposer()
    r = req("h", [10, 11, 12, 13], [10, 11, 12])
    out = run_step(p, [r])
    assert out.draft_block.draft_tokens[0].tolist()[:4] == [13, 10, 11, 12]
    r.output_ids.append(50)  # append-only growth: only the new tail is concat'd
    run_step(p, [r])
    assert cd._state["hist"]["h"][0].tolist() == [10, 11, 12, 13, 10, 11, 12, 50]
    for t in (13, 10, 11, 12):
        r.output_ids.append(t)
    out = run_step(p, [r])
    assert out.draft_block.draft_tokens[0].tolist() == [50, 13, 10, 11, 12]
    r2 = req("h", [10, 11, 12, 13], [])  # same rid, outputs gone: rebuilt, no stale tail
    out = run_step(p, [r2])
    assert cd._state["hist"]["h"][0].tolist() == [10, 11, 12, 13]
    assert out.draft_block.draft_tokens[0].tolist() == (1000 + torch.arange(1, 6)).tolist()
    run_step(p, [req("z", [1, 2, 3, 4], [])])
    assert "h" not in cd._state["hist"], "rids that left the batch are dropped"
    reset_state()


def check_misalign_and_install():
    # 6 + 7: len(reqs) != bs leaves the step alone; drift and double install
    reset_state()
    dt_ref = []
    mod = make_draft_module(dt_ref=dt_ref)
    cd.install_draft(mod)
    sentinel = (1000 + torch.arange(1, GAMMA + 1)).tolist()
    out = run_step(mod.DraftBlockProposer(),
                   [req("a", [10, 11, 12, 13, 10, 11, 12], [])], bs=2)
    assert out.draft_block.draft_tokens[0].tolist() == sentinel
    assert cd._state["ncopy"] is None
    wrapped = mod.DraftBlockProposer.propose
    cd.install_draft(mod)
    assert mod.DraftBlockProposer.propose is wrapped, "second install is a no-op"
    for fn in (cd.install_draft, cd.install_planner):
        try:
            fn(SimpleNamespace())
        except AttributeError:
            continue
        raise AssertionError(f"{fn.__name__} must refuse drifted modules")
    reset_state()


def check_off_subprocess():
    # 8: env unset -> installs nothing
    code = (
        "import sys, types\n"
        "sys.path.insert(0, 'adapter')\n"
        "import copy_drafts as cd\n"
        "assert not cd.ENABLED and cd.MATCH == 8 and cd.MAX_COPY == 5\n"
        "class P:\n"
        "    def propose(self, **kw):\n"
        "        return None\n"
        "class Planner:\n"
        "    def compute_confidence_tensor(self, **kw):\n"
        "        return None\n"
        "orig_p, orig_c = P.propose, Planner.compute_confidence_tensor\n"
        "m = types.SimpleNamespace(DraftBlockProposer=P)\n"
        "cd.install_draft(m)\n"
        "m2 = types.SimpleNamespace(DSparkVerifyPlanner=Planner)\n"
        "cd.install_planner(m2)\n"
        "assert P.propose is orig_p and not hasattr(P, '_dsv41_copy_drafts')\n"
        "assert Planner.compute_confidence_tensor is orig_c\n"
        "print('off-ok')\n"
    )
    env = {k: v for k, v in os.environ.items()
           if k not in ("DSV41_COPY_DRAFTS", "DSV41_COPY_MATCH", "DSV41_COPY_MAX")}
    res = subprocess.run([sys.executable, "-c", code], env=env, cwd=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), capture_output=True, text=True)
    assert res.returncode == 0 and "off-ok" in res.stdout, (res.returncode, res.stdout, res.stderr)


def main():
    assert cd.ENABLED and cd.MATCH == 3 and cd.MAX_COPY == 5
    check_rewrite()
    check_onehot()
    check_confidence()
    check_history()
    check_misalign_and_install()
    check_off_subprocess()
    print("test_copy_drafts: ok")


if __name__ == "__main__":
    main()
