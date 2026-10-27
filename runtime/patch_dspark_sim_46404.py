#!/usr/bin/env python3
"""Backport sglang#46404 (24d55621) onto sglang main@7d996300.

[Fix] dspark: Draw the simulated-acceptance bonus at the simulated row.

Under ``SGLANG_SIMULATE_ACC_LEN`` the bonus token was drawn at the REAL
num_correct row while correct_len was overridden by the simulated length — the
committed sequence then contained the wrong bonus token, and any accept-len
calibration run with sampled (non-greedy) requests was subtly wrong. The fix
draws the bonus at the simulated row everywhere (static_accept_tail kernel +
torch reference, build_accept_outs' target_logits branch,
sample_simulated_bonus via build_speculative_verify_target_probs) and widens
the sampling-accept region to draw the simulated bonus with the request's own
temperatures (begin_sim_bonus staging).

Production impact: NONE unless SGLANG_SIMULATE_ACC_LEN is set (calibration
knob; see .env.tp4.example). All touched symbols exist at 7d996300
(begin_sim_bonus, build_speculative_verify_target_probs — verified
2026-10-27). Applies clean with ``git apply`` on the pristine files.

PIN-MIGRATION NOTE: dspark_worker_v2/dspark_verify sit on the spec rework
line (#46290/#46330 restructure placement); re-derive or drop.

Idempotent. Exits non-zero on anchor drift (these scripts are the
build's drift probes). Optional argv[1] overrides the sglang python
tree root (default: the in-image path).
"""
import pathlib
import sys

DEFAULT_BASE = "/sgl-workspace/sglang/python"
PR = "46404"

FILES = [
    (
        '''sglang/kernels/ops/speculative/dspark/static_accept_tail.py''',
        '''        bonus = tl.load(target_ptr + row * T + correct).to(tl.int64)''',
        [
            ('''        num_correct = tl.min(tl.where(eq, BLOCK_T, offs), 0)
''',
             '''        num_correct = tl.min(tl.where(eq, BLOCK_T, offs), 0)
    if HAS_SIM:
        correct = tl.load(sim_ptr).to(tl.int32)
    else:
        correct = num_correct.to(tl.int32)
'''),
            ('''        bonus = tl.load(target_ptr + row * T + num_correct).to(tl.int64)
    if HAS_SIM:
        correct = tl.load(sim_ptr).to(tl.int32)
    else:
        correct = num_correct.to(tl.int32)
''',
             '''        bonus = tl.load(target_ptr + row * T + correct).to(tl.int64)
'''),
            ('''    rows = torch.arange(bs, device=candidates.device)
''',
             '''    rows = torch.arange(bs, device=candidates.device)
    if simulated_correct_len is not None:
        correct = simulated_correct_len.repeat(bs).to(torch.int32)
    else:
        correct = num_correct
'''),
            ('''        bonus = target[rows, num_correct.long()].to(torch.int64)
    if simulated_correct_len is not None:
        correct = simulated_correct_len.repeat(bs).to(torch.int32)
    else:
        correct = num_correct
''',
             '''        bonus = target[rows, correct.long()].to(torch.int64)
'''),
        ],
    ),
    (
        '''sglang/srt/speculative/dspark_components/dspark_sampling_accept_graph.py''',
        '''The caller routes only all-sampled steps without top-k / top-p, logit adjustments or''',
        [
            ('''The caller routes only all-sampled steps without top-k / top-p, logit adjustments,
grammar or a simulated sampled bonus; everything else keeps the eager accept.
''',
             '''The caller routes only all-sampled steps without top-k / top-p, logit adjustments or
grammar; everything else keeps the eager accept. Under SGLANG_SIMULATE_ACC_LEN the region
draws the bonus at the simulated row from the buffers the caller stages with
begin_sim_bonus.
'''),
            ('''        rejection_noise=rejection_noise,
    )
''',
             '''        rejection_noise=rejection_noise,
    )
    bonus_logits = io.logits
    if epilogue.simulated_correct_len is not None:
        bonus = epilogue.sim_bonus(io.logits, bs).to(bonus.dtype).view_as(bonus)
        bonus_logits = None
'''),
            ('''        simulated_correct_len=epilogue.simulated_correct_len,
        target_logits=io.logits,
''',
             '''        simulated_correct_len=epilogue.simulated_correct_len,
        target_logits=bonus_logits,
'''),
        ],
    ),
    (
        '''sglang/srt/speculative/dspark_components/dspark_verify.py''',
        '''from sglang.srt.speculative.dflash_utils import (''',
        [
            ('''from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments
''',
             '''from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    build_speculative_verify_target_probs,
)
'''),
            ('''            rejection_noise=self._rejection_noise,
        )
''',
             '''            rejection_noise=self._rejection_noise,
        )
        greedy_bonus = simulate_bonus_sampling_info is None and (
            accept_sampling_info is None or accept_sampling_info.is_all_greedy
        )
'''),
            ('''            if accept_sampling_info is None or accept_sampling_info.is_all_greedy
''',
             '''            if greedy_bonus
'''),
            ('''            bonus_in = self._sim_bonus(logits, bs)
''',
             '''            bonus_in = self.sim_bonus(logits, bs)
'''),
            ('''    def _sim_bonus(self, logits, bs: int) -> torch.Tensor:
''',
             '''    def sim_bonus(self, logits, bs: int) -> torch.Tensor:
        """The simulated row's bonus from the buffers begin_sim_bonus staged; graph-safe."""
'''),
            ('''            bonus = self._sim_bonus(logits, bs).to(bonus.dtype).view_as(bonus)
''',
             '''            bonus = self.sim_bonus(logits, bs).to(bonus.dtype).view_as(bonus)
'''),
            ('''            sync_site=SpecTpSyncSite.DSPARK_ACCEPT_GRAPH,
            simulated_correct_len=self.simulated_correct_len,
''',
             '''            sync_site=SpecTpSyncSite.DSPARK_ACCEPT_GRAPH,
            simulated_correct_len=self.simulated_correct_len,
            target_logits=None if self.simulate_fold else logits,
'''),
            ('''                verify_num_draft_tokens=verify_num_draft_tokens,
            )
''',
             '''                verify_num_draft_tokens=verify_num_draft_tokens,
            )
        elif target_logits is not None:
            # The accept's bonus is the real-length row's; take the simulated row's argmax.
            bonus = (
                target_logits.index_select(
                    0, _simulated_rows(correct_len, verify_num_draft_tokens)
                )
                .argmax(dim=-1)
                .to(bonus.dtype)
                .view_as(bonus)
            )
'''),
            ('''    """Temperature-sample every verify row and return the row at the (simulated)
    correct_len as the bonus token. Temperature only: callers exclude top-p/top-k/min-p."""
    bonus = simulated_bonus_sample(
        target_logits=target_logits,
        correct_len=correct_len,
        temperatures=sampling_info.temperatures,
        bs=bs,
        rows_per_request=verify_num_draft_tokens,
    )
    return bonus.to(greedy_bonus.dtype).view_as(greedy_bonus)
''',
             '''    """The bonus token at the (simulated) correct_len row, drawn from the target
    distribution the sampling accept uses: temperature, then top-k / top-p."""
    if sampling_info.need_top_k_sampling or sampling_info.need_top_p_sampling:
        probs = build_speculative_verify_target_probs(
            next_token_logits=target_logits.index_select(
                0, _simulated_rows(correct_len, verify_num_draft_tokens)
            ),
            sampling_info=sampling_info,
            draft_token_num=1,
            bs=bs,
        ).view(bs, -1)
        bonus = torch.multinomial(probs, num_samples=1)
    else:
        bonus = simulated_bonus_sample(
            target_logits=target_logits,
            correct_len=correct_len,
            temperatures=sampling_info.temperatures,
            bs=bs,
            rows_per_request=verify_num_draft_tokens,
        )
    return bonus.to(greedy_bonus.dtype).view_as(greedy_bonus)


def _simulated_rows(correct_len: torch.Tensor, rows_per_request: int) -> torch.Tensor:
    bs = correct_len.shape[0]
    return (
        torch.arange(bs, device=correct_len.device) * rows_per_request
        + correct_len.long()
    )
'''),
        ],
    ),
    (
        '''sglang/srt/speculative/dspark_components/dspark_worker_v2.py''',
        '''        # Simulated acceptance overrides correct_len; the bonus is drawn at that row''',
        [
            ('''        # Simulated acceptance overrides correct_len, so only the bonus token needs the
        # request's temperature: draft + accept run greedy, the bonus is still sampled.
''',
             '''        # Simulated acceptance overrides correct_len; the bonus is drawn at that row
        # with the request's sampling params. SGLANG_SIMULATE_ACC_GREEDY also runs the
        # draft and accept greedy.
'''),
            ('''            self._simulate_acc_greedy
''',
             '''            self._simulated_correct_len is not None
'''),
            ('''            and not batch.has_grammar
            and verify_logits_adjustments_are_noop(sampling_info)
            and not sampling_info.need_top_p_sampling
            and not sampling_info.need_top_k_sampling
            and not sampling_info.need_min_p_sampling
''',
             ''''''),
            ('''            sampling_info = None
''',
             '''            if (
                self._simulate_acc_greedy
                and not batch.has_grammar
                and verify_logits_adjustments_are_noop(sampling_info)
                and not sampling_info.need_top_p_sampling
                and not sampling_info.need_top_k_sampling
                and not sampling_info.need_min_p_sampling
            ):
                sampling_info = None
'''),
            ('''            )
        ):
''',
             '''            )
        ):
            if simulate_bonus_sampling_info is not None:
                epilogue.begin_sim_bonus(
                    bs, simulate_bonus_sampling_info.temperatures[:bs]
                )
'''),
            ('''            and simulate_bonus_sampling_info is None
''',
             '''            # A simulated bonus is drawn in the region from this batch's temperatures.
            and (
                simulate_bonus_sampling_info is None
                or simulate_bonus_sampling_info is sampling_info
            )
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
        # One atomic write per file: the file is either pristine (sentinel
        # absent) or fully patched (sentinel present). OLD blocks carry anchor
        # context lines that legitimately survive the patch, so presence of the
        # sentinel -- absent from the pristine file, unique in the patched one
        # -- is the already-patched test.
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
