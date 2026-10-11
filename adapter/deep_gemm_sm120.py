"""SM120 paged-sparse MQA logits gate lift for the DSV4.1 candidate indexer.

Upstream's configurer gates DEEPGEMM_PAGED_SPARSE_MQA_LOGITS on
``DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and get_platform().is_sm100``
(strict SM100 family), while ``make_candidate_indexer`` only requires
``is_sm100_or_newer()`` — so SM120/121 (GB10) with a checkpoint that enables
the candidate scheme (candidate_topk_blocks=2048) raises at backend init even
though sgl-deep-gemm 0.2.2 ships both ``get_paged_sparse_mqa_logits_metadata``
and ``fp8_fp4_paged_sparse_mqa_logits`` (verified callable in-image).

DSV41_SM120_PAGED_SPARSE=1 recomputes the two flags including SM120/121 in the
Blackwell family. If the JIT has no SM120 kernel for the paged-sparse path it
fails loudly at warmup — fall back to zeroing candidate_topk_blocks (dense
full-top-k path) in that case.
"""
import logging

logger = logging.getLogger(__name__)


def install(module):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("deep_gemm_sm120: no CUDA device at configurer import")
    major = torch.cuda.get_device_capability(0)[0] * 10 + torch.cuda.get_device_capability(0)[1]
    if major not in (120, 121):
        # Not this fleet: leave upstream flags untouched.
        logger.info("deep_gemm_sm120: device sm%d, not SM120/121 — no-op", major)
        return
    was_blackwell = module.DEEPGEMM_BLACKWELL
    module.DEEPGEMM_BLACKWELL = module.ENABLE_JIT_DEEPGEMM
    module.DEEPGEMM_PAGED_SPARSE_MQA_LOGITS = module._supports_paged_sparse_mqa_logits()
    logger.warning(
        "DSV41 deep_gemm_sm120 ARMED: DEEPGEMM_BLACKWELL %s -> %s "
        "(paged sparse MQA logits %s); candidate indexer enabled on sm%d",
        was_blackwell, module.DEEPGEMM_BLACKWELL,
        module.DEEPGEMM_PAGED_SPARSE_MQA_LOGITS, major,
    )
