"""Disable the DSV4.1 candidate indexer on SM120/121 (GB10).

Empirically (2026-10-11): lifting the configurer's is_sm100 gate lets the
candidate path reach deep_gemm's paged sparse MQA logits, which then asserts
``block_kv == 64`` (attention.hpp:438) against this pool's compressed page
layout — the SM120 kernel surface is not parameter-compatible with the
candidate scheme. Zeroing the three candidate keys routes the backend to the
dense full-top-k path (FullTopKIndexer + dense_blocks), which upstream
supports on SM120 and still carries the whole #41589 prefill wave.
"""
import logging

logger = logging.getLogger(__name__)

_KEYS = ("candidate_topk_blocks", "candidate_block_size", "candidate_source_layer_id")


def install(module):
    orig = module.normalize_deepseek_v41_config

    def normalize_deepseek_v41_config(values):
        out = orig(values)
        dropped = {k: out.pop(k) for k in _KEYS if out.get(k)}
        if dropped:
            # candidate_source_layer_id=-1 is the config class's "no candidates"
            # sentinel; 0 would make layer 0 a source with no publisher.
            out.update({k: (-1 if k == "candidate_source_layer_id" else 0)
                        for k in dropped})
            logger.warning(
                "DSV41 sm120_candidates ARMED: zeroed %s (SM120 paged-sparse "
                "candidate kernels are block_kv-incompatible; dense full-top-k path)",
                sorted(dropped),
            )
        return out

    module.normalize_deepseek_v41_config = normalize_deepseek_v41_config
