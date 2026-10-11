"""Transformers >=5.17 compat for the DSV4.1 multimodal token path.

sglang's ``get_processor`` deliberately returns a bare tokenizer for
``deepseek_v41`` with vision (utils/hf_transformers/processor.py: the
``vision_n_layers > 0`` branch calls ``get_tokenizer``), and
``MultimodalSpecialTokens.convert_to_str`` then reads ``processor.tokenizer``.
Transformers 5.17 renamed that attribute to ``_tokenizer`` on
``TokenizersBackend``, so the read raises AttributeError at tokenizer-manager
init and the server never becomes healthy. Upstream main carries the same code
against its own 5.19 nightly stack (verified identical at d32d8fa0).

This shim resolves the tokenizer handle the same way transformers'
``get_tokenizer_from_processor`` does, falling back to the processor itself
when it already exposes ``convert_ids_to_tokens`` (TokenizersBackend does).
"""
import logging

logger = logging.getLogger(__name__)


def install(module):
    cls = getattr(module, "MultimodalSpecialTokens", None)
    if cls is None or not hasattr(cls, "convert_to_str"):
        raise RuntimeError(
            "mm_tokenizer_compat: backend drifted (MultimodalSpecialTokens.convert_to_str missing)"
        )

    def convert_to_str(self, token, processor):
        if token is None or isinstance(token, str):
            return token
        tok = getattr(processor, "tokenizer", None)
        if tok is None and hasattr(processor, "convert_ids_to_tokens"):
            tok = processor  # transformers >= 5.17 TokenizersBackend
        if tok is None:
            tok = getattr(processor, "_tokenizer", None)
        if tok is None:
            raise AttributeError(
                "mm_tokenizer_compat: processor exposes no tokenizer handle "
                f"({type(processor).__name__})"
            )
        return tok.convert_ids_to_tokens([token])[0]

    cls.convert_to_str = convert_to_str
    logger.warning(
        "DSV41 mm_tokenizer_compat ARMED: MultimodalSpecialTokens.convert_to_str "
        "resolves the tokenizer handle across the transformers 5.17 _tokenizer rename"
    )
