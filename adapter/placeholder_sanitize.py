"""Sanitize literal image-placeholder tokens from message text on the recipe line.

sglang >= the deepseek-recipe migration renders DeepSeek-V4/V4.1 prompts through the
external `deepseek-recipe` package (parser/deepseek_recipe_renderer.py). A literal
image placeholder token (``<｜deepseek_image｜>``) inside a message's TEXT is rejected
by the recipe (ConversionError -> HTTP 400/500) and, because the token re-enters the
transcript, the failure is self-sustaining for the rest of the conversation — the
same incident class MiaAI-Lab #15 measured at 111/145 failed Anthropic requests.

This adapter wraps ``deepseek_recipe_renderer.pre_normalize`` (the single entry that
every recipe-rendered request passes through) and rewrites the token to ``[image]``
in every textual field of every message before the recipe sees it. Real image
content (image_url blocks / multimodal inputs) is untouched; only string fields that
would reach the recipe as text are sanitized. Idempotent; if the target symbol is
missing (older tree without the recipe), the hook logs and does nothing.
Set DSV41_PLACEHOLDER_SANITIZE=0 to disable.
"""
import logging
import os

logger = logging.getLogger(__name__)

PLACEHOLDER = '<｜deepseek_image｜>'
REPLACEMENT = '[image]'


def enabled():
    return os.environ.get('DSV41_PLACEHOLDER_SANITIZE', '1').strip() not in ('0', 'false', 'off', 'no')


def _sanitize_value(v):
    if isinstance(v, str):
        return v.replace(PLACEHOLDER, REPLACEMENT) if PLACEHOLDER in v else v
    if isinstance(v, list):
        return [_sanitize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _sanitize_value(x) for k, x in v.items()}
    return v


def sanitize_body(body):
    """Return a body whose message text fields carry no literal placeholder."""
    messages = body.get('messages')
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                for key in ('content', 'reasoning_content'):
                    if key in msg:
                        msg[key] = _sanitize_value(msg[key])
    return body


def make_sanitizing_prenormalize(original):
    def pre_normalize(body):
        return original(sanitize_body(body))
    pre_normalize.__wrapped__ = original
    return pre_normalize


def install(module):
    if not enabled():
        logger.warning('DSV41 placeholder sanitize disabled (DSV41_PLACEHOLDER_SANITIZE=0)')
        return
    original = getattr(module, 'pre_normalize', None)
    if original is None:
        logger.warning('DSV41 placeholder sanitize: pre_normalize not found; not installed')
        return
    if getattr(original, '__wrapped__', None) is not None:
        return
    module.pre_normalize = make_sanitizing_prenormalize(original)
    logger.warning('DSV41 placeholder sanitize: literal %r in message text becomes %r '
                   'before the recipe sees it; DSV41_PLACEHOLDER_SANITIZE=0 disables',
                   PLACEHOLDER, REPLACEMENT)
