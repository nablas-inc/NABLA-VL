import tokenizers
from deepspeed.utils import logger
from transformers import AutoTokenizer, PreTrainedTokenizer, TrainingArguments

from .constants import (
    CHAT_TEMPLATE_WITHOUT_SYSTEM_MESSAGE,
    IM_END,
    IM_SEP,
    IM_START,
    IMAGE_TOKEN,
)


def build_tokenizer(
    args: TrainingArguments,
    *,
    image_token: str = IMAGE_TOKEN,
    padding_side: str = "right",
    chat_template_without_system_message: str = CHAT_TEMPLATE_WITHOUT_SYSTEM_MESSAGE,
) -> PreTrainedTokenizer:
    logger.info("Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side=padding_side,
    )
    tokenizer.add_tokens([image_token], special_tokens=True)
    # Phi-4 strips new lines from the beginning and end of the input,
    # so we need to add the IM_START, IM_SEP, and IM_END tokens without stripping
    tokenizer.add_tokens(
        [
            tokenizers.AddedToken(
                IM_START, lstrip=False, rstrip=False, normalized=False, special=True
            ),
            tokenizers.AddedToken(
                IM_SEP, lstrip=False, rstrip=False, normalized=False, special=True
            ),
            tokenizers.AddedToken(
                IM_END, lstrip=False, rstrip=False, normalized=False, special=True
            ),
        ]
    )
    tokenizer.chat_template = chat_template_without_system_message
    return tokenizer
