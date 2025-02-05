from typing import Any, List, Optional

import torch
from PIL import Image
from torch import LongTensor
from transformers import PreTrainedModel, PreTrainedTokenizer

from .constants import IMAGE_TOKEN, IMAGE_TOKEN_ID, SYSTEM_PROMPT
from .transforms import DataPipeline


def get_input_ids(
    tokenizer: PreTrainedTokenizer,
    instruction: str,
    num_images: int,
) -> LongTensor:
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    if num_images > 0:
        instruction += f"{IMAGE_TOKEN * num_images}\n{instruction}"
    input_ids = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": instruction,
            },
        ],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = torch.where(
        input_ids == image_token_id,
        IMAGE_TOKEN_ID,
        input_ids,
    )
    return input_ids


def run_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    data_pipeline: DataPipeline,
    instruction: str,
    *,
    images: Optional[List[Image.Image]] = None,
    device: Any = "cuda",
) -> str:
    batch = ({"images": images},)
    batch = data_pipeline(batch)
    # Multi-images are arranged into a single image
    batch["input_ids"] = get_input_ids(tokenizer, instruction, 1)
    batch["input_ids"] = batch["input_ids"].to(device)
    # FIXME
    # batch["attention_masks"] = batch["input_ids"].ne(tokenizer.pad_token_id)
    # TODO: Rename it
    for i, mask in enumerate(batch["patch_attention_masks"]):
        batch["patch_attention_masks"][i] = mask.to(device)
    batch["images"] = [i.to(device) for i in batch["images"]]
    with torch.autocast(device), torch.inference_mode():
        response = model.generate(**batch, max_new_tokens=512)
    return tokenizer.batch_decode(response, skip_special_tokens=False)
