from typing import Any, List, Optional

import torch
from PIL import Image
from torch import LongTensor
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer

from .constants import IMAGE_TOKEN, IMAGE_TOKEN_ID, SYSTEM_PROMPT
from .transforms import DataPipeline


def get_input_ids(
    tokenizer: PreTrainedTokenizer,
    instruction: str,
    num_images: int,
    *,
    replace_image_tokens: bool = True,
) -> LongTensor:
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    if num_images > 0:
        instruction = f"{IMAGE_TOKEN * num_images}\n{instruction}"
    if replace_image_tokens is True:
        num_image_tokens = instruction.count(IMAGE_TOKEN)
        if num_image_tokens > 0:
            for i in range(num_image_tokens):
                instruction = instruction.replace(
                    IMAGE_TOKEN,
                    f"<image {i + 1}>",
                    1,
                )
            instruction = IMAGE_TOKEN + "\n" + instruction
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
    max_new_tokens: int = 512,
    replace_image_tokens: bool = True,
) -> str:
    batch = ({"images": images},)
    batch = data_pipeline(batch)
    # Multi-images are arranged into a single image
    batch["input_ids"] = get_input_ids(
        tokenizer,
        instruction,
        1,
        replace_image_tokens=replace_image_tokens,
    )
    batch["input_ids"] = batch["input_ids"].to(device)
    batch["attention_mask"] = batch["input_ids"].ne(tokenizer.pad_token_id)
    # TODO: Rename it
    for i, mask in enumerate(batch["patch_attention_masks"]):
        batch["patch_attention_masks"][i] = mask.to(device)
    batch["images"] = [i.to(device) for i in batch["images"]]
    with torch.autocast(device), torch.inference_mode():
        response = model.generate(
            **batch, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id
        )
    # Skip eos token
    return tokenizer.batch_decode(response[:, :-1], skip_special_tokens=False)


def run_model_with_stream(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    data_pipeline: DataPipeline,
    instruction: str,
    *,
    images: Optional[List[Image.Image]] = None,
    device: Any = "cuda",
    max_new_tokens: int = 512,
    replace_image_tokens: bool = True,
) -> None:
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=False,
        skip_special_tokens=True,
    )
    batch = ({"images": images},)
    batch = data_pipeline(batch)
    # Multi-images are arranged into a single image
    batch["input_ids"] = get_input_ids(
        tokenizer,
        instruction,
        1,
        replace_image_tokens=replace_image_tokens,
    )
    batch["input_ids"] = batch["input_ids"].to(device)
    batch["attention_mask"] = batch["input_ids"].ne(tokenizer.pad_token_id)
    # TODO: Rename it
    for i, mask in enumerate(batch["patch_attention_masks"]):
        batch["patch_attention_masks"][i] = mask.to(device)
    batch["images"] = [i.to(device) for i in batch["images"]]
    with torch.autocast(device), torch.inference_mode():
        model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            streamer=streamer,
        )
