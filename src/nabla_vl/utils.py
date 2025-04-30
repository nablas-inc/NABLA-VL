import math
import os
import random
from typing import Any, Tuple, Union

import numpy as np
import torch
from deepspeed.utils import logger
from torch import BoolTensor, FloatTensor, Tensor
from torch.nn import Module
from transformers import PretrainedConfig, TrainingArguments


def get_dtype_by_args(
    args_or_config: Union[TrainingArguments, PretrainedConfig]
) -> Any:
    if args_or_config.fp16 is True:
        dtype = torch.float16
    else:
        if args_or_config.bf16 is True:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    return dtype


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # RuntimeError: Unable to find a valid cuDNN algorithm to run convolution
    torch.backends.cudnn.benchmark = False


def make_inputs_require_grad(module: Module, input: Tensor, output: Tensor) -> None:
    output.requires_grad_(True)


# https://github.com/nablas-inc/NABLA-VL-Dev/blob/013d42d93332696b7eb4e9061fd5a554e328afd0/src/nabla_vl/train/train.py#L2172C9-L2175C10
def get_total_params(model: Module) -> Tuple[int, int]:
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        if hasattr(p, "ds_numel") is True:
            total_params += p.ds_numel
            if p.requires_grad is True:
                trainable_params += p.ds_numel
        else:
            total_params += p.numel()
            if p.requires_grad is True:
                trainable_params += p.numel()
    return total_params, trainable_params


def to_patch_attention_mask_size(
    h: int,
    w: int,
    patch_size: int,
    *,
    ceil: bool = False,
) -> Tuple[int, int]:
    if ceil is True:
        h_mask = math.ceil(h / patch_size)
        w_mask = math.ceil(w / patch_size)
    else:
        h_mask = int(h / patch_size)
        w_mask = int(w / patch_size)
    return h_mask, w_mask


def get_patch_size(
    image: FloatTensor,
    patch_attention_mask: BoolTensor,
    *,
    default_patch_size: int = 14,
) -> Tuple[FloatTensor, BoolTensor]:
    patch_size_h = int(image.size(-2) / patch_attention_mask.size(-2))
    patch_size_w = int(image.size(-1) / patch_attention_mask.size(-1))
    if patch_size_h != patch_size_w:
        logger.warning(
            "invalid patch size detected: "
            "patch_size_h and patch_size_w are different. "
            f"default patch size (={default_patch_size}) will be used which may causes another error.\n"  # NOQA
            "=== warning report ===\n"
            f"image size={image.size()}\n"
            f"patch_attention_mask size={patch_attention_mask.size()}\n"
            f"patch_size_h={patch_size_h}, patch_size_w={patch_size_w}"
        )
        return default_patch_size
    return patch_size_h
