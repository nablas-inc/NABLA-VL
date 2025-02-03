import abc
import json
import math
import re
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2  # type: ignore
import numpy as np
import torch
import torchvision.transforms.functional as VF
from cv2 import FONT_HERSHEY_COMPLEX_SMALL, LINE_AA  # type: ignore
from deepspeed.utils import logger
from PIL import Image
from torch import BoolTensor, FloatTensor, LongTensor, Tensor
from torchvision.transforms import InterpolationMode
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    TrainingArguments,
)
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils import PreTokenizedInput, TextInput
from transformers.utils import TensorType

from .config import NablaVLConfig
from .constants import (
    IGNORE_TOKEN_ID,
    IM_END,
    IM_SEP,
    IM_START,
    IMAGE_TOKEN,
    IMAGE_TOKEN_ID,
    MEAN,
    STD,
    SYSTEM_PROMPT,
)
from .utils import get_dtype_by_args, get_patch_size, to_patch_attention_mask_size


def build_data_pipeline(
    args_or_config: Union[TrainingArguments, PretrainedConfig],
    tokenizer: PreTrainedTokenizer,
) -> "DataPipeline":
    logger.info("Initializing data pipeline")
    data_pipeline = DataPipeline(
        [
            (
                AddThumbnail(
                    factor=args_or_config.factor,
                    min_pixels=args_or_config.min_pixels,
                    max_pixels=args_or_config.max_pixels,
                    patch_size=args_or_config.patch_size,
                    min_shorter_size=args_or_config.min_shorter_size,
                    max_longer_size=args_or_config.max_longer_size,
                    thumbnail_size=args_or_config.thumbnail_size,
                    add_marks=args_or_config.add_marks,
                )
                if args_or_config.add_thumbnail is True
                else Identity()
            ),
            Tokenize(
                tokenizer,
                apply_chat_template=args_or_config.apply_chat_template,
                eos_token=args_or_config.eos_token,
            ),
            Truncate(max_length=args_or_config.max_length),
            ToTensor(dtype=get_dtype_by_args(args_or_config)),
            AddNumTiles(),
            Factorize(factor=args_or_config.factor),
            AddPatchAttentionMask(patch_size=args_or_config.patch_size),
            AggregateImages(
                factor=args_or_config.factor,
                patch_size=args_or_config.patch_size,
                wrap_images=(
                    args_or_config.wrap_images
                    if hasattr(args_or_config, "wrap_images")
                    else False
                ),
            ),
            (
                AnyRes(
                    factor=args_or_config.factor,
                    min_pixels=args_or_config.min_pixels,
                    max_pixels=args_or_config.max_pixels,
                    min_shorter_size=args_or_config.min_shorter_size,
                    max_longer_size=args_or_config.max_longer_size,
                    max_total_size=args_or_config.max_total_size,
                )
                if args_or_config.use_anyres is True
                else Resize(
                    min_pixels=args_or_config.min_pixels,
                    max_pixels=args_or_config.max_pixels,
                    min_shorter_size=args_or_config.min_shorter_size,
                    max_longer_size=args_or_config.max_longer_size,
                )
            ),
            Scale(scaler=1.0 / 255.0),
            Normalize(
                mean=MEAN[args_or_config.normalize_type],
                std=STD[args_or_config.normalize_type],
            ),
        ],
        tokenizer.padding_side,
        tokenizer.pad_token_id,
        patch_size=args_or_config.patch_size,
    )
    return data_pipeline


class Transform(ABC):
    def __init__(self, *, p: float = 1.0) -> None:
        self.p = p

    def __str__(self) -> str:
        return "Transform"

    @abc.abstractmethod
    def apply(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return sample

    def apply_if_needed(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if torch.rand(1) < self.p:
            sample = self.apply(sample)
        return sample

    def __call__(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return tuple(map(lambda sample: self.apply_if_needed(sample), batch))


class Identity(Transform):
    def apply(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return sample


class DataPipeline(object):
    def __init__(
        self,
        transforms: List[Transform],
        padding_side: str,
        pad_token_id: int,
        *,
        patch_size: int = 14,
    ) -> None:
        self.transforms = transforms
        self.padding_side = padding_side
        self.pad_token_id = pad_token_id
        self.patch_size = patch_size

    def pad_to_max_len(self, x: List[LongTensor]) -> LongTensor:
        if self.padding_side == "left":
            x = [torch.flip(x[i], [0]) for i in range(len(x))]
        x = torch.nn.utils.rnn.pad_sequence(
            x,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        if self.padding_side == "left":
            x = torch.flip(x, [1])
        return x

    def pad_to_max_size(
        self,
        images: List[FloatTensor],
        patch_attention_masks: List[BoolTensor],
    ) -> Tuple[List[FloatTensor], List[BoolTensor]]:
        max_h = max_w = 0
        max_h_mask = max_w_mask = 0
        for image, patch_attention_mask in zip(images, patch_attention_masks):
            h, w = image.size()[-2:]
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            h_mask, w_mask = patch_attention_mask.size()[-2:]
            max_h_mask = max(max_h_mask, h_mask)
            max_w_mask = max(max_w_mask, w_mask)
        assert int(max_h / max_h_mask) == int(max_w / max_w_mask) == self.patch_size
        new_images = []
        new_patch_attention_masks = []
        for image, patch_attention_mask in zip(images, patch_attention_masks):
            n, c, h, w = image.size()
            new_image = torch.zeros(
                (
                    n,
                    c,  # NOTE: All images are converted to RGB
                    max_h,
                    max_w,
                ),
                dtype=image.dtype,
                device=image.device,
            )
            new_image[:, :, :h, :w] = image
            new_images.append(new_image)
            _, h_mask, w_mask = patch_attention_mask.size()
            new_patch_attention_mask = torch.zeros(
                (n, max_h_mask, max_w_mask),
                dtype=torch.bool,
                device=image.device,
            )
            new_patch_attention_mask[:, :h_mask, :w_mask] = patch_attention_mask
            new_patch_attention_masks.append(new_patch_attention_mask)
        return new_images, new_patch_attention_masks

    def __call__(self, batch: Tuple[Dict[str, Any], ...]) -> Dict[str, Any]:
        for transform in self.transforms:
            batch = transform(batch)
        input_ids = []
        labels = []
        images = []
        num_images = []
        patch_attention_masks = []
        num_tiles = []
        for sample in batch:
            if "input_ids" in sample:
                input_ids.append(sample["input_ids"])
            if "label" in sample:
                labels.append(sample["label"])
            if "images" in sample:
                images.extend(sample["images"])
                num_images.extend([len(sample["images"])] * len(sample["images"]))
            if "patch_attention_mask" in sample:
                patch_attention_masks.extend(sample["patch_attention_mask"])
            if "num_tiles" in sample:
                num_tiles.extend(sample["num_tiles"])
        # NOTE: len(input_ids) == len(labels) == 0 during inference
        if len(input_ids) > 0:
            input_ids = self.pad_to_max_len(input_ids)
            attention_masks = input_ids.ne(self.pad_token_id)
        else:
            input_ids = None
            attention_masks = None
        if len(labels) > 0:
            labels = self.pad_to_max_len(labels)
        else:
            labels = None
        images, patch_attention_masks = self.pad_to_max_size(
            images,
            patch_attention_masks,
        )
        return {
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "patch_attention_masks": patch_attention_masks,
            "labels": labels,
            "images": images,
            "num_images": num_images,
            "num_tiles": num_tiles,
        }


class AggregateImages(Transform):
    """Arrange images or video frames to a single image

    Note
    ----
    Because max tile size (=980) could be larger than the usual tile sizes
    (=384 or 448), resizing and processing multiple images or video frames causes many
    paddings.
    """

    def __init__(
        self,
        *,
        factor: int = 14,
        patch_size: int = 14,
        image_token_id: int = IMAGE_TOKEN_ID,
        wrap_images: bool = True,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.factor = factor
        self.patch_size = patch_size
        self.image_token_id = image_token_id
        self.wrap_images = wrap_images

    def apply(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "images" in sample:
            if len(sample["images"]) > 1:
                images = sample["images"]
                num_images_per_row = math.ceil(math.sqrt(len(images)))
                # Estimate the max height and width after arrangement
                hw = 0
                for image in images:
                    hw += max(image.size()[-2:])
                hw = math.ceil(hw / self.factor) * self.factor
                # NOTE: All images have the same dtype and device
                new_image = torch.zeros(
                    (
                        1,
                        3,  # NOTE: All images are converted to RGB
                        hw,
                        hw,
                    ),
                    dtype=images[0].dtype,
                    device=images[0].device,
                )
                h_mask, w_mask = to_patch_attention_mask_size(hw, hw, self.patch_size)
                new_patch_attention_mask = torch.zeros(
                    (
                        1,
                        h_mask,
                        w_mask,
                    ),
                    dtype=torch.bool,
                    device=images[0].device,
                )
                max_h = 0
                max_w = 0
                max_h_per_w = 0
                max_w_per_h = 0
                for i, image in enumerate(images):
                    # Put marks on the top left
                    # TODO: Rename it
                    dtype = image.dtype
                    device = image.device
                    image = image.permute(0, 2, 3, 1)
                    image = image.type(torch.uint8).numpy()
                    img = np.ascontiguousarray(image[0])
                    img = cv2.putText(
                        img=img,
                        text=str(i + 1),
                        org=(4, 24),
                        fontFace=FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale=1.3,
                        color=(255, 0, 0),
                        thickness=1,
                        lineType=LINE_AA,
                    )
                    image = torch.from_numpy(img).to(device, dtype=dtype)
                    image = image[None, :, :, :]
                    image = image.permute(0, 3, 1, 2)
                    h, w = image.size()[-2:]
                    if self.wrap_images is True:
                        if i % num_images_per_row == 0:
                            max_w = 0
                            max_h = max_h_per_w
                    else:
                        if max_w + w > hw:
                            max_w = 0
                            max_h = max_h_per_w
                    hs = max_h
                    he = hs + h
                    max_h_per_w = max(max_h_per_w, he)
                    ws = max_w
                    we = ws + w
                    max_w_per_h = max(max_w_per_h, we)
                    max_w = we
                    hsm, wsm = to_patch_attention_mask_size(
                        hs,
                        ws,
                        self.patch_size,
                    )
                    hem, wem = to_patch_attention_mask_size(
                        he,
                        we,
                        self.patch_size,
                    )
                    new_image[:, :, hs:he, ws:we] = image
                    new_patch_attention_mask[:, hsm:hem, wsm:wem] = True
                # Strip zero part
                max_h_per_w_m, max_w_per_h_m = to_patch_attention_mask_size(
                    max_h_per_w,
                    max_w_per_h,
                    self.patch_size,
                    ceil=True,
                )
                new_image = new_image[:, :, :max_h_per_w, :max_w_per_h]
                new_patch_attention_mask = new_patch_attention_mask[
                    :, :max_h_per_w_m, :max_w_per_h_m
                ]
                # Reduce the number of image tokens to 1
                if "input_ids" in sample and "label" in sample:
                    input_ids = sample["input_ids"]
                    label = sample["label"]
                    keep = input_ids != self.image_token_id
                    # Leave first image token
                    keep[torch.where(input_ids == self.image_token_id)[0][0]] = True
                    input_ids = input_ids[keep]
                    label = label[keep]
                    sample["input_ids"] = input_ids
                    sample["label"] = label
                sample["images"] = [new_image]
                sample["patch_attention_mask"] = [new_patch_attention_mask]
                # Reset num_tiles
                sample["num_tiles"] = [[1, 1]]
        return sample


class AddThumbnail(Transform):
    def __init__(
        self,
        *,
        image_token: str = IMAGE_TOKEN,
        factor: int = 14,
        min_pixels: int = 14 * 14,
        max_pixels: int = 384 * 384,
        patch_size: int = 14,
        dtype: Any = torch.float32,
        scaler: float = 1.0 / 255.0,
        min_shorter_size: int = 70,
        max_longer_size: int = 980,
        m: float = 1.5,
        thumbnail_size: int = 448,
        mode: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
        add_marks: bool = False,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.image_token = image_token
        self.factor = factor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.dtype = dtype
        self.scaler = scaler
        self.min_shorter_size = min_shorter_size
        self.max_longer_size = max_longer_size
        self.m = m
        self.thumbnail_size = thumbnail_size
        self.mode = mode
        self.antialias = antialias
        self.add_marks = add_marks

    def get_mark_positions(self, h: int, w: int) -> List[List[int]]:
        try:
            h, w = smart_resize(
                h,
                w,
                factor=self.factor,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        # smart_resize raises ValueError when image or aspect ratio is too small
        except ValueError:
            raise ValueError("TODO")
        # Resize image to the size of smart_resize
        if min(h, w) < self.min_shorter_size:
            scale = self.min_shorter_size / min(h, w)
            # Round by patch_size to resize patch_attention_mask correctly
            h = math.ceil(scale * h / self.patch_size) * self.patch_size
            w = math.ceil(scale * w / self.patch_size) * self.patch_size
        mark_positions = []
        for i in range(0, h, self.max_longer_size):
            for j in range(0, w, self.max_longer_size):
                mark_positions.append([i, j])
        return mark_positions, (h, w)

    def apply(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "images" in sample:
            images = []
            has_thumbnail = []
            for i in range(len(sample["images"])):
                image = sample["images"][i]
                image = torch.tensor(image, dtype=self.dtype)
                # n x h x w x c ---> n x c x h x w
                image = image.permute(0, 3, 1, 2)
                n, _, h, w = image.size()
                if n == 1 and max(h, w) > int(self.m * self.max_longer_size):
                    image *= self.scaler
                    mark_positions, (h, w) = self.get_mark_positions(
                        *sample["images"][i].shape[1:-1]
                    )
                    scale = self.thumbnail_size / max(h, w)
                    # Round by patch_size to resize patch_attention_mask correctly
                    # NOTE: int causes zero division
                    h_thumbnail = (
                        math.ceil(scale * h / self.patch_size) * self.patch_size
                    )
                    w_thumbnail = (
                        math.ceil(scale * w / self.patch_size) * self.patch_size
                    )
                    h_tile = int(self.max_longer_size / h * h_thumbnail)
                    w_tile = int(self.max_longer_size / w * w_thumbnail)
                    image = VF.resize(
                        image,
                        (h_thumbnail, w_thumbnail),
                        self.mode,
                        None,
                        self.antialias,
                    )
                    image /= self.scaler
                    # n x c x h x w ---> n x h x w x c
                    image = image.permute(0, 2, 3, 1)
                    image = image.numpy().astype(np.uint8)
                    if self.add_marks is True:
                        for j, (h, w) in enumerate(mark_positions):
                            h = int(h_tile / self.max_longer_size * h)
                            w = int(w_tile / self.max_longer_size * w)
                            # TODO: Rename it
                            img = np.ascontiguousarray(image[0])
                            max_h, max_w = img.shape[:-1]
                            max_h -= 1
                            max_w -= 1
                            img = cv2.putText(
                                img=img,
                                text=str(j),
                                org=(w + 2, h + 17),
                                fontFace=FONT_HERSHEY_COMPLEX_SMALL,
                                fontScale=1,
                                color=(255, 0, 0),
                                thickness=1,
                                lineType=LINE_AA,
                            )
                            cv2.rectangle(
                                img=img,
                                pt1=(w, h),
                                pt2=(min(max_w, w + w_tile), min(max_h, h + h_tile)),
                                color=(255, 0, 0),
                                thickness=1,
                                lineType=LINE_AA,
                            )
                            image[0] = img
                    images.extend([image, sample["images"][i]])
                    has_thumbnail.append(True)
                else:
                    images.append(sample["images"][i])
                    has_thumbnail.append(False)
            sample["images"] = images
        if "conversations" in sample:
            new_instruction = ""
            instruction = sample["conversations"][0]["value"]
            position = 0
            for i, match in enumerate(re.finditer(self.image_token, instruction)):
                start, end = match.span()
                new_instruction += instruction[position:start]
                if has_thumbnail[i] is True:
                    new_instruction += self.image_token * 2
                else:
                    new_instruction += self.image_token
                position = end
            new_instruction += instruction[position:]
            sample["conversations"][0]["value"] = new_instruction
        return sample


class AddNumTiles(Transform):
    def __init__(self, *, p: float = 1.0) -> None:
        super().__init__(p=p)

    def apply(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "images" in sample:
            sample["num_tiles"] = []
            for _ in range(len(sample["images"])):
                sample["num_tiles"].append([1, 1])
        return sample


class Factorize(Transform):
    def __init__(
        self,
        *,
        factor: int = 14,
        mode: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.factor = factor
        self.mode = mode
        self.antialias = antialias

    def apply(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "images" in sample:
            for i in range(len(sample["images"])):
                image = sample["images"][i]
                h, w = image.size()[-2:]
                try:
                    h, w = smart_resize(
                        *image.size()[-2:],
                        factor=self.factor,
                        min_pixels=self.factor,
                        max_pixels=float("inf"),
                    )
                # smart_resize raises ValueError when image or aspect ratio is too small
                except ValueError:
                    raise ValueError("TODO")
                image = VF.resize(
                    image,
                    (h, w),
                    self.mode,
                    None,
                    self.antialias,
                )
                sample["images"][i] = image
        return sample


class AddPatchAttentionMask(Transform):
    def __init__(self, *, patch_size: int = 14, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.patch_size = patch_size

    def apply(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "images" in sample:
            patch_attention_mask = []
            for i in range(len(sample["images"])):
                image = sample["images"][i]
                n, _, h, w = image.size()
                h_mask, w_mask = to_patch_attention_mask_size(h, w, self.patch_size)
                patch_attention_mask.append(
                    torch.ones(
                        (
                            n,
                            h_mask,
                            w_mask,
                        ),
                        dtype=torch.bool,
                    )
                )
            sample["patch_attention_mask"] = patch_attention_mask
        return sample


class ToTensor(Transform):
    def __init__(
        self,
        *,
        dtype: Any = torch.float32,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.dtype = dtype

    def apply(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "images" in sample:
            for i in range(len(sample["images"])):
                image = sample["images"][i]
                image = torch.tensor(image, dtype=self.dtype)
                # n x h x w x c ---> n x c x h x w
                image = image.permute(0, 3, 1, 2)
                sample["images"][i] = image
        if "input_ids" in sample:
            input_ids = sample["input_ids"]
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            sample["input_ids"] = input_ids
        if "label" in sample:
            label = sample["label"]
            label = torch.tensor(label, dtype=torch.long)
            sample["label"] = label
        return sample


class Scale(Transform):
    def __init__(
        self,
        *,
        scaler: float = 1.0 / 255.0,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.scaler = scaler

    def apply(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "images" in sample:
            for i in range(len(sample["images"])):
                image = sample["images"][i]
                image *= self.scaler
                sample["images"][i] = image
        return sample


class Resize(Transform):
    def __init__(
        self,
        *,
        factor: int = 14,
        min_pixels: int = 28 * 28,
        max_pixels: int = 384 * 384,
        min_shorter_size: int = 70,
        max_longer_size: int = 980,
        mode: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.factor = factor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.min_shorter_size = min_shorter_size
        self.max_longer_size = max_longer_size
        self.mode = mode
        self.antialias = antialias

    def apply(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "images" in sample:
            for i in range(len(sample["images"])):
                image = sample["images"][i]
                patch_attention_mask = sample["patch_attention_mask"][i]
                try:
                    h, w = smart_resize(
                        *image.size()[-2:],
                        factor=self.factor,
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels,
                    )
                # smart_resize raises ValueError when image or aspect ratio is too small
                except ValueError:
                    raise ValueError("TODO")
                if max(h, w) > self.max_longer_size:
                    scale = self.max_longer_size / max(h, w)
                    h = int(scale * h)
                    w = int(scale * w)
                if min(h, w) < self.min_shorter_size:
                    scale = self.min_shorter_size / min(h, w)
                    h = int(scale * h)
                    w = int(scale * w)
                patch_size = get_patch_size(image, patch_attention_mask)
                image = VF.resize(
                    image,
                    (h, w),
                    self.mode,
                    None,
                    self.antialias,
                )
                patch_attention_mask = VF.resize(
                    patch_attention_mask,
                    to_patch_attention_mask_size(h, w, patch_size),
                    InterpolationMode.NEAREST,
                )
                sample["images"][i] = image
                sample["patch_attention_mask"][i] = patch_attention_mask
        return sample


class AnyRes(Transform):
    def __init__(
        self,
        *,
        factor: int = 14,
        min_pixels: int = 14 * 14,
        max_pixels: int = 384 * 384,
        min_shorter_size: int = 70,
        # TODO: Rename it to like tile_size
        max_longer_size: int = 980,
        # TODO: Rename it to like max_longer_size
        max_total_size: int = -1,
        mode: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
        max_num_tiles: int = 4,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.factor = factor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.min_shorter_size = min_shorter_size
        self.max_longer_size = max_longer_size
        self.max_total_size = max_total_size
        self.mode = mode
        self.antialias = antialias
        self.max_num_tiles = max_num_tiles

    def pad_if_needed(self, x: Tensor, divisor: int) -> Tensor:
        dim = x.dim()
        if dim == 3:
            x = x.unsqueeze(1)
        dtype = x.dtype
        n, c, h, w = x.size()
        if min(h, w) < int(self.max_longer_size / divisor):
            new_x = torch.zeros(
                (
                    n,
                    c,
                    int(self.max_longer_size / divisor),
                    int(self.max_longer_size / divisor),
                ),
                dtype=dtype,
            )
            new_x[:, :, :h, :w] = x
        else:
            new_x = x
        if dim == 3:
            new_x = new_x.squeeze(1)
        return new_x

    def get_patches(
        self,
        image: FloatTensor,
        patch_attention_mask: BoolTensor,
    ) -> Tuple[FloatTensor, BoolTensor, Tuple[int, int]]:
        patch_size = get_patch_size(image, patch_attention_mask)
        new_image = []
        new_patch_attention_mask = []
        num_tiles = [0, 0]
        for i in range(0, image.size(2), self.max_longer_size):
            num_tiles[0] += 1
            for j in range(0, image.size(3), self.max_longer_size):
                ymin = i
                xmin = j
                ymax = ymin + self.max_longer_size
                xmax = xmin + self.max_longer_size
                # TODO: Rename it
                img = self.pad_if_needed(image[:, :, ymin:ymax, xmin:xmax], 1)
                new_image.append(img)
                ymin_mask = int(ymin / patch_size)
                xmin_mask = int(xmin / patch_size)
                ymax_mask = ymin_mask + int(self.max_longer_size / patch_size)
                xmax_mask = xmin_mask + int(self.max_longer_size / patch_size)
                # TODO: Rename it
                mask = patch_attention_mask[:, ymin_mask:ymax_mask, xmin_mask:xmax_mask]
                mask = self.pad_if_needed(mask, patch_size)
                new_patch_attention_mask.append(mask)
                if num_tiles[0] == 1:
                    num_tiles[1] += 1
        new_image = torch.cat(new_image)
        new_patch_attention_mask = torch.cat(new_patch_attention_mask)
        return new_image, new_patch_attention_mask, num_tiles

    def estimate_num_tiles(self, h: int, w: int) -> Tuple[int, int]:
        num_tiles = [0, 0]
        for i in range(0, h, self.max_longer_size):
            num_tiles[0] += 1
            for j in range(0, w, self.max_longer_size):
                num_tiles[1] += 1
        return num_tiles

    def apply(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "images" in sample:
            for i in range(len(sample["images"])):
                image = sample["images"][i]
                # Do not apply AnyRes if the image is too small
                if max(image.size()[-2:]) <= self.max_longer_size:
                    continue
                else:
                    try:
                        h, w = smart_resize(
                            *image.size()[-2:],
                            factor=self.factor,
                            min_pixels=self.min_pixels,
                            max_pixels=self.max_pixels,
                        )
                    # smart_resize raises ValueError when image or aspect ratio is too small
                    except ValueError:
                        raise ValueError("TODO")
                    # Resize image to the size of smart_resize
                    patch_attention_mask = sample["patch_attention_mask"][i]
                    patch_size = get_patch_size(image, patch_attention_mask)
                    if min(h, w) < self.min_shorter_size:
                        scale = self.min_shorter_size / min(h, w)
                        # Round by patch_size to resize patch_attention_mask correctly
                        h = math.ceil(scale * h / patch_size) * patch_size
                        w = math.ceil(scale * w / patch_size) * patch_size
                    # If the number of tiles is larger than max_num_tiles, reduce it to max_num_tiles
                    for _ in range(50):
                        num_rows, num_cols = self.estimate_num_tiles(h, w)
                        if num_rows * num_cols > self.max_num_tiles:
                            # Round width to the multiple of max_longer_size
                            if num_rows == 1 or (
                                num_cols > 1
                                and h % self.max_longer_size > w % self.max_longer_size
                            ):
                                m = int(w / self.max_longer_size)
                                if m * self.max_longer_size == w:
                                    m -= 1
                                scale = m * self.max_longer_size / w
                            # Round height to the multiple of max_longer_size
                            else:
                                m = int(h / self.max_longer_size)
                                if m * self.max_longer_size == h:
                                    m -= 1
                                scale = m * self.max_longer_size / h
                            h = max(self.min_shorter_size, int(scale * h))
                            w = max(self.min_shorter_size, int(scale * w))
                        else:
                            break
                    image = VF.resize(
                        image,
                        (h, w),
                        self.mode,
                        None,
                        self.antialias,
                    )
                    patch_attention_mask = VF.resize(
                        patch_attention_mask,
                        to_patch_attention_mask_size(h, w, patch_size),
                        InterpolationMode.NEAREST,
                    )
                image, patch_attention_mask, num_tiles = self.get_patches(
                    image,
                    patch_attention_mask,
                )
                sample["images"][i] = image
                sample["patch_attention_mask"][i] = patch_attention_mask
                sample["num_tiles"][i] = num_tiles
        return sample


class Normalize(Transform):
    def __init__(
        self,
        *,
        mean: List[float] = MEAN["imagenet"],
        std: List[float] = STD["imagenet"],
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.mean = mean
        self.std = std

    def apply(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "images" in sample:
            for i in range(len(sample["images"])):
                image = sample["images"][i]
                image = VF.normalize(image, self.mean, self.std)
                sample["images"][i] = image
        return sample


class Tokenize(Transform):
    role_table = {"human": "user", "gpt": "assistant"}

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        apply_chat_template: bool = False,
        system_prompt: str = SYSTEM_PROMPT,
        image_token: str = IMAGE_TOKEN,
        eos_token: str = "\n",
        ignore_token_id: int = IGNORE_TOKEN_ID,
        image_token_id: int = IMAGE_TOKEN_ID,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.image_token = image_token
        self.eos_token = eos_token
        self.ignore_token_id = ignore_token_id
        self.image_token_id = image_token_id
        self.apply_chat_template = apply_chat_template
        self.im_start_id = tokenizer.convert_tokens_to_ids(IM_START)
        self.im_sep_id = tokenizer.convert_tokens_to_ids(IM_SEP)
        self.im_end_id = tokenizer.convert_tokens_to_ids(IM_END)

    def add_to_input_ids(
        self,
        input_ids: List[int],
        label: List[int],
        role: str,
        content: str,
    ) -> List[List[int]]:
        if self.apply_chat_template is True:
            x = self.tokenizer.apply_chat_template(
                [
                    {
                        "role": role,
                        "content": content,
                    },
                ],
            )
        else:
            x = self.tokenizer.encode(content)
        input_ids += x
        if role in ["user", "system"]:
            label += [self.ignore_token_id] * len(x)
        else:
            label += x
        return input_ids, label

    def apply(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "conversations" in sample:
            if self.apply_chat_template is True:
                input_ids, label = self.add_to_input_ids(
                    [],
                    [],
                    "system",
                    self.system_prompt,
                )
            else:
                input_ids, label = [], []
            # NOTE: The conversational order must be human -> gpt
            for i in range(len(sample["conversations"])):
                # Some datasets contain incorrect roles
                role = sample["conversations"][i]["from"]
                if role not in self.role_table:
                    logger.warning(
                        f"{role} is not included in {self.role_table}. "
                        "It inferred and replaced role."
                    )
                    if i % 2 == 0:
                        role = "user"
                    else:
                        role = "assistant"
                else:
                    role = self.role_table[role]
                content = sample["conversations"][i]["value"]
                input_ids, label = self.add_to_input_ids(
                    input_ids,
                    label,
                    role,
                    content,
                )
                if self.apply_chat_template is False and role == "assistant":
                    input_ids, label = self.add_to_input_ids(
                        input_ids,
                        label,
                        role,
                        self.eos_token,
                    )
            # NOTE: image_token_id is different from self.image_token_id
            image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
            for i, input_id in enumerate(input_ids):
                if input_id == image_token_id:
                    input_ids[i] = self.image_token_id
                if input_id in [198, self.im_start_id, self.im_sep_id, self.im_end_id]:
                    label[i] = input_id
            sample["input_ids"] = input_ids
            sample["label"] = label
        return sample


class Truncate(Transform):
    def __init__(self, *, max_length: int = 8192, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.max_length = max_length

    def apply(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "input_ids" in sample:
            input_ids = sample["input_ids"]
            end = self.max_length
            input_ids = input_ids[:end]
            sample["input_ids"] = input_ids
        if "label" in sample:
            label = sample["label"]
            end = self.max_length
            label = label[:end]
            sample["label"] = label
        return sample
