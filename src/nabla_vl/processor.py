from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import BoolTensor, FloatTensor, LongTensor
from transformers import AutoImageProcessor, AutoTokenizer, PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PreTokenizedInput, TextInput
from transformers.utils import TensorType

from .config import NablaVLConfig
from .constants import IMAGE_TOKEN, IMAGE_TOKEN_ID, SYSTEM_PROMPT
from .transforms import (
    AddNumTiles,
    AddPatchAttentionMask,
    AddThumbnail,
    AggregateImages,
    AnyRes,
    Factorize,
    Identity,
    Normalize,
    Resize,
    Scale,
    ToTensor,
    Transform,
)


class NablaVLImageProcessor(BaseImageProcessor):
    def __init__(
        self,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        use_anyres: bool = False,
        eos_token: str = "\n",
        factor: int = 14,
        patch_size: int = 14,
        min_shorter_size: int = 70,
        max_longer_size: int = 980,
        thumbnail_size: int = 448,
        min_pixels: int = 196,
        max_pixels: int = 147456,
        max_total_size: int = -1,
        add_thumbnail: bool = False,
        add_marks: bool = False,
        wrap_images: bool = True,
        max_num_tiles: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mean = mean if mean is not None else [0.5, 0.5, 0.5]
        self.std = std if std is not None else [0.5, 0.5, 0.5]
        self.use_anyres = use_anyres
        self.eos_token = eos_token
        self.factor = factor
        self.patch_size = patch_size
        self.min_shorter_size = min_shorter_size
        self.max_longer_size = max_longer_size
        self.thumbnail_size = thumbnail_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_total_size = max_total_size
        self.add_thumbnail = add_thumbnail
        self.add_marks = add_marks
        self.wrap_images = wrap_images
        self.max_num_tiles = max_num_tiles

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

    def get_transforms(self) -> List["Transform"]:
        if hasattr(self, "transforms") is True:
            return self.transforms
        self.transforms = [
            (
                AddThumbnail(
                    factor=self.factor,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                    patch_size=self.patch_size,
                    min_shorter_size=self.min_shorter_size,
                    max_longer_size=self.max_longer_size,
                    thumbnail_size=self.thumbnail_size,
                    add_marks=self.add_marks,
                )
                if self.add_thumbnail is True
                else Identity()
            ),
            # TODO: Process images as numpy array
            ToTensor(dtype=torch.float32),
            AddNumTiles(),
            Factorize(factor=self.factor),
            AddPatchAttentionMask(patch_size=self.patch_size),
            AggregateImages(
                factor=self.factor,
                patch_size=self.patch_size,
                wrap_images=self.wrap_images,
            ),
            (
                AnyRes(
                    factor=self.factor,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                    min_shorter_size=self.min_shorter_size,
                    max_longer_size=self.max_longer_size,
                    max_total_size=self.max_total_size,
                )
                if self.use_anyres is True
                else Resize(
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                    min_shorter_size=self.min_shorter_size,
                    max_longer_size=self.max_longer_size,
                )
            ),
            Scale(scaler=1.0 / 255.0),
            Normalize(
                mean=self.mean,
                std=self.std,
            ),
        ]
        return self.transforms

    def preprocess(
        self,
        images: Union[ImageInput, List[ImageInput]],
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        use_anyres: Optional[bool] = None,
        eos_token: Optional[str] = None,
        factor: Optional[int] = None,
        patch_size: Optional[int] = None,
        min_shorter_size: Optional[int] = None,
        max_longer_size: Optional[int] = None,
        thumbnail_size: Optional[int] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_total_size: Optional[int] = None,
        add_thumbnail: Optional[bool] = None,
        add_marks: Optional[bool] = None,
        wrap_images: Optional[bool] = None,
        max_num_tiles: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        if isinstance(images, list) is False:
            images = [images]
        for i in range(len(images)):
            if images[i] is None:
                images[i] = np.zeros([1, 128, 128, 3], dtype=np.uint8)
            if images[i].ndim == 3:
                images[i] = images[i][np.newaxis, :, :, :]
        batch = ({"images": images},)
        for transform in self.get_transforms():
            batch = transform(batch)
        images = []
        num_images = []
        patch_attention_masks = []
        num_tiles = []
        for sample in batch:
            if "images" in sample:
                images.extend(sample["images"])
                num_images.extend([len(sample["images"])] * len(sample["images"]))
            if "patch_attention_mask" in sample:
                patch_attention_masks.extend(sample["patch_attention_mask"])
            if "num_tiles" in sample:
                num_tiles.extend(sample["num_tiles"])
        images, patch_attention_masks = self.pad_to_max_size(
            images,
            patch_attention_masks,
        )
        return BatchFeature(
            data={
                "images": images,
                "patch_attention_masks": patch_attention_masks,
                "num_tiles": num_tiles,
            },
            tensor_type=return_tensors,
        )


class NablaVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    image_processor: NablaVLImageProcessor
    tokenizer: PreTrainedTokenizer

    def __init__(
        self,
        image_processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[Union[AutoTokenizer, str]] = None,
        chat_template: Optional[str] = None,
    ) -> None:
        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def get_input_ids(
        self,
        instruction: str,
        num_images: int,
    ) -> LongTensor:
        image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        if num_images > 0:
            instruction = f"{IMAGE_TOKEN * num_images}\n{instruction}"
        input_ids = self.tokenizer.apply_chat_template(
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

    def post_process_image_text_to_text(self, generated_outputs):
        # NOTE: generated_outputs could contain image token (-200 for default) which
        # causes an error, so we use .clip(0)
        return self.tokenizer.batch_decode(
            generated_outputs.clip(0),
            skip_special_tokens=True,
        )

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        images: Optional[ImageInput] = None,
        **kwargs,
    ) -> BatchFeature:
        if isinstance(images, list) is False:
            images = [images]
        for i in range(len(images)):
            if isinstance(images[i], Image.Image) is True:
                images[i] = np.array(images[i])[np.newaxis, :, :, :]
        if images[0] is None:
            has_image = False
        else:
            has_image = True
        image_inputs = self.image_processor(images)
        text_inputs = {"input_ids": self.get_input_ids(text, has_image * 1)}
        return BatchFeature(data={"num_images": [1], **text_inputs, **image_inputs})


AutoImageProcessor.register(NablaVLConfig, NablaVLImageProcessor)
