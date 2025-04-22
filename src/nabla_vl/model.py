import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from deepspeed.utils import logger
from torch import BoolTensor, FloatTensor, LongTensor
from torch.nn import GELU, Identity, LayerNorm, Linear, Parameter, Sequential
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    Phi3ForCausalLM,
    Phi3Model,
    PreTrainedTokenizer,
    TrainingArguments,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import NablaVLConfig
from .constants import IGNORE_TOKEN_ID, IMAGE_TOKEN_ID
from .navit import SiglipVisionModel  # type: ignore
from .utils import get_dtype_by_args, get_total_params, make_inputs_require_grad


def build_model(
    args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> "NablaVLForCausalLM":
    logger.info("Initializing model")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    # Values used to initialize model
    config.enable_lazy_init = args.enable_lazy_init
    config.vision_tower_name = args.vision_tower_name
    config.enable_flash_attention_2 = args.enable_flash_attention_2
    config.max_length = args.max_length
    config.padding_side = tokenizer.padding_side
    config.max_vision_tokens = args.max_vision_tokens
    config.use_new_line_token = args.use_new_line_token
    config.neftune_alpha = args.neftune_alpha
    config.num_registers = args.num_registers
    # Values used to initialize data_pipeline
    config.apply_chat_template = args.apply_chat_template
    config.eos_token = args.eos_token
    config.fp16 = args.fp16
    config.bf16 = args.bf16
    config.factor = args.factor
    config.patch_size = args.patch_size
    config.min_pixels = args.min_pixels
    config.max_pixels = args.max_pixels
    config.min_shorter_size = args.min_shorter_size
    config.max_longer_size = args.max_longer_size
    config.max_total_size = args.max_total_size
    config.use_anyres = args.use_anyres
    config.add_thumbnail = args.add_thumbnail
    config.add_marks = args.add_marks
    config.thumbnail_size = args.thumbnail_size
    config.normalize_type = args.normalize_type
    config.wrap_images = args.wrap_images
    config.max_num_tiles = args.max_num_tiles
    dtype = get_dtype_by_args(args)
    model = NablaVLForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    if args.enable_lazy_init is True:
        logger.info("Initializing vision tower in lazy way")
        model.prepare_lazy_init()
        model.vision_tower.to(dtype=dtype)
        model.projector.to(dtype=dtype)
    model.config.use_cache = False
    if args.gradient_checkpointing is True:
        if hasattr(model, "enable_input_require_grads") is True:
            model.enable_input_require_grads()
        else:
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    model.requires_grad_(False)
    model.image_start_tag.requires_grad_(True)
    model.image_end_tag.requires_grad_(True)
    if args.freeze_vision_tower is False:
        for p in model.vision_tower.parameters():
            p.requires_grad = True
    if args.freeze_projector is False:
        for p in model.projector.parameters():
            p.requires_grad = True
    if args.freeze_embedding is False:
        for p in model.model.embed_tokens.parameters():
            p.requires_grad = True
    if args.freeze_llm is False:
        for p in model.model.parameters():
            p.requires_grad = True
    if args.freeze_lm_head is False:
        for p in model.lm_head.parameters():
            p.requires_grad = True
    if args.num_registers > 0:
        model.registers.requires_grad_(True)
    if args.use_new_line_token is True:
        model.new_line_tag.requires_grad_(True)
        model.new_column_tag.requires_grad_(True)
    total_params, trainable_params = get_total_params(model)
    logger.info(f"total params: {total_params}, trainable params: {trainable_params}")
    logger.info(f"{trainable_params / total_params * 100.0:.2f}% params are trainable")
    return model


def split_input_ids(
    input_ids: LongTensor,
    label: LongTensor,
    attention_mask: LongTensor,
    *,
    image_token_id: int = IMAGE_TOKEN_ID,
) -> Tuple[LongTensor, LongTensor, List[int]]:
    input_ids = input_ids[attention_mask]
    label = label[attention_mask]
    visual_token_idxs = (
        [-1]
        + torch.where(input_ids == image_token_id)[0].tolist()
        + [input_ids.shape[0]]
    )
    new_input_ids = []
    new_labels = []
    split_sections = []
    for i in range(len(visual_token_idxs) - 1):
        start = visual_token_idxs[i] + 1
        end = visual_token_idxs[i + 1]
        new_input_ids.append(input_ids[start:end])
        new_labels.append(label[start:end])
        split_sections.append(end - start)
    new_input_ids = torch.cat(new_input_ids)
    new_labels = torch.cat(new_labels)
    return new_input_ids, new_labels, split_sections


def insert_image_features(
    input_embeds: FloatTensor,
    labels: LongTensor,
    image_features: List[FloatTensor],
    split_sections: List[int],
    num_images: int,
    *,
    ignore_token_id: int = IGNORE_TOKEN_ID,
) -> Tuple[FloatTensor, LongTensor]:
    input_embeds = torch.split(input_embeds, split_sections, 0)
    labels = torch.split(labels, split_sections, 0)
    new_input_embeds = []
    new_labels = []
    for i in range(num_images + 1):
        new_input_embeds.append(input_embeds[i])
        new_labels.append(labels[i])
        if i < num_images:
            new_input_embeds.append(image_features[i])
            ignore_ids = torch.full(
                (image_features[i].size(0),),
                ignore_token_id,
                dtype=torch.long,
                device=labels[i].device,
            )
            new_labels.append(ignore_ids)
    new_input_embeds = torch.cat(new_input_embeds)
    new_labels = torch.cat(new_labels)
    return new_input_embeds, new_labels


# TODO: Correct function name
def adjust_inputs_and_labels(
    input_embeds: List[FloatTensor],
    attention_mask: BoolTensor,
    labels: List[LongTensor],
    position_ids: LongTensor,
    padding_side: str = "right",
    *,
    ignore_token_id: int = IGNORE_TOKEN_ID,
):
    n = len(input_embeds)
    max_len = max(x.shape[0] for x in input_embeds)
    new_input_embeds = []
    new_attention_mask = torch.zeros(
        (n, max_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    new_labels = torch.full(
        (n, max_len),
        ignore_token_id,
        dtype=labels[0].dtype,
        device=labels[0].device,
    )
    new_position_ids = torch.zeros(
        (n, max_len),
        dtype=position_ids.dtype,
        device=position_ids.device,
    )
    for i in range(n):
        l = input_embeds[i].size(0)  # noqa
        if padding_side == "left":
            new_input_embeds.append(
                torch.cat(
                    [
                        torch.zeros(
                            (max_len - l, input_embeds[i].size(1)),
                            dtype=input_embeds[i].dtype,
                            device=input_embeds[i].device,
                        ),
                        input_embeds[i],
                    ],
                    0,
                )
            )
            if l > 0:
                new_attention_mask[i, -l:] = True
                new_labels[i, -l:] = labels[i]
                new_position_ids[i, -l:] = torch.arange(
                    0,
                    l,
                    dtype=position_ids.dtype,
                    device=position_ids.device,
                )
        else:
            new_input_embeds.append(
                torch.cat(
                    [
                        input_embeds[i],
                        torch.zeros(
                            (max_len - l, input_embeds[i].size(1)),
                            dtype=input_embeds[i].dtype,
                            device=input_embeds[i].device,
                        ),
                    ],
                    0,
                )
            )
            if l > 0:
                new_attention_mask[i, :l] = True
                new_labels[i, :l] = labels[i]
                new_position_ids[i, :l] = torch.arange(
                    0,
                    l,
                    dtype=position_ids.dtype,
                    device=position_ids.device,
                )
    new_input_embeds = torch.stack(new_input_embeds)
    return new_input_embeds, new_attention_mask, new_labels, new_position_ids


class NablaVLForCausalLM(Phi3ForCausalLM):
    config_class = NablaVLConfig

    def __init__(self, config: NablaVLConfig = None) -> None:
        super().__init__(config)
        self.model = Phi3Model(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.enable_lazy_init is False:
            self.prepare_lazy_init()
        self.post_init()

    def prepare_lazy_init(self) -> None:
        self.image_start_tag = Parameter(torch.randn(1, self.config.hidden_size))
        self.image_end_tag = Parameter(torch.randn(1, self.config.hidden_size))
        if self.config.num_registers > 0:
            self.registers = Parameter(
                torch.randn(self.config.num_registers, self.config.hidden_size)
            )
        self.vision_tower = SiglipVisionModel.from_pretrained(
            self.config.vision_tower_name,
            _flash_attn_2_enabled=self.config.enable_flash_attention_2,
        )
        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = Identity()
        self.projector = Sequential(
            LayerNorm(
                self.vision_tower.config.hidden_size,
                self.vision_tower.config.layer_norm_eps,
            ),
            Linear(self.vision_tower.config.hidden_size, self.config.hidden_size),
            GELU(),
            Linear(self.config.hidden_size, self.config.hidden_size),
        )
        if self.config.use_new_line_token is True:
            self.new_line_tag = Parameter(torch.randn(1, self.config.hidden_size, 1, 1))
            self.new_column_tag = Parameter(torch.randn(1, self.config.hidden_size))

    def get_num_vision_tokens(self, patch_attention_masks: BoolTensor) -> int:
        return patch_attention_masks.sum().item()

    def get_max_vision_tokens(self) -> int:
        if self.training is False:
            return self.config.max_vision_tokens
        # hw = math.sqrt(self.config.max_vision_tokens)
        # scale = random.uniform(0.1, 1.0)
        # return int((scale * hw) ** 2)
        return self.config.max_vision_tokens

    def convert_nmc_to_nchw(
        self,
        features: FloatTensor,
        patch_attention_mask: BoolTensor,
        h: int,
        w: int,
        num_rows: int,
        num_cols: int,
        patch_size: int,
    ) -> Tuple[FloatTensor, BoolTensor]:
        """Convert a sequence of features to a feature map

        How it works:
        1. Reshape features to a feature map of a tile
        2. Merge feature maps into a single feature map

        Note
        ----
        It applies the same operation to patch_attention_mask.

        Parameters
        ----------
        features : FloatTensor
            A sequence of features (n x m x c) where n is the number of tiles and m is
            the number of vision tokens
        patch_attention_mask : BoolTensor
            A patch attention mask (n x h x w) where n is the number of tiles
        h : int
            The height of a single tile
        w : int
            The width of a single tile
        patch_size : int
            The size of a patch (e.g., 14)

        Returns
        -------
        Tuple[FloatTensor, BoolTensor]
            A feature map (1 x c x h x w) and a patch attention mask (1 x h x w)
        """
        # Merge patches
        n, _, c = features.size()
        features = features.reshape(n, int(h / patch_size), int(w / patch_size), c)
        # FIXME: Change name to avoid name conflicts (args)
        # Merge features into a feature map
        _, h, w, _ = features.size()
        assert n == num_rows * num_cols
        features = features.reshape(num_rows, num_cols, h, w, c)
        patch_attention_mask = patch_attention_mask.reshape(num_rows, num_cols, h, w)
        features = features.permute(0, 2, 1, 3, 4)
        patch_attention_mask = patch_attention_mask.permute(0, 2, 1, 3)
        features = features.reshape(num_rows * h, num_cols * w, c)
        patch_attention_mask = patch_attention_mask.reshape(num_rows * h, num_cols * w)
        # h_mask = patch_attention_mask[:, 0].sum()
        # w_mask = patch_attention_mask[0].sum()
        # features = features[None, :h_mask, :w_mask, :]
        features = features[None, :, :, :]
        patch_attention_mask = patch_attention_mask[None, :, :]
        # Resize feature map if the number of vision tokens is too much
        features = features.permute(0, 3, 1, 2)
        return features, patch_attention_mask

    def merge_and_resize(
        self,
        features: FloatTensor,
        patch_attention_mask: BoolTensor,
        h: int,
        w: int,
        num_rows: int,
        num_cols: int,
        patch_size: int,
        num_total_vision_tokens: int,
    ) -> FloatTensor:
        """Resize a feature map if the number of vision tokens is too much　and strip
        off unnecessary tokens while adding special tokens like image start and image
        end tags

        How it works:
        1. Resize a feature map if the number of vision tokens is too much
        2. Strip off unnecessary tokens
        3. Add special tokens like image start and image end tags

        Parameters
        ----------
        features : FloatTensor
            A feature map (1 x c x h x w)
        patch_attention_mask : BoolTensor
            A patch attention mask (1 x h x w)
        h : int
            Not used
        w : int
            Not used
        num_rows : int
            Not used
        num_cols : int
            Not used
        patch_size : int
            Not used
        num_total_vision_tokens : int
            The total number of vision tokens per group

        Returns
        -------
        FloatTensor
            A single sequence of features (l x c) where l is the number of vision tokens
        """
        num_vision_tokens = self.get_num_vision_tokens(patch_attention_mask)
        max_vision_tokens = self.config.max_vision_tokens
        if num_vision_tokens > max_vision_tokens:
            scale = num_vision_tokens / max_vision_tokens
            # More vision tokens are assigned to larger image
            scale *= num_total_vision_tokens / num_vision_tokens
            scale = math.sqrt(scale)
            if scale > 1.1:
                new_h, new_w = features.size()[-2:]
                features = F.interpolate(
                    features,
                    (
                        max(1, int(new_h / scale)),
                        max(1, int(new_w / scale)),
                    ),
                    mode="bilinear",
                )
                patch_attention_mask = patch_attention_mask.type(torch.uint8)
                patch_attention_mask = patch_attention_mask[:, None, :, :]
                patch_attention_mask = F.interpolate(
                    patch_attention_mask,
                    (
                        max(1, int(new_h / scale)),
                        max(1, int(new_w / scale)),
                    ),
                    mode="nearest",
                )
                patch_attention_mask = patch_attention_mask.type(torch.bool)
                patch_attention_mask = patch_attention_mask.squeeze(1)
        if self.config.use_new_line_token is True:
            n, _, h, w = features.size()
            features = torch.cat(
                [
                    features,
                    self.new_line_tag.repeat(n, 1, h, 1),
                ],
                3,
            )
            patch_attention_mask = torch.cat(
                [
                    patch_attention_mask,
                    torch.ones(
                        (
                            n,
                            h,
                            1,
                        ),
                        dtype=torch.bool,
                        device=patch_attention_mask.device,
                    ),
                ],
                2,
            )
        # NCHW -> MC
        features = features.flatten(2).permute(0, 2, 1)
        # Flatten an image, multiple images or a video to a single sequence
        features = features.flatten(0, 1)
        patch_attention_mask = patch_attention_mask.flatten()
        features = features[patch_attention_mask]
        # Add new_column_tag at the end of each row
        if self.config.use_new_line_token is True:
            features = torch.cat(
                [
                    self.new_column_tag.repeat(w, 1),
                    features,
                ],
                0,
            )
        # Add image start and image end tags
        features = torch.cat(
            [
                self.image_start_tag,
                features,
                self.image_end_tag,
            ],
            0,
        )
        return features

    def encode_images(
        self,
        images: List[FloatTensor],
        num_images: List[int],
        patch_attention_masks: List[BoolTensor],
        num_tiles: List[List[int]],
    ) -> List[FloatTensor]:
        """Convert a list of images (n x c x h x w) into a list of features (l x hidden
        dim)

        How it works:
        1. Encode images using imege encoder
        2. Construct a feature map from a sequence of features by referring to num_tiles
        3. Resize a feature map if the number of vision tokens is too much
        4. Build a single sequence from a feature map

        Note
        ----
        It's assumed each image and patch attention mask in the list has the same shape.

        Parameters
        ----------
        images : List[FloatTensor]
            A list of images (n x c x h x w) where n is the number of tiles for a
            single image
        num_images : List[int]
            A list of the number of images per a sample
        patch_attention_masks : List[BoolTensor]
            A list of patch attention masks (n x h x w) where n is the number of tiles
            for a single image
        num_tiles : List[List[int]]
            A list of the number of rows and columns for each image

        Returns
        -------
        List[FloatTensor]
            A list of features (l x hidden dim) where l is the number of vision tokens
        """
        # NOTE: It's assumed each image in list has the same shape
        h, w = images[0].size()[-2:]
        split_sections = []
        for image in images:
            num_patches = image.size(0)
            split_sections.append(num_patches)
        images = torch.cat(images)
        patch_attention_masks = torch.cat(patch_attention_masks)
        features = self.vision_tower.vision_model(
            pixel_values=images,
            patch_attention_mask=patch_attention_masks,
            output_hidden_states=True,
        ).hidden_states[-1]
        features = self.projector(features)
        features = list(torch.split(features, split_sections))
        patch_attention_masks = list(torch.split(patch_attention_masks, split_sections))
        # Convert tile features to a feature map
        # After this operation, features[i] is a feature map of i-th image
        for i in range(len(split_sections)):
            num_rows, num_cols = num_tiles[i]
            features[i], patch_attention_masks[i] = self.convert_nmc_to_nchw(
                features[i],
                patch_attention_masks[i],
                h,
                w,
                num_rows,
                num_cols,
                self.vision_tower.config.patch_size,
            )
        # Get the total number of vision tokens per group
        num_total_vision_tokens = []
        counter = 0
        while counter < len(split_sections):
            num_vision_tokens = 0
            n = num_images[counter]
            for i in range(n):
                num_vision_tokens += self.get_num_vision_tokens(
                    patch_attention_masks[i]
                )
            num_total_vision_tokens += [num_vision_tokens] * n
            counter += n
        # Resize a feature map if the number of vision tokens is too much
        # And strip off unnecessary tokens
        for i in range(len(split_sections)):
            num_rows, num_cols = num_tiles[i]
            features[i] = self.merge_and_resize(
                features[i],
                patch_attention_masks[i],
                h,
                w,
                num_rows,
                num_cols,
                self.vision_tower.config.patch_size,
                num_total_vision_tokens[i],
            )
        return features

    def add_noise_to_inputs_embeds(
        self,
        input_embeds: FloatTensor,
        attention_mask: BoolTensor,
    ) -> FloatTensor:
        input_lengths = torch.sum(attention_mask, 1).to(input_embeds)
        noise_ = torch.zeros_like(input_embeds).uniform_(-1, 1)
        delta = noise_ * attention_mask.unsqueeze(2)
        dims = input_lengths * input_embeds.size(-1)
        mag = self.config.neftune_alpha / torch.sqrt(dims)
        delta = (delta * mag.view(-1, 1, 1)).detach()
        input_embeds = delta + input_embeds
        return input_embeds

    def prepare_inputs_for_forward(
        self,
        input_ids: LongTensor,
        *,
        attention_mask: Optional[LongTensor] = None,
        patch_attention_masks: Optional[BoolTensor] = None,
        position_ids: Optional[LongTensor] = None,
        past_key_values: Optional[List[FloatTensor]] = None,
        labels: Optional[LongTensor] = None,
        images: Optional[List[FloatTensor]] = None,
        # FIXME: Name conflicts with one of vars in this method!
        num_images: Optional[List[int]] = None,
        num_tiles: Optional[List[List[int]]] = None,
        ignore_token_id: int = IGNORE_TOKEN_ID,
        image_token_id: int = IMAGE_TOKEN_ID,
    ):
        # TODO: Allow to take single token input during inference
        if images is None or len(images) == 0 or input_ids.shape[1] == 1:
            return (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                None,
                labels,
            )
        image_features = self.encode_images(
            images,
            num_images,
            patch_attention_masks,
            num_tiles,
        )
        skip_labels = labels
        skip_position_ids = position_ids
        skip_attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0,
                input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            )
        if labels is None:
            labels = torch.full_like(input_ids, ignore_token_id)
        input_embeds = []
        new_labels = []
        image_id = 0
        for i in range(input_ids.size(0)):
            # TODO: Rename it
            num_images = (input_ids[i] == image_token_id).sum().item()
            # Text-only sequence
            if num_images == 0:
                max_length = getattr(self.config, "max_length", None)
                input_ids_per_sample = input_ids[i][attention_mask[i]]
                input_embeds_per_sample = self.model.embed_tokens(input_ids_per_sample)
                labels_per_sample = labels[i][attention_mask[i]]
                # Add registers
                if self.config.num_registers > 0:
                    input_embeds_per_sample = torch.cat(
                        [
                            self.registers,
                            image_features[image_id][0:0],
                            input_embeds_per_sample,
                        ],
                        0,
                    )
                    labels_per_sample = torch.cat(
                        [
                            torch.full(
                                (self.config.num_registers,),
                                ignore_token_id,
                                dtype=torch.long,
                                device=labels_per_sample.device,
                            ),
                            labels_per_sample,
                        ],
                        0,
                    )
                input_embeds.append(input_embeds_per_sample[:max_length])
                new_labels.append(labels_per_sample[:max_length])
                # Skip dummy image features
                image_id += 1
                continue
            input_ids_per_sample, labels_per_sample, split_sections = split_input_ids(
                input_ids[i],
                labels[i],
                attention_mask[i],
            )
            input_embeds_per_sample = self.model.embed_tokens(input_ids_per_sample)
            start = image_id
            end = start + num_images
            image_id += num_images
            input_embeds_per_sample, labels_per_sample = insert_image_features(
                input_embeds_per_sample,
                labels_per_sample,
                image_features[start:end],
                split_sections,
                num_images,
                ignore_token_id=ignore_token_id,
            )
            # Add registers
            if self.config.num_registers > 0:
                input_embeds_per_sample = torch.cat(
                    [
                        self.registers,
                        input_embeds_per_sample,
                    ],
                    0,
                )
                labels_per_sample = torch.cat(
                    [
                        torch.full(
                            (self.config.num_registers,),
                            ignore_token_id,
                            dtype=torch.long,
                            device=labels_per_sample.device,
                        ),
                        labels_per_sample,
                    ],
                    0,
                )
            max_length = getattr(self.config, "max_length", None)
            input_embeds_per_sample = input_embeds_per_sample[:max_length]
            labels_per_sample = labels_per_sample[:max_length]
            input_embeds.append(input_embeds_per_sample)
            new_labels.append(labels_per_sample)
        input_embeds, attention_mask, labels, position_ids = adjust_inputs_and_labels(
            input_embeds,
            attention_mask,
            new_labels,
            position_ids,
            self.config.padding_side,
            ignore_token_id=ignore_token_id,
        )
        # Custom NEFTune implementation
        if self.config.neftune_alpha is not None and self.training is True:
            input_embeds = self.add_noise_to_inputs_embeds(
                input_embeds,
                attention_mask,
            )
        if skip_labels is None:
            labels = None
        if skip_attention_mask is None:
            attention_mask = None
        if skip_position_ids is None:
            position_ids = None
        return (
            None,
            attention_mask,
            position_ids,
            past_key_values,
            input_embeds,
            labels,
        )

    def forward(
        self,
        *,
        input_ids: Optional[LongTensor] = None,
        attention_mask: Optional[LongTensor] = None,
        patch_attention_masks: Optional[BoolTensor] = None,
        position_ids: Optional[LongTensor] = None,
        past_key_values: Optional[List[FloatTensor]] = None,
        inputs_embeds: Optional[FloatTensor] = None,
        labels: Optional[LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[List[FloatTensor]] = None,
        num_images: Optional[List[int]] = None,
        num_tiles: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[LongTensor] = None,
        num_logits_to_keep: int = 1,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_for_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                patch_attention_masks=patch_attention_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                images=images,
                num_images=num_images,
                num_tiles=num_tiles,
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[LongTensor] = None,
        past_key_values: Optional[List[FloatTensor]] = None,
        attention_mask: Optional[LongTensor] = None,
        inputs_embeds: Optional[FloatTensor] = None,
        cache_position: Optional[LongTensor] = None,
        position_ids: Optional[LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        images = kwargs.pop("images", None)
        num_images = kwargs.pop("num_images", None)
        num_tiles = kwargs.pop("num_tiles", None)
        patch_attention_masks = kwargs.pop("patch_attention_masks", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if num_images is not None:
            inputs["num_images"] = num_images
        if num_tiles is not None:
            inputs["num_tiles"] = num_tiles
        if patch_attention_masks is not None:
            inputs["patch_attention_masks"] = patch_attention_masks
        return inputs

    @torch.no_grad()
    def generate(
        self,
        *,
        input_ids: Optional[LongTensor] = None,
        attention_mask: Optional[LongTensor] = None,
        patch_attention_masks: Optional[BoolTensor] = None,
        position_ids: Optional[LongTensor] = None,
        past_key_values: Optional[List[FloatTensor]] = None,
        inputs_embeds: Optional[FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[List[FloatTensor]] = None,
        num_images: Optional[List[int]] = None,
        num_tiles: Optional[List[List[int]]] = None,
        cache_position: Optional[LongTensor] = None,
        **kwargs,
    ) -> GenerateOutput:
        if images is not None:
            (
                _,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_for_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                patch_attention_masks=patch_attention_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                images=images,
                num_images=num_images,
                num_tiles=num_tiles,
            )
        else:
            inputs_embeds = self.model.embed_tokens(input_ids)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            past_key_values=past_key_values,
            **kwargs,
        )


AutoModelForImageTextToText.register(NablaVLConfig, NablaVLForCausalLM)
