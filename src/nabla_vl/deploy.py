from collections.abc import Mapping
from typing import Final, Optional, Protocol, Sequence, Union

import torch
from torch.nn import Linear, Module, Parameter
from transformers import BatchFeature, PretrainedConfig
from vllm import ModelRegistry
from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.model_executor.models.vision import get_vision_encoder_info
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargs
from vllm.multimodal.parse import (
    ImageSize,
    MultiModalDataItems,
    VideoEmbeddingItems,
    VideoProcessorItems,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs

from nabla_vl.config import NablaVLConfig
from nabla_vl.navit import SiglipVisionConfig, SiglipVisionModel
from nabla_vl.processor import NablaVLProcessor


class NablaVLLikeConfig(Protocol):
    vision_config: Final[PretrainedConfig]
    image_token_index: Final[int]
    vision_feature_select_strategy: Final[str]
    vision_feature_layer: Final[Union[int, list[int]]]


def _get_layer_index(feature_layer_index: int, num_hidden_layers: int) -> int:
    """Given a signed vision feature layer, get the number of hidden layers
    needed to leverage it.

    Args:
        feature_layer_index: Index of a required layer in the visual encoder.
        num_hidden_layers: The total number of hidden layers in the visual
            encoder.
    """
    if feature_layer_index < 0:
        return num_hidden_layers + feature_layer_index + 1
    return feature_layer_index


def _get_num_hidden_layers(hf_config: NablaVLLikeConfig) -> int:
    """Determine the number of hidden layers to initialize up to in the
    visual encoder.

    Args:
        hf_config: Model config with vision feature layer(s).
    """
    feature_layers = hf_config.vision_feature_layer
    num_hidden_layers = hf_config.vision_config.num_hidden_layers
    # If we have one feature layer, initialize up to that layer
    if isinstance(feature_layers, int):
        return _get_layer_index(feature_layers, num_hidden_layers)
    # If we have multiple feature layers, initialize up to the deepest one
    elif isinstance(feature_layers, (list, tuple)):
        return max(_get_layer_index(idx, num_hidden_layers) for idx in feature_layers)
    raise TypeError(
        f"vision_layer_feature type: {type(feature_layers)}" " is not supported"
    )


def init_vision_tower_for_nabla_vl(
    hf_config: NablaVLLikeConfig,
    quant_config: Optional[QuantizationConfig],
    *,
    require_post_norm: Optional[bool] = None,
    prefix: str = "",
) -> SiglipVisionModel:
    vision_config = hf_config.vision_config
    # Initialize the vision tower only up to the deepest required feature layer
    num_hidden_layers = _get_num_hidden_layers(hf_config)
    if isinstance(vision_config, SiglipVisionConfig):
        return SiglipVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )
    else:
        msg = f"Unsupported vision config: {type(vision_config)}"
        raise NotImplementedError(msg)


class NablaVLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> NablaVLLikeConfig:
        return self.ctx.get_hf_config(NablaVLConfig)

    def get_vision_encoder_info(self):
        return get_vision_encoder_info(self.get_hf_config())

    def get_hf_processor(self, **kwargs: object):
        hf_processor = self.ctx.get_hf_processor(NablaVLProcessor, **kwargs)
        # In case patch_size is omitted from `processor_config.json`
        # e.g. for E5-V: https://huggingface.co/royokong/e5-v
        if hf_processor.patch_size is None:
            patch_size = self.get_vision_encoder_info().get_patch_size()
            hf_processor.patch_size = patch_size
        return hf_processor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}

    def _apply_feature_select_strategy(
        self,
        strategy: str,
        encoder_num_image_tokens: int,
    ) -> int:
        if strategy == "default":
            return encoder_num_image_tokens - 1
        if strategy == "full":
            return encoder_num_image_tokens

        msg = f"Unexpected feature select strategy: {strategy!r}"
        raise NotImplementedError(msg)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_info = self.get_vision_encoder_info()

        return self._apply_feature_select_strategy(
            hf_config.vision_feature_select_strategy,
            vision_encoder_info.get_num_image_tokens(
                image_width=image_width,
                image_height=image_height,
            ),
        )

    def get_image_size_with_most_features(self) -> ImageSize:
        vision_encoder_info = self.get_vision_encoder_info()
        width = height = vision_encoder_info.get_image_size()
        return ImageSize(width=width, height=height)

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
        )


class NablaVLMultiModalProcessor(BaseMultiModalProcessor[NablaVLProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_sizes=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
            pixel_values_videos=MultiModalFieldConfig.batched("video"),
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        videos = mm_data.pop("videos", [])
        assert isinstance(videos, list)
        if not videos:
            return super()._call_hf_processor(
                prompt=prompt,
                mm_data=mm_data,
                mm_kwargs=mm_kwargs,
            )
        # LLaVA-OneVision processor doesn't support multiple videos
        # with different sizes when converting back to tensors
        # So, we process each component separately
        # NOTE: No prompt replacement is applied in this case
        processor = self.info.get_hf_processor()
        image_token = processor.image_token
        video_token = processor.video_token
        text_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data={},
            mm_kwargs=mm_kwargs,
        )
        images = mm_data.pop("images", [])
        assert isinstance(images, list)
        if images:
            processor_outputs = super()._call_hf_processor(
                prompt=image_token * len(images),
                mm_data={"images": images},
                mm_kwargs=mm_kwargs,
            )
            image_outputs = {
                k: v
                for k, v in processor_outputs.items()
                if k in ("pixel_values", "image_sizes")
            }
        else:
            image_outputs = {}
        pixel_values_videos = []
        for video in videos:
            item_outputs = super()._call_hf_processor(
                prompt=video_token,
                mm_data={"videos": video},
                mm_kwargs=mm_kwargs,
            )
            pixel_values_videos.append(item_outputs["pixel_values_videos"][0])
        video_outputs = {"pixel_values_videos": pixel_values_videos}
        combined_outputs = dict(
            text_outputs,
            **image_outputs,
            **video_outputs,
        )
        return BatchFeature(combined_outputs)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> bool:
        base_result = super()._hf_processor_applies_updates(
            prompt_text=prompt_text,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )
        return base_result and mm_items.get_count("video", strict=False) == 0

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        image_repls = super()._get_prompt_updates(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            out_mm_kwargs=out_mm_kwargs,
        )
        hf_config = self.info.get_hf_config()
        video_token_id = hf_config.video_token_index

        def get_video_replacement(item_idx: int):
            videos = mm_items.get_items(
                "video", (VideoEmbeddingItems, VideoProcessorItems)
            )
            if isinstance(videos, VideoEmbeddingItems):
                num_video_tokens = videos.get_feature_size(item_idx)
            else:
                image_size = videos.get_frame_size(item_idx)
                num_video_tokens = self.info.get_num_video_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    num_frames=videos.get_num_frames(item_idx),
                )
            return [video_token_id] * num_video_tokens

        return [
            *image_repls,
            PromptReplacement(
                modality="video",
                target=[video_token_id],
                replacement=get_video_replacement,
            ),
        ]


class NablaVLDummyInputsBuilder(BaseDummyInputsBuilder[NablaVLProcessingInfo]):
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        processor = self.info.get_hf_processor()
        image_token = processor.image_token
        video_token = processor.video_token
        target_width, target_height = self.info.get_image_size_with_most_features()
        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts
        )
        mm_data = {
            "image": self._get_dummy_images(
                width=target_width, height=target_height, num_images=num_images
            ),
            "video": self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=target_num_frames,
                num_videos=num_videos,
            ),
        }
        return ProcessorInputs(
            prompt_text=image_token * num_images + video_token * num_videos,
            mm_data=mm_data,
        )


class NablaVLMultiModalProjector(Module):

    def __init__(self, config: NablaVLConfig):
        super().__init__()
        self.linear_1 = Linear(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = get_act_fn(config.projector_hidden_act)
        self.linear_2 = Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    NablaVLMultiModalProcessor,
    info=NablaVLProcessingInfo,
    dummy_inputs=NablaVLDummyInputsBuilder,
)
class NablaVLForCausalLM(Module, SupportsMultiModal, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        # Initialize the vision tower only up to the required feature layer
        self.vision_tower = init_vision_tower_for_nabla_vl(
            config,
            quant_config,
            require_post_norm=False,
            prefix=maybe_prefix(prefix, "vision_tower"),
        )
        self.multi_modal_projector = NablaVLMultiModalProjector(config)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.image_newline = Parameter(torch.empty(config.text_config.hidden_size))
        self.make_empty_intermediate_tensors = (
            self.language_model.model.make_empty_intermediate_tensors
        )


ModelRegistry.register_model("NablaVLForCausalLM", NablaVLForCausalLM)


if __name__ == "__main__":
    from vllm import LLM

    model = LLM(model="nablasinc/NABLA-VL")
    print(model)
