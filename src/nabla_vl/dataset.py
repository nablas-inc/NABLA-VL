import os
import random
from typing import Any, Dict, List

import numpy as np
from deepspeed.utils import logger
from torch.utils.data import Dataset
from transformers import TrainingArguments
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from .constants import IMAGE_TOKEN
from .io import load_image, load_json, load_video
from .registry import DATASETS


def build_dataset(args: TrainingArguments) -> Dataset:
    logger.info("Initializing dataset")
    dataset = DATASETS[args.dataset_name].from_args(args)
    logger.info(f"len(dataset)={len(dataset)}")
    return dataset


@DATASETS.register_cls()
class NablaVLDataset(Dataset):
    def __init__(
        self,
        annotation_paths: Dict[str, float],
        image_dir: str,
        *,
        factor: int = 70,
        max_longer_size: int = 980,
        max_total_size: int = -1,
        patch_size: int = 14,
        remove_instructions: bool = False,
        image_token: str = IMAGE_TOKEN,
        fixed_num_vision_tokens_per_image: int = 128,
        dummy_image_size: List[int] = [1, 128, 128, 3],
        max_num_samples: int = -1,
        max_num_images_per_sample: int = -1,
        max_num_characters: int = -1,
        max_num_frames: int = 32,
    ) -> None:
        self.annotation_paths = annotation_paths
        self.image_dir = image_dir
        self.factor = factor
        self.max_longer_size = max_longer_size
        self.max_total_size = max_total_size
        self.patch_size = patch_size
        self.remove_instructions = remove_instructions
        self.image_token = image_token
        self.fixed_num_vision_tokens_per_image = fixed_num_vision_tokens_per_image
        self.dummy_image_size = dummy_image_size
        self.max_num_samples = max_num_samples
        self.max_num_images_per_sample = max_num_images_per_sample
        self.max_num_characters = max_num_characters
        self.max_num_frames = max_num_frames
        self.annotation = []
        for annotation_path, ratio in self.annotation_paths.items():
            logger.info(f"Loading {annotation_path}")
            # TODO: Rename it
            ann = load_json(annotation_path)
            n = int(ratio * len(ann))
            # Constrain the bias of one majority dataset
            if self.max_num_samples > 0 and n > self.max_num_samples:
                n = self.max_num_samples
            ann = ann[:n]
            logger.info(f"num samples={len(ann)} ({annotation_path})")
            self.annotation.extend([(x, annotation_path) for x in ann])

    @classmethod
    def from_args(cls, args: TrainingArguments) -> "NablaVLDataset":
        return cls(
            factor=args.factor,
            max_longer_size=args.max_longer_size,
            max_total_size=args.max_total_size,
            patch_size=args.patch_size,
            annotation_paths=args.annotation_paths,
            image_dir=args.image_dir,
            remove_instructions=args.remove_instructions,
            max_num_samples=args.max_num_samples,
            max_num_images_per_sample=args.max_num_images_per_sample,
            max_num_characters=args.max_num_characters,
            max_num_frames=args.max_num_frames,
        )

    def get_lengths(self) -> List[int]:
        lengths = []
        for x in self.annotation:
            length = 0
            if "conversations" in x:
                conversations = x["conversations"]
                for i in range(len(conversations)):
                    if self.remove_instructions is True:
                        if conversations[i]["from"] == "human":
                            length += 1
                            continue
                    length += len(conversations[i]["value"].split())
            if "image" not in x:
                length = -length
            lengths.append(length)
        return lengths

    def __len__(self) -> int:
        return len(self.annotation)

    def get_num_image_tokens(self, conversations: List[Dict[str, str]]) -> int:
        num_image_tokens = 0
        for i in range(len(conversations)):
            # NOTE: Instruction or response could have more than 1 image token
            num_image_tokens += conversations[i]["value"].count(self.image_token)
        return num_image_tokens

    def check_images_and_conversations_format(
        self,
        images: List[np.ndarray],
        conversations: List[Dict[str, str]],
        offset: int = 0,
    ) -> bool:
        num_images = len(images) - offset
        num_image_tokens = self.get_num_image_tokens(conversations)
        return num_images >= num_image_tokens

    def check_image_shape(self, image: np.ndarray) -> bool:
        h, w = image.shape[:-1]
        try:
            h, w = smart_resize(
                h,
                w,
                factor=self.factor,
                min_pixels=0,
                max_pixels=float("inf"),
            )
        # smart_resize raises ValueError when image or aspect ratio is too small
        except ValueError:
            return False
        if self.max_total_size > 0:
            if max(h, w) > self.max_total_size:
                scale = self.max_total_size / max(h, w)
                new_h, new_w = int(scale * h), int(scale * w)
                if min(new_h, new_w) < self.factor:
                    return False
        return True

    def __getitem__(self, i: int) -> Dict[str, Any]:
        x, annotation_path = self.annotation[i]
        images = []
        if "image" in x:
            has_image = True
            filenames = x["image"]
            # Not a list if it's single-image dataset
            if isinstance(filenames, list) is False:
                filenames = [filenames]
            # TODO: Rename it
            image_paths = [os.path.join(self.image_dir, name) for name in filenames]
            for image_path in image_paths:
                try:
                    image = np.array(load_image(image_path), dtype=np.uint8)
                except Exception:
                    logger.warning(f"detected corrupted image: {image_path}")
                    return self[random.randint(0, len(self) - 1)]
                if self.check_image_shape(image) is False:
                    logger.warning("detected anomaly shaped image and skipped it")
                    return self[random.randint(0, len(self) - 1)]
                image = image[np.newaxis, :, :, :]
                images.append(image)
            if len(images) == 0:
                logger.warning(f"detected empty image: {image_paths}")
                return self[random.randint(0, len(self) - 1)]
        elif "video" in x:
            has_image = True
            filenames = x["video"]
            if isinstance(filenames, list) is False:
                filenames = [filenames]
            # TODO: Rename it
            video_paths = [os.path.join(self.image_dir, name) for name in filenames]
            videos = []
            for video_path in video_paths:
                try:
                    video = load_video(video_path)
                except:  # TODO: Add error handling  # noqa
                    logger.warning(f"detected corrupted video: {video_path}")
                    return self[random.randint(0, len(self) - 1)]
                videos.extend([image[np.newaxis, :, :, :] for image in list(video)])
            if len(videos) == 0:
                logger.warning(f"detected empty video: {video_paths}")
                return self[random.randint(0, len(self) - 1)]
            images.extend(videos)
        # Text-only dataset
        else:
            has_image = False
            # NOTE: Without this dummy data, training with DeepSpeed is stuck if all
            # samples are from text-only datasets
            images.append(np.zeros(self.dummy_image_size, dtype=np.uint8))
        conversations = x["conversations"]
        for i in range(len(conversations)):
            if conversations[i]["value"] is None:
                logger.warning(f"detected empty conversations: {conversations}")
                return self[random.randint(0, len(self) - 1)]
        if conversations[0]["from"] != "human":
            logger.warning(
                "invalid conversations detected: "
                f"`from` value is invalid in conversations.\n"
                "=== warning report ===\n"
                f"annotation path: {annotation_path}\n"
                f"conversations: {conversations}"
            )
            return self[random.randint(0, len(self) - 1)]
        if has_image is True:
            # In stage 1, instructions are replaced with single image token
            if self.remove_instructions is True:
                for j in range(len(conversations)):
                    if conversations[j]["from"] == "human":
                        conversations[j]["value"] = self.image_token
            # Add image token to the first instruction if the number of images is
            # larger than the one of image tokens
            num_images = len(images)
            if self.max_num_images_per_sample > 0:
                if num_images > self.max_num_images_per_sample:
                    return self[random.randint(0, len(self) - 1)]
            num_image_tokens = self.get_num_image_tokens(conversations)
            if num_images > num_image_tokens:
                n = num_images - num_image_tokens
                prefix = self.image_token * n + "\n"
                for j in range(len(conversations)):
                    if conversations[j]["from"] == "human":
                        conversations[j]["value"] = prefix + conversations[j]["value"]
                        break
            if (
                self.check_images_and_conversations_format(images, conversations)
                is False
            ):
                logger.warning(
                    "mismatch detected: "
                    f"the number of {self.image_token} tokens is larger than the one of image!\n"  # noqa
                    "in this case, we don't know which image token to delete.\n"
                    "=== warning report ===\n"
                    f"annotation path: {annotation_path}\n"
                    f"num_images: {num_images}\n"
                    f"conversations: {conversations}"
                )
                return self[random.randint(0, len(self) - 1)]
        else:
            if (
                self.check_images_and_conversations_format(images, conversations, 1)
                is False
            ):
                logger.warning(
                    "mismatch detected: "
                    f"the number of {self.image_token} tokens is larger than the one of image!\n"  # noqa
                    "in this case, we don't know which image token to delete.\n"
                    "=== warning report ===\n"
                    f"annotation path: {annotation_path}\n"
                    f"num_images: {0}\n"
                    f"conversations: {conversations}"
                )
                return self[random.randint(0, len(self) - 1)]
        return {
            "images": images,
            "conversations": conversations,
        }
