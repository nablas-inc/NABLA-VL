from unittest import TestCase

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import FloatTensor
from torchvision.transforms import InterpolationMode

from nabla_vl.constants import IMAGE_TOKEN, IMAGE_TOKEN_ID
from nabla_vl.transforms import AddThumbnail, AggregateImages, Factorize

ANNOTATION_PATH = "../examples/sample_data.json"
IMAGE_PATH = "../examples/birds.jpg"
# Original image size of birds.jpg
H = W = 240


def to_tensor(image: np.ndarray) -> FloatTensor:
    image = torch.tensor(image, dtype=torch.float32)
    image = image.permute(0, 3, 1, 2)
    return image


def load_sample_image() -> FloatTensor:
    image = Image.open(IMAGE_PATH)
    image = image.convert("RGB")
    image = np.array(image, dtype=np.uint8)[np.newaxis, :, :, :]
    image = to_tensor(image)
    return image


class TestTransform(TestCase):
    def test_aggregate_images(self) -> None:
        image = load_sample_image()
        transform = AggregateImages(
            factor=14,
            patch_size=14,
            image_token_id=IMAGE_TOKEN_ID,
            wrap_images=True,
            p=1.0,
        )
        batch = ({"images": [image, image, image, image]},)
        converted_image = transform(batch)[0]["images"][0]
        h, w = converted_image.size()[-2:]
        self.assertEqual(h, 2 * H)
        self.assertEqual(w, 2 * W)

    def test_add_thumbnail(self) -> None:
        image = load_sample_image()
        image = F.interpolate(
            image,
            size=(2048, 2048),
            mode="bilinear",
            align_corners=False,
        )
        image = image.permute(0, 2, 3, 1)
        image = image.numpy()
        transform = AddThumbnail(
            image_token=IMAGE_TOKEN,
            factor=14,
            min_pixels=14 * 14,
            max_pixels=384 * 384,
            patch_size=14,
            dtype=torch.float32,
            scaler=1.0 / 255.0,
            min_shorter_size=70,
            max_longer_size=980,
            m=1.5,
            thumbnail_size=448,
            mode=InterpolationMode.BILINEAR,
            antialias=True,
            add_marks=False,
            p=1.0,
        )
        batch = ({"images": [image]},)
        converted_images = transform(batch)[0]["images"]
        self.assertEqual(len(converted_images), 2)
        h_thumbnail, w_thumbnail = converted_images[0].shape[1:3]
        self.assertEqual(h_thumbnail, 448)
        self.assertEqual(w_thumbnail, 448)
        h_original, w_original = converted_images[1].shape[1:3]
        self.assertEqual(h_original, 2048)
        self.assertEqual(w_original, 2048)

    def test_factorize(self) -> None:
        image = load_sample_image()
        h, w = image.size()[-2:]
        self.assertEqual(h, H)
        self.assertEqual(w, W)
        transform = Factorize(
            factor=14,
            mode=InterpolationMode.BILINEAR,
            antialias=True,
            p=1.0,
        )
        batch = ({"images": [image]},)
        converted_image = transform(batch)[0]["images"][0]
        h, w = converted_image.size()[-2:]
        self.assertEqual(h % 14, 0)
        self.assertEqual(w % 14, 0)

    def test_add_patch_attention_mask(self) -> None:
        pass
