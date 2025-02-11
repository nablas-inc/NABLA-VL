from unittest import TestCase

import numpy as np
import torch
from PIL import Image
from torch import FloatTensor
from torchvision.transforms import InterpolationMode

from nabla_vl.constants import IMAGE_TOKEN_ID
from nabla_vl.transforms import Factorize, AggregateImages

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
