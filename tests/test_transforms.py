from unittest import TestCase

import numpy as np
import torch
from PIL import Image
from torch import FloatTensor
from torchvision.transforms import InterpolationMode

from nabla_vl.transforms import Factorize

ANNOTATION_PATH = "../examples/sample_data.json"
IMAGE_PATH = "../examples/birds.jpg"


def to_tensor(image: np.ndarray) -> FloatTensor:
    image = torch.tensor(image, dtype=torch.float32)
    image = image.permute(0, 3, 1, 2)
    return image


class TestTransform(TestCase):
    def test_factorize(self) -> None:
        image = Image.open(IMAGE_PATH)
        image = image.convert("RGB")
        image = np.array(image, dtype=np.uint8)[np.newaxis, :, :, :]
        image = to_tensor(image)
        h, w = image.size()[-2:]
        self.assertEqual(h, 240)
        self.assertEqual(w, 240)
        transform = Factorize(
            factor=14,
            mode=InterpolationMode.BILINEAR,
            antialias=True,
            p=1.0,
        )
        converted_image = transform(({"images": [image]},))[0]["images"][0]
        h, w = converted_image.size()[-2:]
        self.assertEqual(h % 14, 0)
        self.assertEqual(w % 14, 0)
