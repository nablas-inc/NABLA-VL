import json
from unittest import TestCase

import numpy as np
from PIL import Image

from nabla_vl.dataset import NablaVLDataset

ANNOTATION_PATH = "../examples/sample_data.json"
IMAGE_PATH = "../examples/birds.jpg"


class TestDataset(TestCase):
    def test_dataset(self) -> None:
        dataset = NablaVLDataset({ANNOTATION_PATH: 1.0}, "../examples")
        self.assertEqual(len(dataset), 3)
        image = Image.open(IMAGE_PATH)
        image = image.convert("RGB")
        image = np.array(image, dtype=np.uint8)
        conversations = json.load(open(ANNOTATION_PATH))[0]["conversations"]
        self.assertTrue((dataset[0]["images"][0] == image).all())
        self.assertEqual(dataset[0]["conversations"], conversations)
