from typing import Any, Optional

from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LengthGroupedSampler


# https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py#L622
# This is almost the same sampler to ↑, but it can get length without loading image
class VisionCompatibleLengthGroupedSampler(LengthGroupedSampler):
    def __init__(
        self,
        batch_size: int,
        dataset: Dataset,
        *,
        generator: Optional[Any] = None,
    ):
        if dataset is None:
            raise ValueError
        # NOTE: Customize here to extend it for your use case
        # --- My best sampler ---
        # NOTE: Customize here to extend it for your use case
        self.batch_size = batch_size
        self.lengths = dataset.get_lengths()
        self.generator = generator
