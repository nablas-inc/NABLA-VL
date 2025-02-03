import json
from typing import Any

import numpy as np
from decord import VideoReader, cpu
from PIL import Image


def load_json(path: str) -> Any:
    return json.load(open(path))


def load_image(path: str) -> Image.Image:
    image = Image.open(path)
    image = image.convert("RGB")
    return image


def load_video(path: str, *, max_num_frames: int = 32) -> np.ndarray:
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    total_num_frames = len(vr)
    frame_idxs = [i for i in range(0, total_num_frames, round(vr.get_avg_fps()))]
    if max_num_frames > 0:
        if len(frame_idxs) > max_num_frames:
            frame_idxs = np.linspace(
                0,
                total_num_frames - 1,
                max_num_frames,
                dtype=int,
            ).tolist()
    # Keep watching https://github.com/dmlc/decord/issues/177
    video = vr.get_batch(frame_idxs).asnumpy()
    return video
