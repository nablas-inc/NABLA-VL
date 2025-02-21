# NABLA-VL

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/nablasinc/NABLA-VL-15B) [![arXiv](https://img.shields.io/badge/arXiv-2301.12345-B31B1B.svg)](https://arxiv.org/abs/2301.12345)

## Installation

### Rye

```console
rye sync
# Activate virtual environment
source .venv/bin/activate
```

### FlashAttention

To enable `--attn_implementation flash_attention_2`, you need to install `flash-attn`.

```console
rye add uv
uv pip install torch
CXX=g++ uv pip install flash-attn --no-build-isolation
```

## Getting Started

### Training

<details>
<summary>Data Format</summary>

```jsonc
[
    // ...
    {
        "image": "birds.jpg",  // {image_dir}/birds.jpg will be loaded during training (image_dir comes from config).
        "conversations": [  // conversations is pair(s) of input(s) and output(s).
            {
                "from": "human",
                "value": "<image>\nHow many birds are there?"  // This is prompt. You can drop <image>\n. In that case, it automatically prepends <image>\n.
            },
            {
                "from": "gpt",
                "value": "9"  // This is label.
            }
        ]
    },
    {
        "image": ["image_0.jpg", "image_1.jpg"],  // Use list for multi-image input.
        // ...
    },
    {
        "video": "video.mp4",  // Use video key if input is video.
        // ...
    },
    // ...
]
```

</details>

### Evaluation

<details>
<summary>Example</summary>

```console
python tools/evaluate.py --benchmark-name MMMU --split validation
python tools/evaluate.py --benchmark-name JMMMU --split test
python tools/evaluate.py --benchmark-name MMStar --split val
python tools/evaluate.py --benchmark-name BLINK --split val
```

</details>

### Inference

<details>
<summary>🤗 Pipelines</summary>

```python
import nabla_vl
import torch
from nabla_vl.processor import NablaVLProcessor
from transformers import pipeline


TASK = "image-text-to-text"
MODEL = "nablasinc/NABLA-VL-15B"
DEVICE = "cuda"


processor = NablaVLProcessor.from_pretrained(MODEL, use_fast=False)
pipe = pipeline(TASK, MODEL, processor=processor, torch_dtype=torch.bfloat16)
with torch.autocast(DEVICE), torch.inference_mode():
    response = pipe(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
        text="この画像について教えてください！",
        return_full_text=False,
    )
print(response)
```

```python
import requests
from PIL import Image

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from nabla_vl.constants import CHAT_TEMPLATE_WITHOUT_SYSTEM_MESSAGE
from nabla_vl.inference import run_model_with_stream
from nabla_vl.io import load_image
from nabla_vl.model import NablaVLForCausalLM
from nabla_vl.transforms import build_data_pipeline

MODEL = "nablasinc/NABLA-VL-15B"


model = NablaVLForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
model.to("cuda")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
data_pipeline = build_data_pipeline(model.config, tokenizer)
instruction = "この画像について教えてください！"
images = []
urls = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
]
for url in urls:
    images.append(
        np.array(
            Image.open(
                requests.get(
                    url,
                    stream=True,
                ).raw,
            ).convert("RGB"),
        )[np.newaxis, :, :, :],
    )
run_model_with_stream(
    model,
    tokenizer,
    data_pipeline,
    instruction,
    images=images,
)
```

</details>
