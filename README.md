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

```json
[
    ...
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
        ...
    },
    {
        "video": "video.mp4",  // Use video key if input is video.
        ...
    },
    ...
]
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


processor = NablaVLProcessor.from_pretrained(MODEL)
pipe = pipeline(TASK, MODEL, processor=processor, torch_dtype=torch.bfloat16)
with torch.autocast(DEVICE), torch.inference_mode():
    response = pipe(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
        text="この画像について教えてください！",
        return_full_text=False,
    )
print(response)
```

</details>
