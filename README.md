# NABLA-VL

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/nablasinc/NABLA-VL-15B)
[![arXiv](https://img.shields.io/badge/arXiv-2301.12345-B31B1B.svg)](https://arxiv.org/abs/2301.12345)

> The technical report will be available soon!

**NABLA-VL** is a Japanese Vision-Language Model (VLM) developed by NABLAS. It supports inputs of images, multiple images, and videos, enabling comprehensive multimodal understanding and generation.

---

## ✅ To-Do List

- [x] Publish the NABLA-VL model on Hugging Face
- [ ] Publish the preprint on arXiv

---

## 🚀 Installation

### Using Rye

```bash
rye sync
# Activate the virtual environment
source .venv/bin/activate
```

### Installing FlashAttention

To enable `--attn_implementation flash_attention_2`, install `flash-attn`:

```bash
rye add uv
uv pip install torch
CXX=g++ uv pip install flash-attn --no-build-isolation
```

---

## 📚 Usage

### Training Data Format

​If you would like to see a concrete example, please refer to `examples/sample_data.json`.

```jsonc
[
  {
    "image": "birds.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nHow many birds are there?"
      },
      {
        "from": "gpt",
        "value": "9"
      }
    ]
  },
  {
    "image": ["image_0.jpg", "image_1.jpg"]
  },
  {
    "video": "video.mp4"
  }
]
```

### Evaluation Examples

```bash
python tools/evaluate.py --benchmark-name MMMU --split validation --device cuda
python tools/evaluate.py --benchmark-name JMMMU --split test --device cuda
python tools/evaluate.py --benchmark-name MMStar --split val --device cuda
python tools/evaluate.py --benchmark-name BLINK --split val --device cuda
```

### Inference Examples

#### Using 🤗 Transformers Pipeline

```python
import nabla_vl
import torch
from nabla_vl.processor import NablaVLProcessor
from transformers import pipeline

TASK = "image-text-to-text"
MODEL = "nablasinc/NABLA-VL"
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

#### Using Custom Inference Script

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

MODEL = "nablasinc/NABLA-VL"
DEVICE = "cuda"

model = NablaVLForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, resume_download=True)
model.to(DEVICE)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
data_pipeline = build_data_pipeline(model.config, tokenizer)
instruction = "この画像について教えてください！"
images = []
urls = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
    # Add items here to input multiple images
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
    device=DEVICE,
)
```

---

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements and fixes.

---
