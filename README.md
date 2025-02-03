# NABLA-VL

## Installation

To start with the development, install dependencies with one of the following ways.

### Pip

```console
pip install .
```

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

## How to Run

#### Inference

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
        text="Describe this image",
        return_full_text=False,
    )
print(response)
```
