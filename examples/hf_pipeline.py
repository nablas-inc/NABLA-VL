import torch
from transformers import pipeline

import nabla_vl
from nabla_vl.processor import NablaVLProcessor

TASK = "image-text-to-text"
MODEL = "nablasinc/NABLA-VL-15B"
DEVICE = "cuda"


def main():
    processor = NablaVLProcessor.from_pretrained(MODEL)
    pipe = pipeline(
        TASK,
        MODEL,
        # For architectures which flash attention doesn't support
        # model_kwargs={
        #     "enable_flash_attention_2": False,
        #     "attn_implementation": "eager",
        # },
        processor=processor,
        torch_dtype=torch.bfloat16,
    )
    with torch.autocast(DEVICE), torch.inference_mode():
        response = pipe(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
            text="Describe this image",
            return_full_text=False,
        )
    print(response)


if __name__ == "__main__":
    main()
