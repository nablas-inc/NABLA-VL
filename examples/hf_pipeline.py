import nabla_vl
import torch
from nabla_vl.processor import NablaVLProcessor
from transformers import pipeline


TASK = "image-text-to-text"
MODEL = "nablasinc/NABLA-VL-15B"
DEVICE = "cuda"


def main():
    processor = NablaVLProcessor.from_pretrained(MODEL)
    pipe = pipeline(TASK, MODEL, processor=processor, torch_dtype=torch.bfloat16)
    with torch.autocast(DEVICE), torch.inference_mode():
        response = pipe(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
            text="Describe this image",
            return_full_text=False,
        )
    print(response)


if __name__ == "__main__":
    main()
