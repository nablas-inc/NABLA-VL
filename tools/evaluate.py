import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nabla_vl.inference import run_model
from nabla_vl.model import NablaVLForCausalLM
from nabla_vl.transforms import build_data_pipeline

MODEL_NAME = "nablasinc/NABLA-VL-15B"
DEVICE = "cuda"
INSTRUCTION = {
    "ja": "Optionsの中からアルファベットで解答してください。<image n>はn枚目の画像の文中での位置を表します。",
    "en": "Answer with the option's letter from the given choices directly.",
}


def get_model(model_name, device):
    model = NablaVLForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_pipeline = build_data_pipeline(model.config, tokenizer)
    return model, tokenizer, data_pipeline


def preprocess_mmmu(instruction, max_num_images):
    for i in range(1, max_num_images + 1):
        instruction = instruction.replace(f"<image {i}>", "")
    return instruction


def evaluate_mmmu(
    model,
    tokenizer,
    data_pipeline,
    dataset_name,
    subjects,
    split,
    max_num_images,
    option_key,
    language,
    preprocessor,
    fomatter,
):
    score = 0
    n = 0
    for subject in subjects:
        dataset = load_dataset(dataset_name, name=subject, split=split)
        for item in tqdm(dataset, total=len(dataset), desc=subject):
            # Preprocess images
            images = []
            for i in range(1, max_num_images + 1):
                if item[f"image_{i}"] is None:
                    break
                image = np.array(item[f"image_{i}"].convert("RGB"))[np.newaxis, :, :, :]
                images.append(image)
            # Preprocess text
            choices = ""
            try:
                item[option_key] = preprocessor(item[option_key])
            except TypeError:
                print(item[option_key])
                continue
            for i in range(len(item[option_key])):
                assert 65 + i < 91
                choices += chr(65 + i) + ". " + item[option_key][i] + "\n"
            instruction = preprocess_mmmu(
                item["question"] + "\nOptions:\n" + choices + INSTRUCTION[language],
                max_num_images,
            )
            response = (
                run_model(
                    model,
                    tokenizer,
                    data_pipeline,
                    instruction,
                    images=images,
                )[0]
                .strip()
                .upper()
            )
            score += 1 * (response == fomatter(item["answer"]))
            n += 1
    print(f"score: {score / n * 100.0:.2f}")


def main():
    model, tokenizer, data_pipeline = get_model(MODEL_NAME, DEVICE)
    evaluate_mmmu(
        model,
        tokenizer,
        data_pipeline,
        "JMMMU/JMMMU",
        [
            "Accounting",
            "Agriculture",
            "Architecture_and_Engineering",
            "Basic_Medical_Science",
            "Biology",
            "Chemistry",
            "Clinical_Medicine",
            "Computer_Science",
            "Design",
            "Diagnostics_and_Laboratory_Medicine",
            "Economics",
            "Electronics",
            "Energy_and_Power",
            "Finance",
            "Japanese_Art",
            "Japanese_Heritage",
            "Japanese_History",
            "Manage",
            "Marketing",
            "Materials",
            "Math",
            "Mechanical_Engineering",
            "Music",
            "Pharmacy",
            "Physics",
            "Psychology",
            "Public_Health",
            "World_History",
        ],
        "test",
        7,
        "options",
        "ja",
        lambda x: eval(x),
        lambda x: x,
    )
    """
    evaluate_mmmu(
        model,
        tokenizer,
        data_pipeline,
        "BLINK-Benchmark/BLINK",
        [
            "Art_Style",
            "Counting",
            "Forensic_Detection",
            "Functional_Correspondence",
            "IQ_Test",
            "Jigsaw",
            "Multi-view_Reasoning",
            "Object_Localization",
            "Relative_Depth",
            "Relative_Reflectance",
            "Semantic_Correspondence",
            "Spatial_Relation",
            "Visual_Correspondence",
            "Visual_Similarity",
        ],
        "val",
        4,
        "choices",
        "en",
        lambda x: x,
        lambda x: x[1],
    )
    """


if __name__ == "__main__":
    main()
