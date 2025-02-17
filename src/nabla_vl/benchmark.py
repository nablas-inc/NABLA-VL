import abc
from abc import ABC
from collections import OrderedDict
from typing import Any, Dict, Optional, List

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from deepspeed.utils import logger
from transformers import AutoModelForImageTextToText, AutoTokenizer
from tqdm.auto import tqdm

from .constants import CHAT_TEMPLATE_WITHOUT_SYSTEM_MESSAGE
from .inference import run_model
from .registry import BENCHMARKS
from .transforms import build_data_pipeline


def build_benchmark(
    name: str,
    model_name_or_path: str,
    split: str,
    *,
    device: Any = "cuda",
    dtype: Any = torch.bfloat16,
) -> "Benchmark":
    logger.info(f"Initialize {name}")
    benchmark = BENCHMARKS[name](
        model_name_or_path,
        split,
        device=device,
        dtype=dtype,
    )
    return benchmark


class Benchmark(ABC):
    def __init__(
        self,
        model_name_or_path: str,
        split: str,
        *,
        device: Any = "cuda",
        dtype: Any = torch.bfloat16,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.split = split
        self.device = device
        self.dtype = dtype
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
        )
        self.tokenizer.chat_template = CHAT_TEMPLATE_WITHOUT_SYSTEM_MESSAGE
        self.data_pipeline = build_data_pipeline(self.model.config, self.tokenizer)
        self.load_items()

    @abc.abstractmethod
    def load_items(self) -> None:
        return None

    @abc.abstractmethod
    def get_instruction(self, item: Dict[str, Any]) -> str:
        return ""

    @abc.abstractmethod
    def get_answer(self, item: Dict[str, Any]) -> str:
        return ""

    @abc.abstractmethod
    def get_images(self, item: Dict[str, Any]) -> Optional[List[np.ndarray]]:
        return None

    @abc.abstractmethod
    def compare(self, response: str, answer: str) -> bool:
        return True

    def get_response(self, instruction: str, *, images: Optional[np.ndarray] = None) -> str:
        response = run_model(
            self.model,
            self.tokenizer,
            self.data_pipeline,
            instruction,
            images=images,
            max_new_tokens=12,
        )[0].strip()
        return response

    def evaluate(self) -> float:
        score = 0.0
        n = 0
        for item in tqdm(self.items):
            instruction = self.get_instruction(item)
            images = self.get_images(item)
            answer = self.get_answer(item)
            response = self.get_response(instruction, images=images)
            is_correct = self.compare(response, answer)
            score += float(is_correct)
            n += 1
        score /= n
        return score


@BENCHMARKS.register_cls()
class MMMU(Benchmark):
    names: List[str] = [
        "Accounting",
        "Agriculture",
        "Architecture_and_Engineering",
        "Art",
        "Art_Theory",
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
        "Geography",
        "History",
        "Literature",
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
        "Sociology",
    ]
    prefix: Dict[str, str] = {
        "multiple-choice": "Answer with the option's letter from the given choices directly. Don't include prefix like 'The answer is'",
        "open": "Answer the question using a single word or phrase.",
    }
    max_num_images = 7

    def load_items(self) -> None:
        dsets = []
        progress_bar = tqdm(self.names)
        for name in progress_bar:
            progress_bar.set_postfix(OrderedDict(name=name))
            dset = load_dataset("MMMU/MMMU", name, split=self.split)
            dsets.append(dset)
        self.items = concatenate_datasets(dsets)

    def get_instruction(self, item: Dict[str, Any]) -> str:
        if item["question_type"] == "multiple-choice":
            choices = ""
            if isinstance(item["options"], str) is True:
                item["options"] = eval(item["options"])
            for i in range(len(item["options"])):
                assert 65 + i < 91
                choices += chr(65 + i) + ". " + item["options"][i] + "\n"
            instruction = item["question"]
            instruction += "\nOptions:\n"
            instruction += choices
            instruction += self.prefix["multiple-choice"]
        elif item["question_type"] == "open":
            instruction = item["question"]
            instruction += " "
            instruction += self.prefix["open"]
        else:
            raise ValueError
        return instruction

    def get_answer(self, item: Dict[str, Any]) -> str:
        answer = item["answer"]
        return answer

    def get_images(self, item: Dict[str, Any]) -> Optional[List[np.ndarray]]:
        images = []
        for i in range(1, self.max_num_images + 1):
            if item[f"image_{i}"] is None:
                break
            images.append(np.array(item[f"image_{i}"].convert("RGB"))[np.newaxis, :, :, :])
        return images

    def compare(self, response: str, answer: str) -> bool:
        is_correct = response == answer
        return is_correct


@BENCHMARKS.register_cls()
class JMMMU(MMMU):
    names: List[str] = [
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
    ]
    prefix: Dict[str, str] = {
        "multiple-choice": "選択肢の中から大文字アルファベットのみで解答してください。",
        "open": "単文または１単語で解答してください。",
    }
    max_num_images = 7

    def load_items(self) -> None:
        dsets = []
        progress_bar = tqdm(self.names)
        for name in progress_bar:
            progress_bar.set_postfix(OrderedDict(name=name))
            dset = load_dataset("JMMMU/JMMMU", name, split=self.split)
            dsets.append(dset)
        self.items = concatenate_datasets(dsets)

    def get_instruction(self, item: Dict[str, Any]) -> str:
        if item["question_type"] == "multiple-choice":
            choices = ""
            if isinstance(item["options"], str) is True:
                item["options"] = eval(item["options"])
            for i in range(len(item["options"])):
                assert 65 + i < 91
                choices += chr(65 + i) + ". " + item["options"][i] + "\n"
            instruction = item["question"]
            instruction += "\選択肢:\n"
            instruction += choices
            instruction += self.prefix["multiple-choice"]
        elif item["question_type"] == "open":
            instruction = item["question"]
            instruction += " "
            instruction += self.prefix["open"]
        else:
            raise ValueError
        return instruction
        


@BENCHMARKS.register_cls()
class MMStar(Benchmark):
    prefix: Dict[str, str] = {
        "multiple-choice": "Answer with the option's letter from the given choices directly. Don't include prefix like 'The answer is'",
    }

    def load_items(self) -> None:
        self.items = load_dataset("Lin-Chen/MMStar", split=self.split)

    def get_instruction(self, item: Dict[str, Any]) -> str:
        instruction = item["question"]
        instruction += " "
        instruction += self.prefix["multiple-choice"]
        return instruction

    def get_answer(self, item: Dict[str, Any]) -> str:
        answer = item["answer"]
        return answer

    def get_images(self, item: Dict[str, Any]) -> Optional[List[np.ndarray]]:
        images = [np.array(item[f"image"].convert("RGB"))[np.newaxis, :, :, :]]
        return images

    def compare(self, response: str, answer: str) -> bool:
        is_correct = response == answer
        return is_correct


@BENCHMARKS.register_cls()
class BLINK(MMMU):
    names: List[str] = [
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
    ]
    prefix: Dict[str, str] = {
        "multiple-choice": "Answer with the option's letter from the given choices directly. Don't include prefix like 'The answer is'",
    }
    max_num_images = 4

    def load_items(self) -> None:
        dsets = []
        progress_bar = tqdm(self.names)
        for name in progress_bar:
            progress_bar.set_postfix(OrderedDict(name=name))
            dset = load_dataset("BLINK-Benchmark/BLINK", name, split=self.split)
            dsets.append(dset)
        self.items = concatenate_datasets(dsets)

    def get_instruction(self, item: Dict[str, Any]) -> str:
        choices = ""
        for i in range(len(item["choices"])):
            assert 65 + i < 91
            choices += chr(65 + i) + ". " + item["choices"][i] + "\n"
        instruction = item["question"]
        instruction += "\nOptions:\n"
        instruction += choices
        instruction += self.prefix["multiple-choice"]
        return instruction

    def get_answer(self, item: Dict[str, Any]) -> str:
        answer = item["answer"]
        return answer

    def get_images(self, item: Dict[str, Any]) -> Optional[List[np.ndarray]]:
        images = []
        for i in range(1, self.max_num_images + 1):
            if item[f"image_{i}"] is None:
                break
            images.append(np.array(item[f"image_{i}"].convert("RGB"))[np.newaxis, :, :, :])
        return images

    def compare(self, response: str, answer: str) -> bool:
        is_correct = response == answer[1]
        return is_correct
    