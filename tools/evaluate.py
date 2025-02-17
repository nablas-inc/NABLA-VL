import click

import numpy as np
import torch
from datasets import load_dataset
from deepspeed.utils import logger
from pandas import DataFrame
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from nabla_vl.benchmark import build_benchmark
from nabla_vl.inference import run_model
from nabla_vl.model import NablaVLForCausalLM
from nabla_vl.transforms import build_data_pipeline
from nabla_vl.constants import CHAT_TEMPLATE_WITHOUT_SYSTEM_MESSAGE
from nabla_vl.utils import set_random_seed


@click.command()
@click.option("--benchmark-name", default="MMMU", type=str)
@click.option("--model-name-or-path", default=MODEL_NAME, type=str)
@click.option("--split", type=str)
@click.option("--device", default=DEVICE, type=str)
def main(benchmark_name, model_name_or_path, split, device):
    set_random_seed(42)
    benchmark = build_benchmark(benchmark_name, model_name_or_path, split)
    score = benchmark.evaluate()
    logger.info(f"name: {benchmark_name}, score: {score:.4f}")


if __name__ == "__main__":
    main()