import click
from deepspeed.utils import logger

from nabla_vl.benchmark import build_benchmark
from nabla_vl.utils import set_random_seed

MODEL_NAME = "nablasinc/NABLA-VL"
DEVICE = "cuda"


@click.command()
@click.option("--benchmark-name", default="MMMU", type=str)
@click.option("--model-name-or-path", default=MODEL_NAME, type=str)
@click.option("--split", type=str)
@click.option("--device", default=DEVICE, type=str)
@click.option("--image-dir", type=str)
def main(benchmark_name, model_name_or_path, split, device, image_dir):
    set_random_seed(42)
    benchmark = build_benchmark(
        benchmark_name,
        model_name_or_path,
        split,
        device=device,
        image_dir=image_dir,
    )
    score = benchmark.evaluate()
    logger.info(f"name: {benchmark_name}, score: {score:.4f}")


if __name__ == "__main__":
    main()
