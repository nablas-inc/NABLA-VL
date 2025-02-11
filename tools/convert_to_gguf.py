import glob
import os

import click


@click.command()
@click.option("--model-path")
def main(model_path):
    checkpoint_paths = []
    for path in glob.glob(os.path.join(model_path, "*")):
        if (
            path.endswith(".bin") is True
            and "pytorch" in path.split("/")[-1].split("\\")[-1]
        ):
            checkpoint_paths.append(path)
        elif (
            path.endswith(".safetensors")
            and "model" in path.split("/")[-1].split("\\")[-1]
        ):
            checkpoint_paths.append(path)
    print(checkpoint_paths)


if __name__ == "__main__":
    main()
