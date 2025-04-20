from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch.distributed as dist
import wandb
from deepspeed.utils import logger
from torch.utils.data import RandomSampler, Sampler
from transformers import HfArgumentParser, Trainer, TrainingArguments

import nabla_vl
import nabla_vl.transforms
from nabla_vl.sampler import VisionCompatibleLengthGroupedSampler


class TrainerWorkaround(Trainer):
    def create_optimizer(self):
        self.optimizer = super().create_optimizer()
        # Scheduler-free optimizers need to call .train() or .eval() before first .step()  # NOQA
        if hasattr(self.optimizer, "train") is True:
            self.optimizer.train()
        return self.optimizer

    def _get_train_sampler(self) -> Sampler:
        if self.args.group_by_length is True:
            return VisionCompatibleLengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                self.train_dataset,
            )
        else:
            return RandomSampler(self.train_dataset)


@dataclass
class NablaTrainingArguments(TrainingArguments):
    # General
    json_path: Optional[str] = field(default=None)
    model_name_or_path: str = field(default="microsoft/phi-4")
    enable_lazy_init: bool = field(default=False)
    # Tokenizer
    max_length: int = field(default=4096)
    # Dataset
    dataset_name: str = field(default="NablaVLDataset")
    annotation_paths: List[str] = field(default_factory=lambda: {})
    image_dir: str = field(default="")
    remove_instructions: bool = field(default=False)
    max_num_samples: int = field(default=-1)
    max_num_images_per_sample: int = field(default=-1)
    max_num_characters: int = field(default=-1)
    max_num_frames: int = field(default=32)
    # Data pipeline
    apply_chat_template: bool = field(default=False)
    normalize_type: str = field(default="inception")
    use_anyres: bool = field(default=False)
    eos_token: str = field(
        default="\n",
        metadata={
            "help": (
                "Token which is appended to sequence. "
                "This is used only when Resize transform is used."
            )
        },
    )
    factor: int = field(
        default=14,
        metadata={"help": "Image size is rounded by this value."},
    )
    patch_size: int = field(default=14)
    min_shorter_size: int = field(default=70)
    max_longer_size: int = field(default=980)
    thumbnail_size: int = field(default=448)
    min_pixels: int = field(default=196)
    max_pixels: int = field(default=147456)
    max_total_size: int = field(default=-1)
    add_thumbnail: bool = field(default=False)
    add_marks: bool = field(default=False)
    wrap_images: bool = field(default=True)
    max_num_tiles: int = field(default=4)
    # Model
    vision_tower_name: str = field(
        default="HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit"
    )
    enable_flash_attention_2: bool = field(default=True)
    freeze_vision_tower: bool = field(default=False)
    freeze_projector: bool = field(default=False)
    freeze_embedding: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_lm_head: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2")
    gradient_checkpointing: bool = field(default=True)
    max_vision_tokens: int = field(default=4900)
    use_new_line_token: bool = field(default=False)
    use_new_column_token: bool = field(default=False)
    neftune_alpha: Optional[float] = field(default=False)
    num_registers: int = field(default=-1)
    # Accelerate
    fp8: bool = field(default=False)
    # Others
    num_gpus_per_node: int = field(default=8)
    num_nodes: int = field(default=1)


def main() -> None:
    parser = HfArgumentParser(NablaTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    if args.json_path is not None:
        logger.info(f"Loading {args.json_path}")
        args = parser.parse_json_file(args.json_path)[0]
    nabla_vl.utils.set_random_seed(args.seed)
    if args.local_rank == 0:
        wandb.init(entity=args.entity, project=args.project)
    tokenizer = nabla_vl.tokenizer.build_tokenizer(args)
    dataset = nabla_vl.dataset.build_dataset(args)
    data_pipeline = nabla_vl.transforms.build_data_pipeline(args, tokenizer)
    model = nabla_vl.model.build_model(args, tokenizer)
    trainer = TrainerWorkaround(
        model=model,
        args=args,
        data_collator=data_pipeline,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    # Resume training
    if len(list(Path(args.output_dir).glob("checkpoint-*"))) > 0:
        logger.info("checkpoint detected! resuming training process.")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    trainer.save_model(args.output_dir)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
