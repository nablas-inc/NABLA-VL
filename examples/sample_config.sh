
export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=4

# FIXME:
# export NCCL_SOCKET_IFNAME=net1

# WIth INFO, so many message
export NCCL_DEBUG=WARN

# https://github.com/huggingface/transformers/issues/5486
export TOKENIZERS_PARALLELISM=false

NUM_GPUS=8
NNODES=1
DEEPSPEED_DIR="./examples"
CHECKPOINT_DIR="./checkpoints"
CONFIG_DIR="./examples"

# Single node training

ACCELERATE_CPU_AFFINITY=1 torchrun --standalone --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" \
    ./tools/train.py \
    --deepspeed "${DEEPSPEED_DIR}/zero3.json" \
    --output_dir "${CHECKPOINT_DIR}" \
    --json_path "${CONFIG_DIR}/sample_json.json"