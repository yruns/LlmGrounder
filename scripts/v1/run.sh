
ALBUMENTATIONS_DISABLE_VERSION_CHECK="1" torchrun --nproc_per_node=1 --master_port=10001 scripts/v1/main.py

TOKENIZERS_PARALLELISM="false"
OMP_NUM_THREADS="1"

export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

ALBUMENTATIONS_DISABLE_VERSION_CHECK="1" accelerate launch --num_processes=4 --main_process_port=10022 scripts/v1/main.py