
ALBUMENTATIONS_DISABLE_VERSION_CHECK="1" torchrun --nproc_per_node=1 --master_port=10001 scripts/v1/main.py

TOKENIZERS_PARALLELISM="false"
OMP_NUM_THREADS="1"