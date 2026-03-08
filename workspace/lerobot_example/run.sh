#!/bin/bash
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wallx

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Fix multi-process stability: avoid thread contention in pyav/ffmpeg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# NCCL settings: increase timeout for slow data loading, enable debug
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN

# Avoid fork-safety issues with pyav video decoding
export PYTHONFAULTHANDLER=1

# print current time
echo "[current time: $(date +'%Y-%m-%d %H:%M:%S')]"

code_dir="/data/vla/wall-x"
config_path="/data/vla/wall-x/workspace/lerobot_example"

# Use a fixed port instead of a random one
export PORT=$((21000 + $RANDOM % 30000))

MASTER_PORT=10239 # use 5 digits ports

export LAUNCHER="accelerate launch --num_processes=$NUM_GPUS --main_process_port=$PORT"

export SCRIPT="${code_dir}/train_qact.py"
export SCRIPT_ARGS="--config ${config_path}/config_qact_from_vlm.yml --seed $MASTER_PORT"

echo "Running command: $LAUNCHER $SCRIPT $SCRIPT_ARGS"

$LAUNCHER $SCRIPT $SCRIPT_ARGS
