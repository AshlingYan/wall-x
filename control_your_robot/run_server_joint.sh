#!/bin/bash
# Wall-X 推理服务器启动脚本 - 关节角模型
# 在 ygx03 上运行

# 激活环境
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wallx

# 设置路径
cd /data/vla/wall-x/control_your_robot
export PYTHONPATH=/data/vla/wall-x:$PYTHONPATH

# 配置
CHECKPOINT_PATH="/data/vla/wall-x/checkpoints/stack_blocks_joint/4"
SERVER_IP="0.0.0.0"
SERVER_PORT=10000

echo "=========================================="
echo "启动 Wall-X 推理服务器"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "监听地址: $SERVER_IP:$SERVER_PORT"
echo "=========================================="

# 启动服务器
python scripts/server_wallx_joint.py
