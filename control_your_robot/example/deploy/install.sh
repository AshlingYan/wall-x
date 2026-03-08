#!/bin/bash
# Wall-X 部署环境快速安装脚本

set -e  # 遇到错误立即退出

echo "=== Wall-X Deployment Environment Setup ==="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查是否在conda环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}Warning: Not in a conda environment${NC}"
    echo "建议先创建并激活conda环境:"
    echo "  conda create -n wallx python=3.10 -y"
    echo "  conda activate wallx"
    echo ""
    read -p "继续安装? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "1. 安装 PyTorch (CUDA 12.4)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "2. 安装核心依赖..."
pip install transformers==4.49.0
pip install qwen-vl-utils==0.0.11
pip install Pillow opencv-python

echo ""
echo "3. 安装数值计算库..."
pip install numpy>=2.0.0
pip install safetensors==0.7.0
pip install diffusers==0.36.0
pip install torchdiffeq==0.2.5

echo ""
echo "4. 安装配置和工具库..."
pip install PyYAML
pip install omegaconf>=2.3.0
pip install accelerate>=1.10.0
pip install peft>=0.17.1

echo ""
echo "5. 验证安装..."
python -c "
import torch
import transformers
import qwen_vl_utils
import PIL
import cv2
import numpy as np
import safetensors
import diffusers

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Transformers: {transformers.__version__}')
print('All packages installed successfully!')
"

echo ""
echo -e "${GREEN}=== 安装完成! ===${NC}"
echo ""
echo "运行环境检查:"
echo "  bash check_env.sh"
echo ""
echo "测试部署脚本:"
echo "  python piper_deploy_wallx.py --checkpoint <path> --dry-run --no-wait"