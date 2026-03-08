#!/bin/bash
# Wall-X 部署环境检查脚本

echo "=== Wall-X Deployment Environment Check ==="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查函数
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo "1. Python 版本:"
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo "  Python: $PYTHON_VERSION"
    if python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
        check_pass "Python version >= 3.10"
    else
        check_fail "Python version < 3.10, please upgrade"
    fi
else
    check_fail "Python not found"
fi

echo ""
echo "2. Conda 环境:"
if command -v conda &> /dev/null; then
    CONDA_ENV=$(conda env list | grep "*" | awk '{print $1}')
    echo "  Current env: $CONDA_ENV"
    check_pass "Conda installed"
else
    check_warn "Conda not found (optional)"
fi

echo ""
echo "3. CUDA 和 GPU:"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU: $GPU_INFO"
    check_pass "NVIDIA GPU detected"

    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')
        echo "  CUDA: $CUDA_VERSION"
        check_pass "CUDA toolkit found"
    else
        check_warn "nvcc not found in PATH"
    fi
else
    check_fail "nvidia-smi not found - no GPU detected?"
fi

echo ""
echo "4. PyTorch:"
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "  PyTorch: $TORCH_VERSION"

    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
        echo "  CUDA available: Yes ($GPU_COUNT GPUs)"
        check_pass "PyTorch with CUDA support"
    else
        check_fail "PyTorch installed but CUDA not available"
    fi
else
    check_fail "PyTorch not installed"
fi

echo ""
echo "5. 关键依赖包:"
declare -a PACKAGES=(
    "transformers"
    "qwen_vl_utils"
    "PIL"
    "cv2"
    "numpy"
    "safetensors"
    "diffusers"
    "Pillow"
    "omegaconf"
)

for pkg in "${PACKAGES[@]}"; do
    IMPORT_NAME=$(echo $pkg | sed 's/-/_/g')
    if python -c "import $IMPORT_NAME" 2>/dev/null; then
        VERSION=$(python -c "import $IMPORT_NAME; print(getattr($IMPORT_NAME, '__version__', 'OK'))" 2>/dev/null)
        echo -e "  ${GREEN}✓${NC} $pkg: $VERSION"
    else
        echo -e "  ${RED}✗${NC} $pkg: NOT INSTALLED"
    fi
done

echo ""
echo "6. 磁盘空间:"
if command -v df &> /dev/null; then
    DISK_USAGE=$(df -h . | tail -1 | awk '{print $4 " available"}')
    echo "  Current directory: $DISK_USAGE"
    check_pass "Disk space check"
fi

echo ""
echo "7. PYTHONPATH:"
if [ -n "$PYTHONPATH" ]; then
    echo "  $PYTHONPATH"
    check_warn "PYTHONPATH is set"
else
    check_warn "PYTHONPATH not set (may need to set it)"
fi

echo ""
echo "8. 部署脚本:"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/piper_deploy_wallx.py" ]; then
    check_pass "piper_deploy_wallx.py found"
else
    check_fail "piper_deploy_wallx.py not found in $SCRIPT_DIR"
fi

echo ""
echo "=== Check Complete ==="
echo ""
echo "如果所有检查都通过, 可以运行部署脚本:"
echo "  python piper_deploy_wallx.py --checkpoint <path> --dry-run --no-wait"
echo ""
echo "如果有缺失的依赖, 请运行:"
echo "  pip install -r requirements.txt"