#!/bin/bash
# 修复环境问题
echo "Current Python: $(which python)"
echo "Current PYTHONPATH: $PYTHONPATH"

# 强制使用 conda 环境
export PATH="/data/miniconda3/envs/wallx/bin:$PATH"

echo "Fixed Python: $(which python)"
echo "Testing torchdiffeq..."
python -c "import torchdiffeq; print('✓ torchdiffeq OK')"
