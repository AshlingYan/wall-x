# Wall-X 真机部署指南

## 目录

1. [系统要求](#系统要求)
2. [环境配置](#环境配置)
3. [文件部署](#文件部署)
4. [配置文件](#配置文件)
5. [运行部署](#运行部署)
6. [故障排除](#故障排除)

---

## 系统要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA GPU (16GB VRAM) | NVIDIA H100/A100 (80GB) |
| CPU | 8核心 | 16核心+ |
| RAM | 32GB | 64GB+ |
| 存储 | 50GB 可用空间 | 200GB+ SSD |

### 软件要求

- **操作系统**: Linux (Ubuntu 20.04+ 推荐)
- **CUDA**: 12.1+
- **Python**: 3.10+
- **Git**: 用于下载代码

### 检查GPU

```bash
nvidia-smi
nvcc --version
```

---

## 环境配置

### 1. 创建 Conda 环境

```bash
# 创建环境
conda create -n wallx python=3.10 -y
conda activate wallx
```

### 2. 安装 PyTorch (CUDA 版本)

```bash
# CUDA 12.4 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 验证安装
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. 安装依赖包

```bash
# 从 requirements.txt 安装
pip install -r requirements.txt

# 或手动安装核心依赖
pip install transformers==4.49.0
pip install qwen-vl-utils==0.0.11
pip install Pillow opencv-python
pip install numpy safetensors diffusers
pip install PyYAML omegaconf accelerate
```

---

## 文件部署

### 目录结构

```
deployment_robot/
├── wall-x/                          # Wall-X 核心代码
│   ├── wall_x/
│   │   ├── model/                   # 模型定义
│   │   ├── data/                    # 数据处理
│   │   ├── serving/                 # 服务模块
│   │   └── infer/                   # 推理模块
│   ├── checkpoints/                 # 模型权重
│   │   └── stack_blocks_flow/35/    # 你的checkpoint
│   │       ├── config.yml           # 训练配置
│   │       ├── model.safetensors    # 模型权重
│   │       ├── normalizer_action.pth
│   │       ├── normalizer_propri.pth
│   │       └── processor_path/      # 如果需要
│   └── requirements.txt
│
└── control_your_robot/              # 机器人控制代码
    └── example/deploy/
        ├── piper_deploy_wallx.py    # 部署脚本
        ├── DEPLOYMENT_GUIDE.md      # 本文档
        └── requirements.txt
```

### 从源机器复制文件

```bash
# 在源机器上打包
cd /data/vla
tar czf wallx_deploy.tar.gz \
    wall-x/checkpoints/stack_blocks_flow/35 \
    wall-x/wall_x/ \
    control_your_robot/

# 在目标机器上解压
tar xzf wallx_deploy.tar.gz
```

### 使用 rsync 同步 (推荐)

```bash
# 在源机器运行
rsync -avz --progress \
    /data/vla/wall-x/ \
    user@target_robot:/path/to/deployment/wall-x/ \
    --exclude "*/.git" \
    --exclude "*/__pycache__" \
    --exclude "*/rollouts" \
    --exclude "*/workspace"

rsync -avz --progress \
    /data/vla/control_your_robot/ \
    user@target_robot:/path/to/deployment/control_your_robot/
```

---

## 配置文件

### 1. 检查 checkpoint 文件

确保以下文件存在：

```bash
CHECKPOINT_PATH="/path/to/wall-x/checkpoints/stack_blocks_flow/35"

ls -lh $CHECKPOINT_PATH
# 应该包含:
# - config.yml
# - model.safetensors (或多个 safetensors 文件)
# - normalizer_action.pth
# - normalizer_propri.pth
```

### 2. 检查 config.yml

```bash
cat $CHECKPOINT_PATH/config.yml | grep -A 5 "customized_robot_config"
```

确认以下配置正确：
```yaml
customized_robot_config:
  follow_left_ee_cartesian_pos: 3
  follow_left_ee_rotation: 3
  follow_left_gripper: 1
  follow_right_ee_cartesian_pos: 3
  follow_right_ee_rotation: 3
  follow_right_gripper: 1
```

### 3. 设置环境变量 (可选)

```bash
# 添加到 ~/.bashrc
export PYTHONPATH=/path/to/wall-x:/path/to/control_your_robot:$PYTHONPATH
source ~/.bashrc
```

---

## 运行部署

### 1. 测试环境

```bash
cd /path/to/control_your_robot/example/deploy

# 测试导入
python -c "
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.model.action_head import Normalizer
import torch
print('Import successful!')
"
```

### 2. Dry-run 测试

```bash
# 使用 --no-wait 跳过交互
PYTHONPATH=/path/to/wall-x:$PYTHONPATH python piper_deploy_wallx.py \
    --checkpoint /path/to/wall-x/checkpoints/stack_blocks_flow/35 \
    --task "stack the blocks" \
    --dry-run \
    --no-wait \
    --episode-num 1 \
    --max-step 10
```

预期输出：
```
==================================================
Wall-X Deployment for stack_blocks
==================================================
Model loaded successfully!
=== Episode 1/1 ===
Step: 0/10
Step: 10/10
Episode 1 finished. Steps: 10
```

### 3. 连接真实机器人

需要修改 `piper_deploy_wallx.py` 中的以下部分：

```python
# 1. 初始化机器人 (第340-343行)
if not args.dry_run:
    from your_robot_module import YourRobot
    robot = YourRobot(config="path/to/robot_config.yaml")
    robot.connect()

# 2. 重置机器人 (第352-354行)
if not args.dry_run:
    robot.reset()

# 3. 获取观测数据 (第377-381行)
else:
    data = robot.get_observation()
    img_arr, state = input_transform(data)

# 4. 发送动作 (第391-394行)
if not args.dry_run:
    move_data = output_transform(action)
    robot.move(move_data)
```

---

## 机器人接口适配

### 输入格式

机器人需要提供以下数据：

```python
# 图像数据 (3个相机)
img_arr = (
    cam_head_color,      # shape: (H, W, 3), dtype: uint8
    cam_right_wrist_color,  # shape: (H, W, 3), dtype: uint8
    cam_left_wrist_color,   # shape: (H, W, 3), dtype: uint8
)

# 状态数据 (14维双臂)
state = np.concatenate([
    left_arm_joint,      # 6维: 关节位置
    left_arm_gripper,    # 1维: 夹爪状态
    right_arm_joint,     # 6维: 关节位置
    right_arm_gripper,   # 1维: 夹爪状态
])  # shape: (14,)
```

### 输出格式

模型输出动作格式：

```python
# action: 14维向量
# [左臂6关节, 左臂夹爪, 右臂6关节, 右臂夹爪]

# 转换为机器人指令
move_data = {
    "arm": {
        "left_arm": {
            "joint": action[:6],      # 左臂关节位置
            "gripper": action[6]      # 左臂夹爪
        },
        "right_arm": {
            "joint": action[7:13],    # 右臂关节位置
            "gripper": action[13]     # 右臂夹爪
        }
    }
}
```

---

## 故障排除

### 1. CUDA 相关错误

**错误**: `CUDA out of memory`
```bash
# 解决方案: 减少batch size或使用更小的模型
# 或清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"
```

### 2. ImportError

**错误**: `ModuleNotFoundError: No module named 'qwen_vl_utils'`
```bash
# 确保安装了 qwen-vl-utils
pip install qwen-vl-utils==0.0.11

# 检查 PYTHONPATH
echo $PYTHONPATH
```

### 3. 模型加载错误

**错误**: `FileNotFoundError: checkpoint not found`
```bash
# 检查checkpoint路径
ls -lh /path/to/checkpoint/

# 确保使用绝对路径
--checkpoint /absolute/path/to/checkpoint
```

### 4. 图像处理错误

**错误**: `ValueError: Image features and image tokens do not match`
```bash
# 这通常是代码问题,确保使用最新版本的部署脚本
# 确保使用了 preprocesser_call 函数
```

---

## 性能优化

### 1. 减少推理延迟

```bash
# 使用 bfloat16 精度 (默认已启用)
# 模型加载时会自动转换为 bfloat16

# 增加控制频率 (默认 20Hz)
# 修改脚本中的: time.sleep(1 / 20)
```

### 2. 多GPU支持

```bash
# 指定使用的GPU
CUDA_VISIBLE_DEVICES=0 python piper_deploy_wallx.py ...
```

---

## 参考命令

### 完整的部署命令示例

```bash
# 1. 激活环境
conda activate wallx

# 2. 进入部署目录
cd /path/to/deployment/control_your_robot/example/deploy

# 3. Dry-run测试
PYTHONPATH=/path/to/wall-x:$PYTHONPATH python piper_deploy_wallx.py \
    --checkpoint /path/to/wall-x/checkpoints/stack_blocks_flow/35 \
    --task "stack the blocks" \
    --dry-run \
    --episode-num 3 \
    --max-step 100

# 4. 真机运行 (需要先配置机器人接口)
PYTHONPATH=/path/to/wall-x:$PYTHONPATH python piper_deploy_wallx.py \
    --checkpoint /path/to/wall-x/checkpoints/stack_blocks_flow/35 \
    --task "stack the blocks" \
    --episode-num 5 \
    --max-step 1000
```

---

## 联系支持

- **Wall-X GitHub**: [x-square-robot/wall-oss](https://github.com/x-square-robot/wall-oss)
- **训练脚本位置**: `/data/vla/wall-x/`
- **部署脚本位置**: `/data/vla/wall-x/control_your_robot/example/deploy/`

---

## 附录: 环境检查脚本

保存为 `check_env.sh`:

```bash
#!/bin/bash
echo "=== Wall-X Deployment Environment Check ==="
echo ""

echo "1. Python version:"
python --version

echo ""
echo "2. Conda environment:"
conda env list | grep "*"

echo ""
echo "3. PyTorch:"
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo ""
echo "4. GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "5. Key packages:"
python -c "
packages = ['transformers', 'qwen_vl_utils', 'PIL', 'cv2', 'numpy', 'safetensors']
for pkg in packages:
    try:
        mod = __import__(pkg.replace('-', '_'))
        version = getattr(mod, '__version__', 'installed')
        print(f'{pkg}: {version}')
    except ImportError:
        print(f'{pkg}: NOT INSTALLED')
"

echo ""
echo "6. Checkpoint files:"
CHECKPOINT_PATH="/path/to/wall-x/checkpoints/stack_blocks_flow/35"
if [ -d "$CHECKPOINT_PATH" ]; then
    ls -lh $CHECKPOINT_PATH
else
    echo "Checkpoint path not found: $CHECKPOINT_PATH"
fi
```

运行检查:
```bash
bash check_env.sh
```