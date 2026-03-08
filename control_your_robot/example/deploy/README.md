# Wall-X 真机部署

本目录包含 Wall-X 模型在真实机器人上的部署脚本和配置文件。

## 快速开始

### 1. 环境准备

```bash
# 运行自动安装脚本
bash install.sh

# 或手动安装
pip install -r requirements.txt
```

### 2. 检查环境

```bash
# 运行环境检查脚本
bash check_env.sh
```

### 3. 测试部署

```bash
# Dry-run 测试（不连接真实机器人）
python piper_deploy_wallx.py \
    --checkpoint /path/to/checkpoint \
    --task "stack the blocks" \
    --dry-run \
    --no-wait
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `piper_deploy_wallx.py` | 主部署脚本（推荐使用） |
| `requirements.txt` | Python依赖列表 |
| `install.sh` | 自动安装脚本 |
| `check_env.sh` | 环境检查脚本 |
| `DEPLOYMENT_GUIDE.md` | 详细部署指南 |

## 系统要求

- **Python**: 3.10+
- **CUDA**: 12.1+
- **GPU**: NVIDIA GPU (16GB+ VRAM)
- **PyTorch**: 2.6.0

详细要求请查看 [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## 使用示例

```bash
# Dry-run 模式测试
python piper_deploy_wallx.py \
    --checkpoint /data/vla/wall-x/checkpoints/stack_blocks_flow/35 \
    --task "stack the blocks" \
    --dry-run

# 真机运行（需要先配置机器人接口）
python piper_deploy_wallx.py \
    --checkpoint /data/vla/wall-x/checkpoints/stack_blocks_flow/35 \
    --task "stack the blocks" \
    --episode-num 5 \
    --max-step 1000
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint` | 模型checkpoint路径 | 必需 |
| `--task` | 任务指令 | "stack the blocks" |
| `--dry-run` | 使用虚拟数据测试 | False |
| `--no-wait` | 跳过ENTER等待 | False |
| `--episode-num` | 运行轮数 | 5 |
| `--max-step` | 每轮最大步数 | 1000 |

## 机器人接口配置

要连接真实机器人，需要修改 `piper_deploy_wallx.py` 中的以下部分：

1. **初始化机器人** (第340行)
2. **获取观测数据** (第377行)
3. **发送动作指令** (第391行)

详细配置请参考 [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## 故障排除

### 常见问题

1. **CUDA out of memory**
   - 检查GPU内存: `nvidia-smi`
   - 减小batch size或使用更大的GPU

2. **ImportError**
   - 确保安装了所有依赖: `pip install -r requirements.txt`
   - 检查PYTHONPATH设置

3. **模型加载失败**
   - 确认checkpoint路径正确
   - 检查文件完整性: `ls -lh <checkpoint_path>`

## 目录结构

```
deployment/
├── piper_deploy_wallx.py       # 主部署脚本
├── requirements.txt            # 依赖列表
├── install.sh                  # 安装脚本
├── check_env.sh                # 环境检查
├── DEPLOYMENT_GUIDE.md         # 详细指南
└── README.md                   # 本文件
```

## 支持的机器人配置

当前脚本配置支持：

- **双臂机器人** (14维动作空间)
  - 左臂: 6个关节 + 1个夹爪
  - 右臂: 6个关节 + 1个夹爪

- **3个相机视角**
  - face_view (头部相机)
  - left_wrist_view (左手腕相机)
  - right_wrist_view (右手腕相机)

## 联系方式

- **Wall-X GitHub**: https://github.com/x-square-robot/wall-oss
- **问题反馈**: 提交GitHub Issue