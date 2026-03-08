import os
import yaml
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.data.load_lerobot_dataset import load_test_dataset, get_data_configs
from wall_x.model.model_utils import register_normalizers
import copy


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["data"]["model_type"] = config.get("model_type")

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_horizon", type=int, default=10, help="Prediction horizon")
    parser.add_argument("--origin_action_dim", type=int, default=14, help="Original action dimension")
    parser.add_argument("--checkpoint_path", type=str,
                        default="/data/vla/wall-x/checkpoints/stack_blocks_joint/5",
                        help="Path to checkpoint directory")
    parser.add_argument("--save_dir", type=str,
                        default="/data/vla/wall-x/workspace/stack_blocks/openloop_plots",
                        help="Directory to save plots")
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="Number of episodes to visualize")
    args = parser.parse_args()

    origin_action_dim = args.origin_action_dim
    pred_horizon = args.pred_horizon

    # 检查checkpoint是否存在
    model_path = args.checkpoint_path
    if not os.path.exists(model_path):
        print(f"❌ Checkpoint not found at {model_path}")
        print(f"请等待训练完成 epoch 5 后再运行此脚本")
        print(f"或者手动指定已存在的checkpoint路径")
        print(f"\n可用的checkpoint路径示例：")
        print(f"  --checkpoint_path /data/vla/wall-x/checkpoints/stack_blocks_joint/5")
        print(f"  --checkpoint_path /data/vla/wall-x/checkpoints/stack_blocks_joint/10")
        exit(1)

    config_path = f"{model_path}/config.yml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found at {config_path}")
        print(f"Checkpoint可能不完整")
        exit(1)

    save_dir = args.save_dir

    print("=" * 60)
    print("开环图绘制 - Stack Blocks Dataset")
    print("=" * 60)
    print(f"Checkpoint: {model_path}")
    print(f"Action dim: {origin_action_dim}")
    print(f"Prediction horizon: {pred_horizon}")
    print(f"Save directory: {save_dir}")
    print("=" * 60)

    # 加载训练配置
    config = load_config(config_path)
    
    # 添加processor_path（如果不存在）
    if "processor_path" not in config:
        config["processor_path"] = config.get("pretrained_wallx_path",
            "/data/vla/wall-x/hf_home/hub/models--x-square-robot--wall-oss-flow-v0.1/snapshots/325f0307b724ac4d1da4564c242c29d17f13ef9e")

    # 注册归一化器
    normalizer_action, normalizer_propri = register_normalizers(config, model_path)

    # 加载模型（使用自定义机器人配置）
    model = Qwen2_5_VLMoEForAction.from_pretrained(
        model_path,
        train_config=config,
        action_tokenizer_path=None  # flow模式不需要action tokenizer
    )

    model.set_normalizer(
        copy.deepcopy(normalizer_action),
        copy.deepcopy(normalizer_propri)
    )
    model.eval()
    model = model.to("cuda")
    model.to_bfloat16_for_selected_params()

    print("✅ 模型加载完成")

    # 获取测试数据加载器
    dataload_config = get_data_configs(config["data"])
    lerobot_config = dataload_config.get("lerobot_config", {})

    print(f"📊 数据集: {lerobot_config.get('repo_id', 'unknown')}")
    print(f"📁 数据路径: {lerobot_config.get('root', 'unknown')}")

    dataset = load_test_dataset(
        config, lerobot_config, normalizer_action, normalizer_propri, seed=42
    )
    dataloader = dataset.get_dataloader()

    total_frames = len(dataloader)
    print(f"📈 总帧数: {total_frames}")

    # 确定预测模式
    predict_mode = "fast" if config.get("use_fast_tokenizer", False) else "diffusion"
    print(f"🔮 预测模式: {predict_mode}")

    action_dim = 14 if predict_mode == "diffusion" else origin_action_dim
    gt_traj = torch.zeros((total_frames, origin_action_dim))
    pred_traj = torch.zeros((total_frames, origin_action_dim))

    # 动作维度名称（用于绘图标签）
    action_labels = [
        'Left Joint 1', 'Left Joint 2', 'Left Joint 3', 'Left Joint 4',
        'Left Joint 5', 'Left Joint 6', 'Left Gripper',
        'Right Joint 1', 'Right Joint 2', 'Right Joint 3', 'Right Joint 4',
        'Right Joint 5', 'Right Joint 6', 'Right Gripper'
    ]

    # 使用tqdm显示进度
    print("🚀 开始预测...")
    for idx, batch in tqdm(
        enumerate(dataloader), total=total_frames, desc="Predicting"
    ):
        if idx % pred_horizon == 0 and idx + pred_horizon < total_frames:
            batch = batch.to("cuda")
            with torch.no_grad():
                if predict_mode == "fast":
                    outputs = model.predict(
                        **batch,
                        action_dim=action_dim,
                        pred_horizon=pred_horizon,
                        predict_mode=predict_mode,
                    )
                else:
                    outputs = model(
                        **batch,
                        action_dim=action_dim,
                        action_horizon=pred_horizon,
                        mode="predict",
                        predict_mode=predict_mode,
                    )
                if outputs.get("predict_action") is not None:
                    pred_traj[idx : idx + pred_horizon] = (
                        outputs["predict_action"][:, :, :origin_action_dim]
                        .detach()
                        .cpu()
                        .squeeze(0)
                    )
                else:
                    print(f"⚠️  Warning: predict_action is None at idx {idx}, skipping")

            # 反归一化真实动作
            gt_action_chunk = batch["action_chunk"][:, :, :origin_action_dim]
            dof_mask = batch["dof_mask"].to(gt_action_chunk.dtype)
            denormalized_gt = (
                model.action_preprocessor.normalizer_action.unnormalize_data(
                    gt_action_chunk,
                    [lerobot_config.get("repo_id", "stack_blocks_test_v1")],
                    dof_mask,
                ).squeeze(0)
            )
            gt_traj[idx : idx + pred_horizon] = denormalized_gt.detach().cpu()

    print("✅ 预测完成")

    gt_traj_np = gt_traj.numpy()
    pred_traj_np = pred_traj.numpy()

    timesteps = gt_traj.shape[0]

    # 绘图
    fig, axs = plt.subplots(
        origin_action_dim, 1, figsize=(20, 3 * origin_action_dim), sharex=True
    )
    fig.suptitle("Stack Blocks - Action Prediction (Open Loop)", fontsize=16)

    for i in range(origin_action_dim):
        axs[i].plot(range(timesteps), gt_traj_np[:, i], label="Ground Truth", linewidth=2)
        axs[i].plot(range(timesteps), pred_traj_np[:, i], label="Prediction", linewidth=2, alpha=0.7)
        axs[i].set_ylabel(f"{action_labels[i]}\n(rad)", fontsize=10)
        axs[i].legend(loc='upper right')
        axs[i].grid(True, alpha=0.3)

        # 计算并显示MSE
        mse = ((gt_traj_np[:, i] - pred_traj_np[:, i]) ** 2).mean()
        axs[i].text(0.02, 0.98, f'MSE: {mse:.6f}',
                   transform=axs[i].transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axs[-1].set_xlabel("Timestep", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"stack_blocks_comparison_ep{os.path.basename(model_path)}.png")
    plt.savefig(save_path, dpi=150)
    print(f"💾 图表已保存到: {save_path}")
    plt.close()

    # 计算总体统计
    overall_mse = ((gt_traj_np - pred_traj_np) ** 2).mean()
    print(f"\n📊 总体MSE: {overall_mse:.6f}")

    # 计算每个维度的MSE
    print("\n📊 各维度MSE:")
    for i in range(origin_action_dim):
        mse = ((gt_traj_np[:, i] - pred_traj_np[:, i]) ** 2).mean()
        print(f"  {action_labels[i]}: {mse:.6f}")
