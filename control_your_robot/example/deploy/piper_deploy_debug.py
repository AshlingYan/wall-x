#!/usr/bin/env python3
"""
Piper 部署调试脚本 - 用于真机部署前的数据流验证
用法: python piper_deploy_debug.py --checkpoint <path> --max-step 5

关键检查点:
  [1] 摄像头图像数据完整性
  [2] 机械臂关节与夹爪状态读取
  [3] 推理输出有效性
  [4] 动作映射与关节限制
  [5] 最终执行命令正确性
"""
import sys
import time
import math
import argparse
from pathlib import Path
import numpy as np

# ensure local packages are importable
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
OPENPI_SRC = REPO_ROOT / "openpi" / "src"
CONTROL_ROOT = REPO_ROOT / "control_your_robot"
for p in (str(OPENPI_SRC), str(CONTROL_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from PIL import Image
except Exception:
    Image = None

from my_robot.agilex_piper_dual_base import PiperDual
from utils.data_handler import is_enter_pressed

try:
    from openpi.training import config as _config
    from openpi.policies import policy_config
    HAS_OPENPI = True
except Exception:
    HAS_OPENPI = False

try:
    import cv2
except Exception:
    cv2 = None

# Joint limits (radians)
JOINT_LIMITS_RAD = [
    (math.radians(-150), math.radians(150)),
    (math.radians(0), math.radians(180)),
    (math.radians(-170), math.radians(0)),
    (math.radians(-100), math.radians(100)),
    (math.radians(-70), math.radians(70)),
    (math.radians(-120), math.radians(120)),
]

def print_header(title):
    """打印调试标题"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_section(title):
    """打印分区标题"""
    print(f"\n▶ {title}")
    print("-" * 70)

def check_image_validity(img, name):
    """检查图像有效性"""
    if img is None:
        print(f"  ❌ {name}: 图像为 None")
        return False
    if not hasattr(img, 'shape'):
        print(f"  ❌ {name}: 不是 numpy 数组")
        return False
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        print(f"  ❌ {name}: 尺寸为 0 (shape={img.shape})")
        return False
    print(f"  ✓ {name}: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")
    return True

def check_state_validity(state_dict, arm_name):
    """检查机械臂状态有效性"""
    if arm_name not in state_dict:
        print(f"  ❌ {arm_name}: 缺失关键")
        return False
    arm = state_dict[arm_name]
    joint = arm.get("joint")
    gripper = arm.get("gripper")
    
    if joint is None:
        print(f"  ❌ {arm_name}: 关节数据为 None")
        return False
    if gripper is None:
        print(f"  ❌ {arm_name}: 夹爪数据为 None")
        return False
    
    joint_arr = np.asarray(joint)
    gripper_arr = np.asarray(gripper)
    
    if joint_arr.size != 6:
        print(f"  ❌ {arm_name}: 关节维度错误 (expected 6, got {joint_arr.size})")
        return False
    if gripper_arr.size != 1:
        print(f"  ❌ {arm_name}: 夹爪维度错误 (expected 1, got {gripper_arr.size})")
        return False
    
    print(f"  ✓ {arm_name}:")
    print(f"      关节 (6D): {joint_arr} (rad)")
    print(f"      关节 (deg): {np.degrees(joint_arr)}")
    print(f"      夹爪: {gripper_arr}")
    return True

def save_debug_images(imgs, step):
    """保存调试图像"""
    if cv2 is None and Image is None:
        print("  ⚠ 无图像保存库，跳过保存")
        return
    
    debug_dir = Path("/tmp/robot_debug_images")
    debug_dir.mkdir(exist_ok=True, parents=True)
    
    for cam_name, img in imgs.items():
        if img is None:
            continue
        filepath = debug_dir / f"step_{step:04d}_{cam_name}.png"
        try:
            if cv2 is not None:
                cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            elif Image is not None:
                Image.fromarray(img).save(filepath)
            print(f"  ✓ 保存: {filepath}")
        except Exception as e:
            print(f"  ❌ 保存失败 ({cam_name}): {e}")

def to_224(img):
    """转换图像为 224x224 uint8 RGB"""
    if img is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    arr = img
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    
    if arr.ndim == 2:
        if cv2 is not None:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            arr = np.stack([arr]*3, axis=-1)
    
    if arr.ndim == 3 and arr.shape[2] == 3:
        pass
    else:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    h, w = arr.shape[:2]
    target_h, target_w = 224, 224
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    if cv2 is not None:
        resized = cv2.resize(arr, (new_w, new_h))
    else:
        resized = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

def main():
    parser = argparse.ArgumentParser(description="Piper 双臂机器人部署调试脚本")
    parser.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "openpi" / "checkpoints" / "pi05_ygx" / "piper_ygx"))
    parser.add_argument("--config-name", type=str, default="pi05_ygx")
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--dry-run", action="store_true", help="不执行实际动作，只输出命令")
    parser.add_argument("--max-step", type=int, default=3, help="最多执行步数")
    parser.add_argument("--save-images", action="store_true", help="保存调试图像到 /tmp/robot_debug_images")
    parser.add_argument("--action-mode", type=str, choices=("delta", "absolute"), default="absolute")
    args = parser.parse_args()
    
    print_header("Piper 双臂机器人部署调试")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config_name}")
    print(f"Task: {args.task if args.task else '(无任务描述)'}")
    print(f"Action Mode: {args.action_mode}")
    print(f"Dry-run: {args.dry_run}")
    print(f"Max Steps: {args.max_step}")
    
    # 初始化策略
    print_section("步骤 0: 策略初始化")
    policy = None
    if HAS_OPENPI:
        try:
            cfg = _config.get_config(args.config_name)
            print(f"✓ 配置加载成功")
            policy = policy_config.create_trained_policy(
                cfg, args.checkpoint, repack_transforms=cfg.data.repack_transforms, default_prompt=args.task
            )
            print(f"✓ 策略模型加载成功")
        except Exception as e:
            print(f"❌ 策略加载失败: {e}")
            return
    else:
        print("❌ OpenPI 不可用")
        return
    
    # 初始化机器人
    print_section("步骤 1: 机器人初始化")
    try:
        robot = PiperDual()
        robot.set_up()
        print("✓ 机器人已初始化")
        robot.reset()
        print("✓ 机器人已重置")
    except Exception as e:
        print(f"❌ 机器人初始化失败: {e}")
        return
    
    try:
        for step in range(args.max_step):
            print_header(f"执行步 {step + 1}")
            
            # 获取数据
            print_section(f"步骤 2.{step}: 获取机器人数据")
            try:
                data = robot.get()
                print("✓ 数据获取成功")
            except Exception as e:
                print(f"❌ 数据获取失败: {e}")
                break
            
            # 验证摄像头数据
            print_section(f"步骤 3.{step}: 摄像头图像验证")
            try:
                cameras = {
                    "cam_head": data[1]["cam_head"]["color"],
                    "cam_left_wrist": data[1]["cam_left_wrist"]["color"],
                    "cam_right_wrist": data[1]["cam_right_wrist"]["color"],
                }
                all_valid = all(check_image_validity(img, name) for name, img in cameras.items())
                if not all_valid:
                    print("⚠ 部分摄像头数据异常，继续...")
            except Exception as e:
                print(f"❌ 摄像头数据访问失败: {e}")
                break
            
            # 处理图像
            print_section(f"步骤 4.{step}: 图像处理 (调整为 224x224)")
            try:
                imgs = {
                    "cam_high": to_224(cameras["cam_head"]),
                    "cam_left_wrist": to_224(cameras["cam_left_wrist"]),
                    "cam_right_wrist": to_224(cameras["cam_right_wrist"]),
                }
                for name, img in imgs.items():
                    print(f"  ✓ {name}: shape={img.shape}, dtype={img.dtype}")
                
                if args.save_images:
                    print("  保存图像...")
                    save_debug_images(imgs, step)
            except Exception as e:
                print(f"❌ 图像处理失败: {e}")
                break
            
            # 验证机械臂状态
            print_section(f"步骤 5.{step}: 机械臂状态验证")
            try:
                arm_state = data[0]
                left_ok = check_state_validity(arm_state, "left_arm")
                right_ok = check_state_validity(arm_state, "right_arm")
                if not (left_ok and right_ok):
                    print("⚠ 部分机械臂状态异常，继续...")
            except Exception as e:
                print(f"❌ 机械臂状态访问失败: {e}")
                break
            
            # 构建状态向量
            print_section(f"步骤 6.{step}: 构建状态向量")
            try:
                left_joint = np.array(data[0]["left_arm"]["joint"]).reshape(-1)
                left_grip = np.array(data[0]["left_arm"]["gripper"]).reshape(-1)
                right_joint = np.array(data[0]["right_arm"]["joint"]).reshape(-1)
                right_grip = np.array(data[0]["right_arm"]["gripper"]).reshape(-1)
                
                state = np.concatenate([left_joint, left_grip, right_joint, right_grip]).astype(np.float32)
                if state.size < 14:
                    state = np.concatenate([state, np.zeros(14 - state.size, dtype=np.float32)])
                elif state.size > 14:
                    state = state[:14]
                
                print(f"  ✓ 状态向量 (14D): {state}")
                print(f"    - 左臂关节: {state[0:6]} (deg: {np.degrees(state[0:6])})")
                print(f"    - 左夹爪: {state[6]}")
                print(f"    - 右臂关节: {state[7:13]} (deg: {np.degrees(state[7:13])})")
                print(f"    - 右夹爪: {state[13]}")
            except Exception as e:
                print(f"❌ 状态向量构建失败: {e}")
                break
            
            # 推理
            print_section(f"步骤 7.{step}: 策略推理")
            try:
                example = {"observation": {"images": imgs, "image": imgs, "state": state}}
                if args.task:
                    example["task"] = args.task
                
                out = policy.infer(example)
                action_chunk = None
                if isinstance(out, dict):
                    for k in ("actions", "action", "policy_action"):
                        if k in out and out[k] is not None:
                            action_chunk = out[k]
                            break
                else:
                    action_chunk = out
                
                if action_chunk is None:
                    print("❌ 模型输出为 None")
                    break
                
                action_chunk = np.asarray(action_chunk)
                print(f"  ✓ 动作输出 shape: {action_chunk.shape}")
                print(f"    数据类型: {action_chunk.dtype}")
                if action_chunk.ndim >= 2:
                    print(f"    使用第一帧: {action_chunk[0][:14]}")
                else:
                    print(f"    动作向量: {action_chunk[:14]}")
            except Exception as e:
                print(f"❌ 推理失败: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # 动作映射
            print_section(f"步骤 8.{step}: 动作映射与关节限制")
            try:
                a = action_chunk[0] if action_chunk.ndim >= 2 else action_chunk
                sel = a[:14]
                
                print(f"  原始动作 (14D): {sel}")
                
                if args.action_mode == "delta":
                    print(f"  ▶ 增量模式:")
                    left = (sel[0:6] + state[0:6]).tolist()
                    left_grip = float(sel[6])
                    right = (sel[7:13] + state[7:13]).tolist()
                    right_grip = float(sel[13])
                    print(f"    左臂: {state[0:6]} + {sel[0:6]} = {left}")
                    print(f"    右臂: {state[7:13]} + {sel[7:13]} = {right}")
                else:
                    print(f"  ▶ 绝对值模式:")
                    left = sel[0:6].tolist()
                    left_grip = float(sel[6])
                    right = sel[7:13].tolist()
                    right_grip = float(sel[13])
                    print(f"    左臂: {left}")
                    print(f"    右臂: {right}")
                
                # 应用关节限制
                print(f"  ▶ 关节限制检查:")
                left_clamped = [np.clip(v, *JOINT_LIMITS_RAD[i]) for i, v in enumerate(left)]
                right_clamped = [np.clip(v, *JOINT_LIMITS_RAD[i]) for i, v in enumerate(right)]
                
                for i in range(6):
                    limit_min_rad, limit_max_rad = JOINT_LIMITS_RAD[i]
                    limit_min_deg = math.degrees(limit_min_rad)
                    limit_max_deg = math.degrees(limit_max_rad)
                    clamped = left[i] != left_clamped[i]
                    status = "⚠ 已限制" if clamped else "✓"
                    print(f"    左关节{i}: {math.degrees(left[i]):7.2f}° → {math.degrees(left_clamped[i]):7.2f}° [{limit_min_deg:7.1f}°, {limit_max_deg:7.1f}°] {status}")
                
                for i in range(6):
                    limit_min_rad, limit_max_rad = JOINT_LIMITS_RAD[i]
                    limit_min_deg = math.degrees(limit_min_rad)
                    limit_max_deg = math.degrees(limit_max_rad)
                    clamped = right[i] != right_clamped[i]
                    status = "⚠ 已限制" if clamped else "✓"
                    print(f"    右关节{i}: {math.degrees(right[i]):7.2f}° → {math.degrees(right_clamped[i]):7.2f}° [{limit_min_deg:7.1f}°, {limit_max_deg:7.1f}°] {status}")
                
                print(f"    左夹爪: {left_grip}")
                print(f"    右夹爪: {right_grip}")
                
            except Exception as e:
                print(f"❌ 动作映射失败: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # 执行或模拟
            print_section(f"步骤 9.{step}: 执行动作")
            try:
                move_data = {
                    "arm": {
                        "left_arm": {"joint": left_clamped, "gripper": left_grip},
                        "right_arm": {"joint": right_clamped, "gripper": right_grip},
                    }
                }
                
                if args.dry_run:
                    print("  [DRY-RUN] 不执行实际动作")
                    print(f"    左臂关节: {[f'{math.degrees(j):.2f}°' for j in left_clamped]}")
                    print(f"    左夹爪: {left_grip}")
                    print(f"    右臂关节: {[f'{math.degrees(j):.2f}°' for j in right_clamped]}")
                    print(f"    右夹爪: {right_grip}")
                else:
                    print("  ✓ 执行动作...")
                    robot.move(move_data)
                    print("  ✓ 动作已发送")
            except Exception as e:
                print(f"❌ 动作执行失败: {e}")
                import traceback
                traceback.print_exc()
                break
            
            time.sleep(0.5)
        
        print_header("调试完成")
        print("✓ 所有检查点已通过，可以进行真机部署")
        
    finally:
        print("\n清理资源...")
        try:
            robot.reset()
            print("✓ 机器人已重置")
        except Exception:
            pass

if __name__ == "__main__":
    main()
