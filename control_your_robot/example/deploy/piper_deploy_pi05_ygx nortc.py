#!/usr/bin/env python3
#!/usr/bin/env python3
"""Piper deployment script adapted to OpenPI policy API (pi0.5_ygx).
Usage: adjust --checkpoint and --config-name as needed. Supports --dry-run.


运行示例：

（1）action只输出，不执行：
python control_your_robot/example/deploy/piper_deploy_pi05_ygx.py \
  --remote-ws 127.0.0.1:8005 \
  --dry-run \
  --auto-start \
  --task "There are three blocks on the table, the color of the blocks is red, green and blue. Stack blue on green, green on red."
 
（2）action执行：  
python control_your_robot/example/deploy/piper_deploy_pi05_ygx.py \
  --remote-ws 127.0.0.1:8005 \
  --max-step 100 \
  --max-queue-size 50 \
  --task "There are three blocks on the table, the color of the blocks is red, green and blue. Stack blue on green, green on red."

"""
import sys
import time
import math
import argparse
from pathlib import Path
import numpy as np

# ensure local packages are importable (openpi in openpi/src)
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
OPENPI_SRC = REPO_ROOT / "openpi" / "src"
CONTROL_ROOT = REPO_ROOT / "control_your_robot"
for p in (str(OPENPI_SRC), str(CONTROL_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from openpi_client import image_tools
except Exception:
    image_tools = None

try:
    from PIL import Image
except Exception:
    Image = None

def _local_convert_to_uint8(img):
    a = img
    if a is None:
        return np.zeros((224,224,3), dtype=np.uint8)
    if a.dtype == np.float32 or a.dtype == np.float64:
        a = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
        return a
    if a.dtype == np.uint16:
        # scale to 0..255
        a = (np.clip(a, 0, 65535) / 256).astype(np.uint8)
        return a
    if a.dtype == np.uint8:
        return a
    # other types
    return a.astype(np.uint8)

def _local_resize_with_pad(img, target_w, target_h):
    if img is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    arr = img
    # convert single-channel to 3-channel
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    h, w = arr.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if Image is not None:
        try:
            pil = Image.fromarray(arr)
            pil = pil.resize((new_w, new_h), resample=Image.BILINEAR)
            resized = np.asarray(pil)
        except Exception:
            resized = arr
            # fallback to basic numpy nearest-neighbor resize (very slow/low-quality)
            resized = resized.repeat(new_h//h+1, axis=0)[:new_h,:,:].repeat(new_w//w+1, axis=1)[:,:new_w,:]
    else:
        # if cv2 available do resizing there
        if cv2 is not None:
            resized = cv2.resize(arr, (new_w, new_h))
        else:
            # best-effort naive repeat
            resized = arr
            resized = resized.repeat(new_h//h+1, axis=0)[:new_h,:,:].repeat(new_w//w+1, axis=1)[:,:new_w,:]
    canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

from my_robot.agilex_piper_dual_base import PiperDual
from utils.data_handler import is_enter_pressed

# Optional remote clients
try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
except Exception:
    WebsocketClientPolicy = None

import requests
import json_numpy

# Try OpenPI policy API, fallback to legacy PI0_DUAL if available
try:
    from openpi.training import config as _config
    from openpi.policies import policy_config
    HAS_OPENPI = True
except Exception:
    HAS_OPENPI = False
    try:
        from policy.openpi.inference_model import PI0_DUAL
    except Exception:
        PI0_DUAL = None

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False

# Joint limits (radians)
JOINT_LIMITS_RAD = [
    (math.radians(-150), math.radians(150)),
    (math.radians(0), math.radians(180)),
    (math.radians(-170), math.radians(0)),
    (math.radians(-100), math.radians(100)),
    (math.radians(-70), math.radians(70)),
    (math.radians(-120), math.radians(120)),
]

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def to_224(img):
    # Normalize/resize image to 224x224 uint8 RGB.
    # Prefer openpi_client.image_tools.resize_with_pad + convert_to_uint8 when available
    if img is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    # If image_tools available, use its resize+pad and conversion utilities
    if image_tools is not None:
        try:
            out = image_tools.resize_with_pad(img, 224, 224)
            out = image_tools.convert_to_uint8(out)
            return out
        except Exception:
            pass

    # Fallback: use cv2 when available
    arr = img
    # If float in 0..1, convert to uint8
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)

    # If single-channel, convert to 3-channel BGR
    if arr.ndim == 2:
        if cv2 is not None:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            arr = np.stack([arr]*3, axis=-1)

    # Ensure 3 channels
    if arr.ndim == 3 and arr.shape[2] == 3:
        pass
    else:
        # unexpected shape -> return black 224x224
        return np.zeros((224, 224, 3), dtype=np.uint8)

    # Use cv2 to resize with padding to 224x224 while preserving aspect ratio
    h, w = arr.shape[:2]
    target_h, target_w = 224, 224
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    if cv2 is not None:
        resized = cv2.resize(arr, (new_w, new_h))
    else:
        # naive numpy resize fallback (not ideal) - center crop/resize via simple repeat
        resized = np.asarray(Image.fromarray(arr).resize((new_w, new_h))) if 'Image' in globals() else np.zeros((new_h, new_w, 3), dtype=np.uint8)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

def build_example_from_data(data, task_text=None):
    head = data[1]["cam_head"]["color"]
    left = data[1]["cam_left_wrist"]["color"]
    right = data[1]["cam_right_wrist"]["color"]
    imgs = {
        "cam_high": to_224(head),
        "cam_left_wrist": to_224(left),
        "cam_right_wrist": to_224(right),
    }
    # build 14-dim proprioceptive state: left(6)+left_grip(1)+right(6)+right_grip(1)
    state = np.concatenate([
        np.array(data[0]["left_arm"]["joint"]).reshape(-1),
        np.array(data[0]["left_arm"]["gripper"]).reshape(-1),
        np.array(data[0]["right_arm"]["joint"]).reshape(-1),
        np.array(data[0]["right_arm"]["gripper"]).reshape(-1),
    ]).astype(np.float32)
    # Ensure state is exactly 14 dims (pad or truncate) to match pi05_ygx expectations
    if state.size < 14:
        pad = np.zeros(14 - state.size, dtype=np.float32)
        state = np.concatenate([state, pad])
    elif state.size > 14:
        state = state[:14]

    # Provide both `images` (legacy) and `image` (repacked/keyed) to match repack_transforms
    example = {"observation": {"images": imgs, "image": imgs, "state": state}}
    if task_text:
        example["task"] = task_text
    return example, state

def map_model_to_robot(action_vec, current_state, mode="delta"):
    a = np.asarray(action_vec).reshape(-1)
    if a.shape[0] < 14:
        raise RuntimeError("model output dim < 14")
    sel = a[:14]
    # mode: 'delta' -> model outputs are deltas to add to current state
    #       'absolute' -> model outputs are absolute target values
    if mode == "delta":
        left = (sel[0:6] + current_state[0:6]).tolist()
        left_grip = float(sel[6])
        right = (sel[7:13] + current_state[7:13]).tolist()
        right_grip = float(sel[13])
    else:
        left = sel[0:6].tolist()
        left_grip = float(sel[6])
        right = sel[7:13].tolist()
        right_grip = float(sel[13])
    # clamp joints
    left = [clamp(v, *JOINT_LIMITS_RAD[i]) for i, v in enumerate(left)]
    right = [clamp(v, *JOINT_LIMITS_RAD[i]) for i, v in enumerate(right)]
    move_inner = {
        "left_arm": {"joint": left, "gripper": left_grip},
        "right_arm": {"joint": right, "gripper": right_grip},
    }
    return {"arm": move_inner}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "openpi" / "checkpoints" / "pi05_ygx" / "piper_ygx"))
    parser.add_argument("--config-name", type=str, default="pi05_ygx")
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--simulate", action="store_true", help="Simulate model outputs without an inference backend")
    parser.add_argument("--action-mode", type=str, choices=("delta","absolute"), default="absolute", help="Interpret model outputs as 'delta' (add to current state) or 'absolute' (use directly)")
    parser.add_argument("--auto-start", action="store_true", help="Skip waiting for Enter and start immediately (useful for automated tests)")
    parser.add_argument("--max-step", type=int, default=1000)
    parser.add_argument("--num-episode", type=int, default=1)
    parser.add_argument("--remote-ws", type=str, default="", help="Connect to remote websocket policy server as host:port")
    parser.add_argument("--remote-http", type=str, default="", help="Connect to remote HTTP policy server as host:port (POST /act)")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV image display to save memory")
    parser.add_argument("--max-queue-size", type=int, default=100, help="Maximum action queue size to prevent memory overflow")
    parser.add_argument("--use-latest-action", action="store_true", help="Only use the latest action from queue, discard old ones")
    args = parser.parse_args()

    print("HAS_OPENPI:", HAS_OPENPI)
    
    # 检查系统资源
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"💾 Memory: {mem.used/1024**3:.1f}GB / {mem.total/1024**3:.1f}GB used ({mem.percent}%)")
        if mem.percent > 80:
            print("⚠️  Warning: High memory usage detected. Consider using --no-display flag.")
    except:
        pass
    policy = None
    model_wrapper = None
    simulate = args.simulate
    action_mode = args.action_mode
    auto_start = args.auto_start

    # Remote websocket client: if provided, prefer this as the policy provider
    def _make_ws_policy(host: str, port: int, retries: int = 5, delay: float = 2.0):
        if WebsocketClientPolicy is None:
            raise RuntimeError("WebsocketClientPolicy not available in this environment")
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                print(f"Attempting WebSocket connect to {host}:{port} (attempt {attempt})")
                return WebsocketClientPolicy(host=host, port=port)
            except Exception as e:
                last_exc = e
                print(f"WebSocket connect failed (attempt {attempt}): {e}")
                time.sleep(delay)
        raise RuntimeError(f"Failed to connect WebSocket after {retries} attempts") from last_exc

    if args.remote_ws:
        try:
            host, port = args.remote_ws.split(":") if ":" in args.remote_ws else (args.remote_ws, 8000)
            port = int(port)
            print(f"Using remote WebSocket policy at {host}:{port}")
            policy = _make_ws_policy(host, port, retries=6, delay=2.0)
        except Exception as e:
            print(f"Failed to connect websocket policy: {e}")

    # Remote HTTP client wrapper (simple POST /act compatible with XVLA-like servers)
    if policy is None and args.remote_http:
        try:
            host, port = args.remote_http.split(":") if ":" in args.remote_http else (args.remote_http, 8000)
            port = int(port)
            url = f"http://{host}:{port}/act"
            print(f"Using remote HTTP policy at {url}")

            class HTTPPolicyWrapper:
                def __init__(self, url):
                    self.url = url
                    self.timeout = 30

                def infer(self, obs: dict) -> dict:
                    # Expect obs to follow example structure: {"observation": {"images": {...}, "state": ndarray}, "task": ...}
                    try:
                        obsn = obs.get("observation", {})
                        imgs = obsn.get("images", {})
                        # choose high camera if available
                        img = imgs.get("cam_high") or imgs.get("cam_head") or next(iter(imgs.values()))
                        state = obsn.get("state")
                        # ensure image/state are serializable (convert to uint8/resized)
                        try:
                            img_s = to_224(img)
                        except Exception:
                            img_s = img

                        # send proprio and all three images so remote servers matching pi05_ygx can consume
                        payload = {
                            "proprio": json_numpy.dumps(np.asarray(state, dtype=np.float32)),
                            "image_cam_high": json_numpy.dumps(to_224(imgs.get("cam_high") if isinstance(imgs, dict) else img_s)),
                            "image_cam_left_wrist": json_numpy.dumps(to_224(imgs.get("cam_left_wrist") if isinstance(imgs, dict) else img_s)),
                            "image_cam_right_wrist": json_numpy.dumps(to_224(imgs.get("cam_right_wrist") if isinstance(imgs, dict) else img_s)),
                            "language_instruction": obs.get("task", ""),
                            "domain_id": 10,
                            "steps": 10,
                        }

                        resp = requests.post(self.url, json=payload, timeout=self.timeout)
                        resp.raise_for_status()
                        j = resp.json()
                        # return dict with one of expected keys
                        return {"action": np.asarray(j.get("action", j.get("actions"))) }
                    except Exception as e:
                        print(f"HTTPPolicyWrapper infer error: {e}")
                        return {"action": None}

            policy = HTTPPolicyWrapper(url)
        except Exception as e:
            print(f"Failed to set up HTTP policy wrapper: {e}")
    # Prefer a local OpenPI policy when available, otherwise keep any remote
    # policy already constructed (via --remote-ws/--remote-http). Only exit
    # if we have no policy, no legacy model wrapper, and not in simulate mode.
    if HAS_OPENPI:
        cfg = _config.get_config(args.config_name)
        print("Loading policy from", args.checkpoint)
        policy = policy_config.create_trained_policy(
            cfg, args.checkpoint, repack_transforms=cfg.data.repack_transforms, default_prompt=args.task
        )
        print("policy loaded")
    else:
        if PI0_DUAL is not None:
            print("OpenPI not available; falling back to PI0_DUAL wrapper")
            model_wrapper = PI0_DUAL(args.checkpoint, args.task)
        elif simulate:
            print("Simulation mode enabled: generating fake model outputs.")

    # If no policy/model_wrapper/simulate available at this point, exit.
    if policy is None and model_wrapper is None and not simulate:
        print("No inference backend available (openpi, remote, or PI0_DUAL). Exiting.")
        return

    # Use a fake robot in dry-run or simulation to avoid touching hardware
    if args.dry_run or simulate:
        if simulate:
            print("Using FakeRobot (simulate) — no hardware will be touched.")
            class FakeRobot:
                def __init__(self):
                    self._left_joints = [0.0] * 6
                    self._right_joints = [0.0] * 6
                def set_up(self):
                    return
                def reset(self):
                    return
                def get(self):
                    state = {
                        "left_arm": {"joint": np.array(self._left_joints), "gripper": np.array([0.0])},
                        "right_arm": {"joint": np.array(self._right_joints), "gripper": np.array([0.0])},
                    }
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    imgs = {
                        "cam_head": {"color": img.copy()},
                        "cam_left_wrist": {"color": img.copy()},
                        "cam_right_wrist": {"color": img.copy()},
                    }
                    return (state, imgs)
                def move(self, move_data):
                    print("FAKE MOVE EXECUTE:", move_data)
            robot = FakeRobot()
        else:
            # dry-run: use real robot for sensing, but don't execute moves
            print("Using dry-run mode — reading REAL camera and joint data from hardware, but NOT sending commands to robot.")
            print("📸 Attempting to connect to RealSense cameras...")
            print("🤖 Attempting to connect to Piper dual arms...")
            class DryRunRobot(PiperDual):
                def __init__(self):
                    super().__init__()
                    self.action_count = 0
                    
                def move(self, move_data):
                    self.action_count += 1
                    print(f"\n🔍 [DRY-RUN Action #{self.action_count}] MOVE NOT EXECUTED:")
                    # 解析并显示动作信息
                    if "arm" in move_data:
                        for arm_name, arm_data in move_data["arm"].items():
                            if "joint" in arm_data:
                                joints = arm_data["joint"]
                                gripper = arm_data.get("gripper", 0.0)
                                print(f"   {arm_name}:")
                                print(f"      Joints: [{', '.join(f'{j:.4f}' for j in joints)}] rad")
                                print(f"      Gripper: {gripper:.4f} ({'CLOSE' if gripper > 0.5 else 'OPEN'})")
            robot = DryRunRobot()
            robot.set_up()
    else:
        robot = PiperDual()
        robot.set_up()

    try:
        robot.reset()
        for ep in range(args.num_episode):
            print(f"Episode {ep} start")
            step = 0
            # wait for Enter (unless auto_start)
            if not auto_start:
                while not is_enter_pressed():
                    print("waiting for start command...")
                    time.sleep(1)

            action_queue = []
            # If max_step <= 0: run indefinitely until interrupted
            if args.max_step <= 0:
                run_condition = lambda s: True
            else:
                run_condition = lambda s: s < args.max_step
            
            print(f"\n{'='*60}")
            print(f"🚀 Starting Episode {ep} - Max steps: {args.max_step}")
            print(f"   Mode: {'DRY-RUN (no execution)' if args.dry_run else 'REAL EXECUTION'}")
            print(f"{'='*60}\n")
            
            while run_condition(step):
                try:
                    data = robot.get()
                except Exception as e:
                    print(f"⚠️ Error getting robot data at step {step}: {e}")
                    time.sleep(0.1)
                    continue
                
                # 显示当前步骤信息
                print(f"\n📊 [Step {step}/{args.max_step}] Reading from hardware...")
                
                # 显示关节状态
                left_joints = data[0]["left_arm"]["joint"]
                right_joints = data[0]["right_arm"]["joint"]
                left_gripper = data[0]["left_arm"]["gripper"]
                right_gripper = data[0]["right_arm"]["gripper"]
                print(f"   Left  arm: joints={np.array(left_joints).round(4).tolist()}, gripper={left_gripper:.4f}")
                print(f"   Right arm: joints={np.array(right_joints).round(4).tolist()}, gripper={right_gripper:.4f}")
                
                # 显示图像信息
                head_img = data[1]["cam_head"]["color"]
                left_img = data[1]["cam_left_wrist"]["color"]
                right_img = data[1]["cam_right_wrist"]["color"]
                print(f"   Images: head={head_img.shape if head_img is not None else None}, "
                      f"left_wrist={left_img.shape if left_img is not None else None}, "
                      f"right_wrist={right_img.shape if right_img is not None else None}")
                
                # 使用OpenCV显示图像 - 横向排列（优化内存）
                # 使用OpenCV显示图像 - 横向排列（优化内存）
                if CV2_AVAILABLE and not args.no_display:
                    try:
                        # 创建显示窗口（只需创建一次）
                        if step == 0:
                            cv2.namedWindow("Robot Cameras", cv2.WINDOW_NORMAL)
                            cv2.resizeWindow("Robot Cameras", 960, 240)
                        
                        # 准备三个图像，横向排列 - 缩小尺寸减少内存
                        imgs_to_show = []
                        for img, name in [(head_img, "Head"), (left_img, "Left"), (right_img, "Right")]:
                            if img is not None and img.size > 0:
                                # RGB -> BGR
                                display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[-1] == 3 else img
                                # 缩小到 320x240 以节省内存
                                display_img = cv2.resize(display_img, (320, 240))
                                # 添加标签
                                cv2.putText(display_img, name, (10, 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                imgs_to_show.append(display_img)
                            else:
                                # 黑色占位图
                                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                                cv2.putText(blank, name, (10, 120), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                imgs_to_show.append(blank)
                        
                        # 横向拼接：Head | Left | Right
                        combined = np.hstack(imgs_to_show)
                        
                        cv2.imshow("Robot Cameras", combined)
                        
                        # 等待1ms让窗口刷新
                        key = cv2.waitKey(1)
                        # 按 'q' 键也可以退出
                        if key == ord('q'):
                            print("\n⏸️ User pressed 'q' to quit")
                            break
                    except Exception as e:
                        print(f"⚠️ OpenCV display error: {e}")
                        # 发生错误时关闭窗口避免内存泄漏
                        try:
                            cv2.destroyAllWindows()
                        except:
                            pass
                
                example, state = build_example_from_data(data, task_text=args.task)

                # Action queue policy (important):
                # - If queue has pending actions, execute them first and DO NOT request a new chunk.
                # - Only request a new action chunk when the queue is empty.
                # This avoids executing stale chunks that were inferred from older observations,
                # which often looks like the robot "going back to origin".
                if not action_queue:
                    action_chunk = None
                    if policy is not None:
                        print("   🔮 Requesting action from policy server...")
                        try:
                            # If using the websocket client, send a flattened dict matching
                            # the training repack paths so the server's RepackTransform can
                            # pick up 'image' / 'state' / 'task' correctly.
                            try:
                                is_ws_client = WebsocketClientPolicy is not None and isinstance(policy, WebsocketClientPolicy)
                            except Exception:
                                is_ws_client = False

                            if is_ws_client:
                                imgs_map = example["observation"]["images"]
                                # Provide multiple key styles to be robust against different server repack expectations.
                                ws_obs = {
                                    # direct final-form keys (server transforms like ResizeImages expect 'image')
                                    "image": {
                                        "cam_high": imgs_map.get("cam_high"),
                                        "cam_left_wrist": imgs_map.get("cam_left_wrist"),
                                        "cam_right_wrist": imgs_map.get("cam_right_wrist"),
                                    },
                                    # numeric proprioceptive state
                                    "state": example["observation"]["state"],
                                    # tokenized prompt already numeric to avoid string leaves
                                    "tokenized_prompt": np.zeros((200,), dtype=np.int32),
                                    "tokenized_prompt_mask": np.zeros((200,), dtype=np.int32),
                                    # include a textual task/prompt so server TokenizePrompt/TaskToPrompt
                                    # can find/convert the prompt when required
                                    "task": args.task if args.task else example.get("task", ""),
                                    # also include 'prompt' key directly so TokenizePrompt can consume it
                                    "prompt": args.task if args.task else example.get("task", ""),
                                }
                                out = policy.infer(ws_obs)
                            else:
                                out = policy.infer(example)

                            # try common keys (avoid using `or` with numpy arrays — check membership/None)
                            if isinstance(out, dict):
                                for k in ("actions", "action", "policy_action"):
                                    if k in out and out[k] is not None:
                                        action_chunk = out[k]
                                        break
                            else:
                                action_chunk = out
                        except Exception as e:
                            print(f"   ❌ Policy inference failed: {e}")
                            if "ConnectionClosed" in str(type(e).__name__) or "timeout" in str(e).lower():
                                print(f"   🔄 WebSocket connection lost. Please restart the server.")
                                print(f"   🛑 Stopping execution...")
                                break
                            action_chunk = None
                    elif model_wrapper is not None:
                        # legacy wrapper
                        # PI0_DUAL expects raw images,state interface; attempt to use similar calls
                        img_arr = (example["observation"]["images"]["cam_high"],
                                   example["observation"]["images"]["cam_right_wrist"],
                                   example["observation"]["images"]["cam_left_wrist"])
                        model_wrapper.update_observation_window(img_arr, example["observation"]["state"])
                        action_chunk = model_wrapper.get_action()
                    elif simulate:
                        # deterministic fake outputs (use small sinusoids as deltas)
                        D = 32
                        sel = np.zeros(D, dtype=np.float32)
                        for i in range(14):
                            sel[i] = 0.05 * math.sin(step * 0.3 + i)
                        action_chunk = np.asarray([sel])
                    else:
                        time.sleep(0.01)
                        continue

                    if action_chunk is None:
                        time.sleep(0.01)
                        continue

                    # normalize to numpy array of frames
                    action_chunk = np.asarray(action_chunk)
                    if action_chunk.ndim == 1:
                        action_chunk = action_chunk.reshape(1, -1)

                    # Replace queue with the latest inferred chunk (avoid stale accumulation)
                    if args.use_latest_action:
                        # Keep only the latest frame when requested
                        action_queue = [action_chunk[-1]]
                    else:
                        # Enforce max_queue_size cap
                        if action_chunk.shape[0] > args.max_queue_size:
                            action_chunk = action_chunk[: args.max_queue_size]
                        action_queue = [a for a in action_chunk]

                    print(f"   ✅ Queued {len(action_queue)} action(s) from latest inference")

                # Execute one action per control step
                if action_queue:
                    a = action_queue.pop(0)
                    move_data = map_model_to_robot(a, state, mode=action_mode)
                    print(f"   🎯 Executing action (mode: {action_mode}), remaining queue: {len(action_queue)}")

                    # 显示动作变化（用于调试）
                    if step % 5 == 0:  # 每5步显示一次详细信息
                        left_delta = a[:6] - state[:6]
                        right_delta = a[7:13] - state[7:13]
                        print(f"   📊 Action delta:")
                        print(f"      Left  arm movement: {np.linalg.norm(left_delta):.4f} rad")
                        print(f"      Right arm movement: {np.linalg.norm(right_delta):.4f} rad")

                    robot.move(move_data)  # DryRunRobot会自动只打印不执行
                    step += 1
                    print(f"   ✓ Step {step} completed\n")
                else:
                    print(f"   ⚠️  No action available, waiting...")
                time.sleep(1/30)

            print(f"\n{'='*60}")
            print(f"✅ Episode {ep} finished - Executed {step} steps")
            print(f"{'='*60}")
    finally:
        # 关闭OpenCV窗口
        if CV2_AVAILABLE:
            cv2.destroyAllWindows()
        print("finished")


if __name__ == "__main__":
    main()