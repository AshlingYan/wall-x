#!/usr/bin/env python3
"""
Wall-X 推理服务器 - 实际推理版本
"""
import sys
import os
sys.path.append("/data/vla/wall-x")
sys.path.append("./")
os.environ["INFO_LEVEL"] = "INFO"

import socket
import time
import numpy as np
import cv2
import yaml
import torch
from PIL import Image

from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from utils.bisocket import BiSocket
from utils.data_handler import debug_print
from wall_x.data.utils import preprocesser_call


class WallXServer:
    def __init__(self, checkpoint_path, device="cuda:0"):
        self.device = device
        debug_print("Server", f"Loading model from {checkpoint_path}", "INFO")

        # 加载配置
        config_path = os.path.join(checkpoint_path, "config.yml")
        with open(config_path, "r") as f:
            self.train_config = yaml.safe_load(f)

        debug_print("Server", f"Model type: {self.train_config.get('model_type')}", "INFO")
        debug_print("Server", f"Action horizon: {self.train_config.get('data', {}).get('action_horizon_flow', 10)}", "INFO")

        # 加载模型
        self.model = Qwen2_5_VLMoEForAction.from_pretrained(checkpoint_path)
        self.model.eval()
        self.model = self.model.to(device)
        self.model = self.model.bfloat16()

        debug_print("Server", "Model loaded successfully", "INFO")

    def decode_images(self, message):
        """解码图像"""
        img_arr = []
        for data in message["img_arr"]:
            jpeg_bytes = np.array(data).tobytes().rstrip(b"\0")
            nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(nparr, 1)
            img_arr.append(img)
        return img_arr

    def get_action(self, img_arr, state):
        """使用状态和图像生成动作"""
        state_dim = len(state)
        batch_size = 1

        # 构造文本 - 使用正确的 Wall-X 格式
        role_start_symbol = "<|im_start|>"
        role_end_symbol = "<|im_end|>"
        vision_start_symbol = "<|vision_start|>"
        vision_end_symbol = "<|vision_end|>"
        image_pad_symbol = "<|image_pad|>"
        propri_symbol = "<|propri|>"
        action_symbol = "<|action|>"

        # System prompt
        prologue = f"{role_start_symbol}system\nYou are a helpful assistant.{role_end_symbol}\n"

        # User request with observations
        user_request = f"{role_start_symbol}user\nObservation:"
        camera_names = ["face_view", "left_wrist_view", "right_wrist_view"]
        for cam_name in camera_names:
            user_request += f" {cam_name}: {vision_start_symbol}{image_pad_symbol}{vision_end_symbol}"
        user_request += "\nInstruction:"

        instruction = "Stack the blocks"
        text_prompt = f"\nPredict the next action in robot action.\nProprioception: {propri_symbol}\n"
        user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"

        # Assistant response
        action_chunk_size = 32  # 基于频率映射
        assistant_message = f"{role_start_symbol}assistant\n{action_symbol * action_chunk_size}"

        # 完整文本
        text = prologue + user_message + assistant_message

        # 转换图像为 PIL Image
        pil_images = []
        for img in img_arr:
            if isinstance(img, np.ndarray):
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(img_rgb))

        debug_print("Server", f"Processing {len(pil_images)} images", "DEBUG")

        # 使用 Wall-X 的自定义预处理
        inputs = preprocesser_call(
            processor=self.model.processor,
            images=pil_images if len(pil_images) > 0 else None,
            text=text,
            return_tensors="pt"
        )

        debug_print("Server", f"Input keys: {list(inputs.keys())}", "DEBUG")

        # 处理状态
        proprioception = torch.tensor(state, dtype=torch.float32)
        proprioception = proprioception.unsqueeze(0).unsqueeze(0)

        seq_length = inputs['input_ids'].shape[1]

        attention_mask = inputs['attention_mask']
        moe_token_types = torch.zeros((batch_size, seq_length), dtype=torch.long)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        agent_pos_mask = torch.ones((batch_size, 1, state_dim), dtype=torch.float32)
        dof_mask = torch.ones((batch_size, 32, state_dim), dtype=torch.float32)

        dataset_names = ["stack_blocks"]

        # 移到 GPU
        device = self.device
        inputs_moved = {
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device),
        }

        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            inputs_moved["pixel_values"] = inputs["pixel_values"].to(device)
            inputs_moved["image_grid_thw"] = inputs["image_grid_thw"].to(device)

        moe_token_types = moe_token_types.to(device)
        position_ids = position_ids.to(device)
        proprioception = proprioception.to(device).bfloat16()
        agent_pos_mask = agent_pos_mask.to(device).bfloat16()
        dof_mask = dof_mask.to(device).bfloat16()

        # 运行推理
        action_horizon = self.train_config.get('data', {}).get('action_horizon_flow', 10)
        dof_config = self.train_config.get('dof_config', {})
        action_dim = sum(dof_config.values())

        debug_print("Server", f"Running inference: action_horizon={action_horizon}, action_dim={action_dim}", "DEBUG")

        with torch.no_grad():
            try:
                output = self.model.generate_flow_action(
                    input_ids=inputs_moved["input_ids"],
                    action_horizon=action_horizon,
                    action_dim=action_dim,
                    num_inference_timesteps=10,
                    attention_mask=inputs_moved.get("attention_mask"),
                    position_ids=inputs_moved.get("position_ids"),
                    moe_token_types=moe_token_types,
                    proprioception=proprioception,
                    agent_pos_mask=agent_pos_mask,
                    dof_mask=dof_mask,
                    pixel_values=inputs_moved.get("pixel_values"),
                    image_grid_thw=inputs_moved.get("image_grid_thw"),
                    dataset_names=dataset_names,
                    unnorm=False,
                )

                if isinstance(output, torch.Tensor):
                    action = output.cpu().float().numpy()
                else:
                    action = output

                debug_print("Server", f"Inference successful, output shape: {action.shape}", "DEBUG")

                # 反归一化
                checkpoint_path = "/data/vla/wall-x/checkpoints/stack_blocks_joint/4"
                action = self.denormalize_action(action, checkpoint_path)

                return action

            except Exception as e:
                debug_print("Server", f"Inference failed: {e}", "ERROR")
                import traceback
                traceback.print_exc()

                # 如果推理失败，返回小幅随机动作
                action = np.random.randn(14).astype(np.float32) * 0.01  # 非常小的动作
                return action

    def denormalize_action(self, action, checkpoint_path, dataset_name="stack_blocks"):
        """反归一化"""
        normalizer_path = os.path.join(checkpoint_path, "normalizer_action.pth")
        if os.path.exists(normalizer_path):
            from wall_x.model.action_head import Normalizer
            normalizer = torch.load(normalizer_path)
            action_tensor = torch.tensor(action).unsqueeze(0)
            action_denorm = normalizer.unnormalize_data(action_tensor, [dataset_name])
            return action_denorm.squeeze(0).numpy()

        return action

    def infer(self, message):
        """推理接口"""
        debug_print("Server", "Inference triggered", "INFO")

        # 解码图像
        img_arr = self.decode_images(message)
        state = message["state"]

        debug_print("Server", f"Decoded {len(img_arr)} images, state dim: {len(state)}", "DEBUG")

        # 获取动作
        action = self.get_action(img_arr, state)

        debug_print("Server", f"Action generated: shape={action.shape}, range=[{action.min():.3f}, {action.max():.3f}]", "DEBUG")

        return {"action": action}

    def close(self):
        pass


def main():
    SERVER_IP = "0.0.0.0"
    SERVER_PORT = 10000
    CHECKPOINT_PATH = "/data/vla/wall-x/checkpoints/stack_blocks_joint/4"

    debug_print("Server", f"Starting Wall-X Server (full inference)", "INFO")
    debug_print("Server", f"Checkpoint: {CHECKPOINT_PATH}", "INFO")

    server = WallXServer(CHECKPOINT_PATH)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((SERVER_IP, SERVER_PORT))
    server_socket.listen(1)

    debug_print("Server", f"Listening on {SERVER_IP}:{SERVER_PORT}", "INFO")
    debug_print("Server", "Waiting for client connection...", "INFO")

    try:
        while True:
            conn, addr = server_socket.accept()
            debug_print("Server", f"Connected by {addr}", "INFO")

            bisocket = BiSocket(conn, server.infer, send_back=True)

            while bisocket.running.is_set():
                time.sleep(0.5)

            debug_print("Server", "Client disconnected", "WARNING")

    except KeyboardInterrupt:
        debug_print("Server", "Shutting down...", "WARNING")
    finally:
        server_socket.close()


if __name__ == "__main__":
    main()
