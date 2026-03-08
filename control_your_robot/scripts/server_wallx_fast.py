#!/usr/bin/env python3
"""
Wall-X 模型推理服务器 - 返回反归一化的末端位姿
"""
import sys
import os
sys.path.append('./')
sys.path.append('/data/vla/wall-x')
sys.path.append('/data/vla/wall-x-libero')
os.environ["INFO_LEVEL"] = "INFO"

import socket
import time
import numpy as np
import cv2
import pathlib
import copy
import yaml

from utils.bisocket import BiSocket
from utils.data_handler import debug_print

from wall_x.serving.policy.wall_x_policy import WallXPolicy

class WallXFlowServer:
    def __init__(self, checkpoint_path, config_path, action_tokenizer_path, device="cuda:0"):
        self.device_str = device
        
        with open(config_path, 'r') as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)

        debug_print("Server", "="*60, "INFO")
        debug_print("Server", "Wall-X Flow Server (Denormalized EE Pose)", "INFO")
        debug_print("Server", "="*60, "INFO")
        debug_print("Server", f"Checkpoint: {checkpoint_path}", "INFO")
        debug_print("Server", f"Device: {device}", "INFO")
        debug_print("Server", "="*60, "INFO")
        debug_print("Server", "Loading model...", "INFO")

        self.policy = WallXPolicy(
            model_path=checkpoint_path,
            train_config=train_config,
            action_tokenizer_path=action_tokenizer_path,
            action_dim=14,
            agent_pos_dim=14,
            pred_horizon=10,
            device=device,
            dtype="bfloat16",
            predict_mode="diffusion",
            default_prompt="stack the blocks",
            camera_key=["face_view", "left_wrist_view", "right_wrist_view"],
        )

        self.normalizer_action = self.policy.normalizer_action
        debug_print("Server", "Model loaded successfully!", "INFO")
        self.inference_count = 0

    def denormalize_action(self, normalized_action):
        """反归一化动作到真实末端位姿"""
        action_tensor = torch.from_numpy(normalized_action).float().unsqueeze(0)
        denorm_action = self.normalizer_action.unnormalize_data(
            action_tensor, 
            ["stack_blocks_three_merged"]
        )
        return denorm_action[0].cpu().numpy()

    def infer(self, message):
        """Flow模式推理：返回反归一化的末端位姿"""
        inference_start = time.time()
        
        # 解码图像
        img_arr = []
        for data in message["img_arr"]:
            jpeg_bytes = np.array(data).tobytes().rstrip(b'\0')
            nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(nparr, 1)
            img_arr.append(img)

        state = message["state"]
        
        # 准备obs
        obs = {}
        obs["face_view"] = img_arr[0]
        obs["left_wrist_view"] = img_arr[1]
        obs["right_wrist_view"] = img_arr[2]
        obs["prompt"] = "stack the blocks"
        obs["state"] = state.astype(np.float32)
        obs["dataset_names"] = "stack_blocks_three_merged"

        # 推理
        result = self.policy.infer(obs)
        inference_time = time.time() - inference_start
        
        if "predict_action" in result and result["predict_action"] is not None:
            actions = result["predict_action"][0]
            self.inference_count += 1
            
            # 反归一化
            denorm_actions = self.denormalize_action(actions)
            
            debug_print("Server", "="*60, "INFO")
            debug_print("Server", f"Inference #{self.inference_count}", "INFO")
            debug_print("Server", f"Time: {inference_time:.3f}s", "INFO")
            debug_print("Server", f"Trajectory shape: {denorm_actions.shape}", "INFO")
            
            # 打印反归一化后的第一个动作
            first_action = denorm_actions[0]
            debug_print("Server", "-"*60, "INFO")
            debug_print("Server", "First action (DENORMALIZED):", "INFO")
            debug_print("Server", f"  Left pos (m):  {first_action[0:3]}", "INFO")
            debug_print("Server", f"  Left rot:     {first_action[3:6]}", "INFO")
            debug_print("Server", f"  Left gripper: {first_action[6:7]}", "INFO")
            debug_print("Server", f"  Right pos (m): {first_action[7:10]}", "INFO")
            debug_print("Server", f"  Right rot:    {first_action[10:13]}", "INFO")
            debug_print("Server", f"  Right gripper:{first_action[13:14]}", "INFO")
            debug_print("Server", "-"*60, "INFO")
            debug_print("Server", "="*60, "INFO")
            
            return {
                "trajectory": denorm_actions,
                "action": denorm_actions[0]
            }
        else:
            debug_print("Server", "Warning: predict_action is None", "ERROR")
            return {
                "trajectory": np.zeros((10, 14)),
                "action": np.zeros(14)
            }

    def close(self):
        if hasattr(self, 'bisocket'):
            self.bisocket.close()

import torch

def main():
    SERVER_IP = "0.0.0.0"
    SERVER_PORT = 10000
    CHECKPOINT_PATH = "/data/vla/wall-x/checkpoints/stack_blocks_joint/4"
    CONFIG_PATH = "/data/vla/wall-x/checkpoints/stack_blocks_joint/4/config.yml"
    ACTION_TOKENIZER_PATH = None  # flow模式不需要action tokenizer

    server = WallXFlowServer(
        checkpoint_path=CHECKPOINT_PATH,
        config_path=CONFIG_PATH,
        action_tokenizer_path=ACTION_TOKENIZER_PATH,
        device="cuda:0"
    )

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((SERVER_IP, SERVER_PORT))
    server_socket.listen(1)

    debug_print("Server", f"Listening on {SERVER_IP}:{SERVER_PORT}", "INFO")
    debug_print("Server", "Waiting for client connection...", "INFO")
    debug_print("Server", "="*60, "INFO")

    try:
        while True:
            conn, addr = server_socket.accept()
            debug_print("Server", f"Connected by {addr}", "INFO")
            debug_print("Server", "="*60, "INFO")

            bisocket = BiSocket(conn, server.infer, send_back=True)
            server.bisocket = bisocket

            while bisocket.running.is_set():
                time.sleep(0.5)

            debug_print("Server", "Client disconnected. Waiting for next client...", "WARNING")
            server.close()

    except KeyboardInterrupt:
        debug_print("Server", "Shutting down...", "WARNING")
    finally:
        server_socket.close()

if __name__ == "__main__":
    main()
