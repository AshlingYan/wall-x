#!/usr/bin/env python3
"""
Wall-X 推理服务器 - 仅使用状态
"""
import sys
import os
sys.path.append("/data/vla/wall-x")
sys.path.append("./")
os.environ["INFO_LEVEL"] = "INFO"

import socket
import time
import numpy as np
import yaml
import torch

from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from utils.bisocket import BiSocket
from utils.data_handler import debug_print


class WallXServer:
    def __init__(self, checkpoint_path, device="cuda:0"):
        self.device = device
        debug_print("Server", f"Loading model from {checkpoint_path}", "INFO")

        # 加载配置
        config_path = os.path.join(checkpoint_path, "config.yml")
        with open(config_path, "r") as f:
            self.train_config = yaml.safe_load(f)

        # 加载模型
        self.model = Qwen2_5_VLMoEForAction.from_pretrained(checkpoint_path)
        self.model.eval()
        self.model = self.model.to(device)
        self.model = self.model.bfloat16()

        debug_print("Server", "Model loaded successfully", "INFO")

    def get_action(self, state):
        """仅使用状态生成动作"""
        state_dim = len(state)
        batch_size = 1

        # 构造文本
        text = "<|im_start|>system\nYou are a helpful robot assistant.<|im_end|>\n"
        text += "<|im_start|>user\nStack the blocks<|im_end|>\n"
        text += "<|im_start|>assistant\n"

        processor = self.model.processor
        inputs = processor(text=[text], return_tensors="pt")

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
        moe_token_types = moe_token_types.to(device)
        position_ids = position_ids.to(device)
        proprioception = proprioception.to(device).bfloat16()
        agent_pos_mask = agent_pos_mask.to(device).bfloat16()
        dof_mask = dof_mask.to(device).bfloat16()

        # 运行模型
        try:
            with torch.no_grad():
                output = self.model(
                    input_ids=inputs_moved["input_ids"],
                    attention_mask=inputs_moved["attention_mask"],
                    moe_token_types=moe_token_types,
                    position_ids=position_ids,
                    proprioception=proprioception,
                    agent_pos_mask=agent_pos_mask,
                    dof_mask=dof_mask,
                    dataset_names=dataset_names,
                    mode="validate"
                )
                
            debug_print("Server", "Model forward successful", "DEBUG")
            
            # 返回随机动作（14维）
            action = np.random.randn(14).astype(np.float32)
            return action
            
        except Exception as e:
            debug_print("Server", f"Model forward failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return np.zeros(14)

    def infer(self, message):
        """推理接口"""
        state = message["state"]
        action = self.get_action(state)
        return {"action": action}

    def close(self):
        pass


def main():
    SERVER_IP = "0.0.0.0"
    SERVER_PORT = 10000
    CHECKPOINT_PATH = "/data/vla/wall-x/checkpoints/stack_blocks_joint/4"

    debug_print("Server", f"Starting Wall-X Server (state-only mode)", "INFO")
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
