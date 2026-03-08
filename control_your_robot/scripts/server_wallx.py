#!/usr/bin/env python3
"""
Wall-X 模型推理服务器 - 批量推理优化版
推理一次预测10步动作，然后分批发送给客户端
"""
import sys
import os
sys.path.append('./')
os.environ["INFO_LEVEL"] = "INFO"

import socket
import time
import numpy as np
import cv2
import torch
import pathlib
import copy
from collections import deque

# 添加路径
sys.path.append('/data/vla/wall-x')

from utils.bisocket import BiSocket
from utils.data_handler import debug_print

# 导入 Wall-X 模型
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.data.utils import preprocesser_call
from wall_x.model.action_head import Normalizer

class WallXServer:
    def __init__(self, checkpoint_path, device="cuda:0", control_freq=10, instruction="stack the blocks"):
        self.control_freq = control_freq
        self.device_str = device
        self.instruction = instruction
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.action_horizon = 10
        self.action_dim = 14
        self.action_buffer = deque(maxlen=self.action_horizon)  # 缓存预测的动作序列

        debug_print("Server", f"Loading model from {checkpoint_path}", "INFO")
        debug_print("Server", f"Action horizon: {self.action_horizon}, Action dim: {self.action_dim}", "INFO")
        debug_print("Server", f"Device: {device}", "INFO")
        debug_print("Server", f"Mode: Batch inference with action replay", "INFO")

        self.model = Qwen2_5_VLMoEForAction.from_pretrained(
            checkpoint_path,
            action_tokenizer_path=None,
        )

        self.model = self.model.to("cuda").to_bfloat16_for_selected_params()
        self.model.eval()

        self.processor = self.model.processor

        # 加载 normalizers
        action_norm_path = self.checkpoint_path / "normalizer_action.pth"
        state_norm_path = self.checkpoint_path / "normalizer_propri.pth"

        debug_print("Server", f"Loading action normalizer from {action_norm_path}", "INFO")
        self.action_normalizer = Normalizer.from_ckpt(str(action_norm_path))

        debug_print("Server", f"Loading state normalizer from {state_norm_path}", "INFO")
        self.state_normalizer = Normalizer.from_ckpt(str(state_norm_path))

        # 设置 normalizers 到模型
        action_norm_cuda = copy.deepcopy(self.action_normalizer).to("cuda")
        state_norm_cuda = copy.deepcopy(self.state_normalizer).to("cuda")
        self.model.set_normalizer(action_norm_cuda, state_norm_cuda)
        debug_print("Server", "Normalizers set on model", "INFO")

        self.observation_window = None
        self.last_inference_time = 0
        self.inference_count = 0

    def update_observation_window(self, imgs, state):
        """更新观测数据"""
        self.observation_window = {
            "images": {
                "face_view": imgs[0],
                "left_wrist_view": imgs[1],
                "right_wrist_view": imgs[2],
            },
            "state": state
        }

    def predict_actions(self):
        """执行推理，预测未来10步动作"""
        if self.observation_window is None:
            return [np.zeros(self.action_dim)] * self.action_horizon

        with torch.no_grad():
            inference_start = time.time()

            # 准备输入
            prep_start = time.time()
            images = self.observation_window["images"]
            state = torch.from_numpy(self.observation_window["state"]).float().unsqueeze(0)

            text = f"<|propri|>Observation: face_view: <|vision_start|><|image_pad|><|vision_end|> left_wrist_view: <|vision_start|><|image_pad|><|vision_end|> right_wrist_view: <|vision_start|><|image_pad|><|vision_end|> Instruction:{self.instruction} {'<|action|>' * self.action_horizon}"

            image_list = [[images["face_view"], images["left_wrist_view"], images["right_wrist_view"]]]
            inputs = preprocesser_call(
                processor=self.processor,
                images=image_list,
                text=[text],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            pixel_values = inputs["pixel_values"].to(self.device_str)
            image_grid_thw = inputs["image_grid_thw"].to(self.device_str)
            input_ids = inputs["input_ids"].to(self.device_str)
            attention_mask = inputs["attention_mask"].to(self.device_str)

            action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")
            moe_token_types = input_ids == action_token_id

            dof_mask = torch.ones(1, self.action_horizon, self.action_dim, dtype=torch.float32)

            state = state.to(self.device_str)
            prep_time = time.time() - prep_start

            # 模型推理
            model_start = time.time()
            outputs = self.model(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                input_ids=input_ids,
                attention_mask=attention_mask,
                moe_token_types=moe_token_types,
                dof_mask=dof_mask,
                agent_pos=state,
                action_dim=self.action_dim,
                action_horizon=self.action_horizon,
                mode="predict",
                predict_mode="diffusion",
                dataset_names=["stack_blocks_test_v1"],
            )
            model_time = time.time() - model_start
            total_time = time.time() - inference_start

            # 获取预测的动作序列
            if outputs.get("predict_action") is not None:
                pred_actions = outputs["predict_action"][0].cpu().numpy()  # [action_horizon, action_dim]
                self.inference_count += 1
                debug_print("Server", f"Inference #{self.inference_count}: Predicted {len(pred_actions)} actions | Prep: {prep_time:.3f}s | Model: {model_time:.3f}s | Total: {total_time:.3f}s", "INFO")
                return list(pred_actions)
            else:
                debug_print("Server", "Warning: predict_action is None", "ERROR")
                return [np.zeros(self.action_dim)] * self.action_horizon

    def infer(self, message):
        """推理接口"""
        # 解码图像
        img_arr = []
        for data in message["img_arr"]:
            jpeg_bytes = np.array(data).tobytes().rstrip(b'\0')
            nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(nparr, 1)
            img_arr.append(img)

        state = message["state"]

        # 更新观测
        self.update_observation_window(img_arr, state)

        # 检查是否需要重新推理（缓冲区空了或超过2秒）
        current_time = time.time()
        need_new_inference = (
            len(self.action_buffer) == 0 or 
            (current_time - self.last_inference_time) > 6.0
        )

        if need_new_inference:
            # 执行推理，预测未来10步
            actions = self.predict_actions()
            self.action_buffer.clear()
            self.action_buffer.extend(actions)
            self.last_inference_time = current_time

        # 从缓冲区取出下一个动作
        if len(self.action_buffer) > 0:
            action = self.action_buffer.popleft()
            return {"action": action}
        else:
            return {"action": np.zeros(self.action_dim)}

    def close(self):
        if hasattr(self, 'bisocket'):
            self.bisocket.close()

def main():
    SERVER_IP = "0.0.0.0"
    SERVER_PORT = 10000
    CHECKPOINT_PATH = "/data/vla/wall-x/checkpoints/stack_blocks_flow/35"
    INSTRUCTION = "stack the blocks"

    debug_print("Server", "="*60, "INFO")
    debug_print("Server", "Wall-X Model Server - Batch Replay Mode", "INFO")
    debug_print("Server", "="*60, "INFO")
    debug_print("Server", f"Checkpoint: {CHECKPOINT_PATH}", "INFO")
    debug_print("Server", f"Instruction: {INSTRUCTION}", "INFO")
    debug_print("Server", f"Action horizon: 10, Action dim: 14", "INFO")
    debug_print("Server", f"Device: cuda:0", "INFO")
    debug_print("Server", f"Optimization: Predict 10 actions once, replay them", "INFO")
    debug_print("Server", "="*60, "INFO")
    debug_print("Server", f"Listening on {SERVER_IP}:{SERVER_PORT}", "INFO")
    debug_print("Server", "Waiting for client connection...", "INFO")
    debug_print("Server", "="*60, "INFO")

    server = WallXServer(
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda:0",
        control_freq=10,
        instruction=INSTRUCTION
    )

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((SERVER_IP, SERVER_PORT))
    server_socket.listen(1)

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
