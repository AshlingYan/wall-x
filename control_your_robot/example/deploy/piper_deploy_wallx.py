#!/usr/bin/env python3
"""Wall-X deployment script for stack_blocks model (based on piper_deploy_pi05_ygx.py).
Usage:
    python piper_deploy_wallx.py --checkpoint /data/vla/wall-x/checkpoints/stack_blocks_flow/35 --task "stack the blocks"
"""
import sys
import os
import time
import argparse
import numpy as np
from pathlib import Path
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor

# Add paths FIRST before importing qwen_vl_utils
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
WALLX_ROOT = REPO_ROOT / "wall-x"
CONTROL_ROOT = REPO_ROOT / "control_your_robot"

for p in (str(WALLX_ROOT), str(CONTROL_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Now import qwen_vl_utils after paths are added
from qwen_vl_utils.vision_process import smart_resize

from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.model.action_head import Normalizer
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLConfig


class WallXPolicy:
    """Wall-X Policy wrapper for stack_blocks task."""

    def __init__(self, checkpoint_path, task="stack the blocks"):
        self.checkpoint_path = Path(checkpoint_path)
        self.task = task

        # Load config
        import yaml
        config_path = self.checkpoint_path / "config.yml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Image processing config
        self.IMAGE_FACTOR = 28
        self.MAX_PIXELS = 16384 * 28 * 28
        self.MIN_PIXELS = 4 * 28 * 28

        # Resolution for each camera
        self.resolution = {
            "face_view": 256,
            "left_wrist_view": 256,
            "right_wrist_view": 256,
        }

        # Camera name mapping
        self.cam_mapping = {
            "cam_head": "face_view",
            "cam_left_wrist": "left_wrist_view",
            "cam_right_wrist": "right_wrist_view",
        }

        # Action config (14-dim dual arm)
        self.action_dim = 14
        self.action_horizon = 10

        # Load model (processor will be loaded by from_pretrained)
        self._load_model()

        # Get processor from loaded model
        self.processor = self.model.processor

        # Load normalizers
        self._load_normalizers()

        # Observation cache
        self.observation_window = None
        self.current_instruction = self.task

    def _load_model(self):
        """Load Wall-X model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}...")

        # Use from_pretrained to handle all config automatically
        # It will load config.yml from checkpoint and handle customized_robot_config
        self.model = Qwen2_5_VLMoEForAction.from_pretrained(
            str(self.checkpoint_path),
            action_tokenizer_path=None,  # flow mode doesn't need it
        )

        # Move to GPU and set to eval mode
        self.model = self.model.to("cuda").to_bfloat16_for_selected_params()
        self.model.eval()

        print("Model loaded successfully!")

    def _load_normalizers(self):
        """Load action and state normalizers."""
        import copy

        # Load action normalizer
        action_norm_path = self.checkpoint_path / "normalizer_action.pth"
        if action_norm_path.exists():
            self.action_normalizer = Normalizer.from_ckpt(str(action_norm_path))
        else:
            self.action_normalizer = None

        # Load state normalizer
        state_norm_path = self.checkpoint_path / "normalizer_propri.pth"
        if state_norm_path.exists():
            self.state_normalizer = Normalizer.from_ckpt(str(state_norm_path))
        else:
            self.state_normalizer = None

        # Set normalizers on the model (required for flow mode)
        if self.action_normalizer is not None and self.state_normalizer is not None:
            # Move normalizers to CUDA before setting on model
            action_norm_cuda = copy.deepcopy(self.action_normalizer).to("cuda")
            state_norm_cuda = copy.deepcopy(self.state_normalizer).to("cuda")
            self.model.set_normalizer(action_norm_cuda, state_norm_cuda)

    def _preprocess_image(self, img_array):
        """Preprocess image for model input."""
        # Convert to PIL if needed
        if isinstance(img_array, np.ndarray):
            if img_array.dtype == np.uint8:
                img = Image.fromarray(img_array)
            else:
                # Assume float in [0, 1]
                img = Image.fromarray((img_array * 255).astype(np.uint8))
        else:
            img = img_array

        # Get original size
        orig_width, orig_height = img.size

        # Apply smart_resize
        resized_height, resized_width = smart_resize(
            orig_height,
            orig_width,
            factor=self.IMAGE_FACTOR,
            min_pixels=self.MIN_PIXELS,
            max_pixels=self.MAX_PIXELS,
        )

        img = img.resize((resized_width, resized_height))
        return img

    def reset_observation_windows(self):
        """Reset observation cache."""
        self.observation_window = None

    def set_language(self, instruction):
        """Set task instruction."""
        self.current_instruction = instruction

    def update_observation_window(self, img_arr, state):
        """Update observation with new images and state.

        Args:
            img_arr: tuple of 3 images (cam_head, cam_right_wrist, cam_left_wrist)
            state: 14-dim state vector [left_arm(7), right_arm(7)]
        """
        # Preprocess images
        images = {}
        for cam_name, img in zip(["cam_head", "cam_right_wrist", "cam_left_wrist"], img_arr):
            view_name = self.cam_mapping[cam_name]
            images[view_name] = self._preprocess_image(img)

        # Normalize state if normalizer available
        if self.state_normalizer is not None:
            state_tensor = torch.from_numpy(state).float()
            state_normalized = self.state_normalizer.normalize_data(
                state_tensor.unsqueeze(0),
                ["stack_blocks_test_v1"]
            ).squeeze(0)
            state = state_normalized.numpy()

        self.observation_window = {
            "images": images,
            "state": state,
        }

    def get_action(self):
        """Get action from current observation.

        Returns:
            action_chunk: numpy array of shape [action_horizon, action_dim]
        """
        if self.observation_window is None:
            # Return zero actions if no observation
            return np.zeros((self.action_horizon, self.action_dim))

        with torch.no_grad():
            # Prepare inputs
            images = self.observation_window["images"]
            state = torch.from_numpy(self.observation_window["state"]).float().unsqueeze(0)

            # Prepare text with image placeholders
            # Add <|image_pad|> for each camera view
            # For flow mode, need action_horizon number of <|action|> tokens
            text = f"<|propri|>Observation: face_view: <|vision_start|><|image_pad|><|vision_end|> left_wrist_view: <|vision_start|><|image_pad|><|vision_end|> right_wrist_view: <|vision_start|><|image_pad|><|vision_end|> Instruction:{self.current_instruction} {'<|action|>' * self.action_horizon}"

            # Use preprocesser_call for unified processing (handles image expansion correctly)
            from wall_x.data.utils import preprocesser_call
            image_list = [[images["face_view"], images["left_wrist_view"], images["right_wrist_view"]]]
            inputs = preprocesser_call(
                processor=self.processor,
                images=image_list,
                text=[text],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            pixel_values = inputs["pixel_values"].to("cuda")
            image_grid_thw = inputs["image_grid_thw"].to("cuda")
            input_ids = inputs["input_ids"].to("cuda")
            attention_mask = inputs["attention_mask"].to("cuda")

            # Calculate moe_token_types - identifies which tokens are action tokens
            action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")
            moe_token_types = input_ids == action_token_id

            # Create dof_mask - mask for valid degrees of freedom
            # All 14 dimensions are valid for dual arm setup
            dof_mask = torch.ones(1, self.action_horizon, self.action_dim, dtype=torch.float32)

            state = state.to("cuda")

            # Predict
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
                predict_mode="diffusion",  # flow mode
                dataset_names=["stack_blocks_test_v1"],  # Dataset name for normalization
            )

            # Get predicted actions (already unnormalized by the model)
            if outputs.get("predict_action") is not None:
                pred_actions = outputs["predict_action"][0].cpu().numpy()  # [action_horizon, 14]
            else:
                print("Warning: predict_action is None, returning zeros")
                pred_actions = np.zeros((self.action_horizon, self.action_dim))

            return pred_actions


def input_transform(data):
    """Transform robot data to model input format.

    Args:
        data: tuple (state_dict, image_dict)
    """
    state_dict, image_dict = data

    # Extract state: left_arm(joint+gripper) + right_arm(joint+gripper)
    state = np.concatenate([
        np.array(state_dict["left_arm"]["joint"]).reshape(-1),
        np.array(state_dict["left_arm"]["gripper"]).reshape(-1),
        np.array(state_dict["right_arm"]["joint"]).reshape(-1),
        np.array(state_dict["right_arm"]["gripper"]).reshape(-1)
    ])

    # Extract images
    img_arr = (
        image_dict["cam_head"]["color"],
        image_dict["cam_right_wrist"]["color"],
        image_dict["cam_left_wrist"]["color"],
    )

    return img_arr, state


def output_transform(action):
    """Transform model action to robot command format.

    Args:
        action: 14-dim vector [left_arm(7), right_arm(7)]
    """
    move_data = {
        "arm": {
            "left_arm": {
                "joint": action[:6],
                "gripper": action[6],
            },
            "right_arm": {
                "joint": action[7:13],
                "gripper": action[13],
            }
        }
    }
    return move_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="/data/vla/wall-x/checkpoints/stack_blocks_flow/35",
                        help="Path to Wall-X checkpoint")
    parser.add_argument("--task", type=str, default="stack the blocks",
                        help="Task instruction")
    parser.add_argument("--episode-num", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--max-step", type=int, default=1000,
                        help="Max steps per episode")
    parser.add_argument("--video", action="store_true",
                        help="Record video")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without robot connection")
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip ENTER key wait (for automated testing)")

    args = parser.parse_args()

    from utils.data_handler import debug_print, is_enter_pressed

    # Initialize policy
    print("=" * 50)
    print("Wall-X Deployment for stack_blocks")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Task: {args.task}")
    print("=" * 50)

    policy = WallXPolicy(args.checkpoint, args.task)

    if not args.dry_run:
        # TODO: Initialize robot based on your setup
        # This section needs to be adapted to your robot
        pass

    for episode in range(args.episode_num):
        print(f"\n=== Episode {episode + 1}/{args.episode_num} ===")

        # Reset
        policy.reset_observation_windows()
        policy.set_language(args.task)

        # TODO: Reset robot
        # if not args.dry_run:
        #     robot.reset()

        # Wait for start signal (skip if --no-wait is set)
        if not args.no_wait:
            print("Press ENTER to start...")
            while not is_enter_pressed():
                time.sleep(0.1)
        else:
            print("Starting immediately (--no-wait enabled)...")

        print("Running... Press ENTER to stop")
        step = 0

        try:
            while step < args.max_step:
                if args.dry_run:
                    # Use dummy data for testing
                    img_arr = (
                        np.zeros((224, 224, 3), dtype=np.uint8),
                        np.zeros((224, 224, 3), dtype=np.uint8),
                        np.zeros((224, 224, 3), dtype=np.uint8),
                    )
                    state = np.zeros(14)
                else:
                    # Get observation from robot
                    # data = robot.get()
                    # img_arr, state = input_transform(data)
                    pass

                # Update policy
                policy.update_observation_window(img_arr, state)

                # Get action chunk
                action_chunk = policy.get_action()

                # Execute each action in chunk
                for action in action_chunk:
                    if not args.dry_run:
                        # Send to robot
                        move_data = output_transform(action)
                        # robot.move(move_data)
                        pass

                    if step % 10 == 0:
                        print(f"Step: {step}/{args.max_step}")

                    step += 1
                    time.sleep(1 / 20)  # 20 Hz

                    if not args.no_wait and is_enter_pressed():
                        print("Stopped by user")
                        break

                if not args.no_wait and is_enter_pressed():
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        print(f"Episode {episode + 1} finished. Steps: {step}")


if __name__ == "__main__":
    main()