"""
Runs trained policy on MetaWorld tasks, records RGB frames, saves videos.

Usage:
    python evaluate.py \
        --checkpoint ./logs/e2e_vla/checkpoints/best.ckpt \
        --tasks pick-place basketball \
        --episodes_per_task 10 \
        --output_dir ./eval_results \
        --device cuda:0
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import imageio

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "policy"))
sys.path.insert(0, os.path.join(REPO_ROOT, "policy", "diffusion_policy_3d"))

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from vlm.vlm import VisualTaskPlanner
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.env.metaworld.metaworld_wrapper import MetaWorldEnv
from helpers import ConfigWrapper, prepare_meta

TASK_INSTRUCTIONS = {
    "pick-place": "Pick up the cube and place it at the target location.",
    "basketball": "Pick up the ball and place it into the basketball hoop.",
    # "push":       "Push the object to the target location.",
    # "reach":      "Move the end effector to the target position.",
    # "drawer-close": "Close the drawer by pushing it shut.",
}

VLM_HIDDEN_SIZE = 2560


def build_model(device, load_vlm=True):
    """Build the EndToEndRobotPolicy with live VLM."""

    shape_meta = {
        "obs": {
            "point_cloud": {"shape": [1024, 6], "type": "point_cloud"},
            "agent_pos":   {"shape": [9], "type": "low_dim"},
            "rgb_image":   {"shape": [128, 128, 3], "type": "rgb"},
        },
        "action": {"shape": [4]},
    }

    planner = VisualTaskPlanner(
        load_vlm=load_vlm,
        vlm_dim=VLM_HIDDEN_SIZE,
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        freeze_vlm=True,
        latent_dim=512,
        num_pooling_queries=16,
        contrastive_weight=1.0,
        contrastive_temperature=0.1,
        num_vlm_layers_to_use=2,
        layer_fusion_method="learned_weighted",
        use_multi_layer=True,
    )

    # Policy (System 1) 
    shape_meta_with_latent = shape_meta.copy()
    shape_meta_with_latent["obs"]["task_latent"] = {
        "shape": [512],
        "type": "low_dim",
    }
    shape_meta_with_latent = prepare_meta(shape_meta_with_latent)

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="sample",
    )

    policy = DP3(
        shape_meta=shape_meta_with_latent,
        noise_scheduler=noise_scheduler,
        use_point_crop=True,
        condition_type="film",
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        diffusion_step_embed_dim=128,
        down_dims=[512, 1024, 2048],
        crop_shape=[80, 80],
        encoder_output_dim=64,
        horizon=16,
        kernel_size=5,
        n_action_steps=8,
        n_obs_steps=2,
        n_groups=8,
        num_inference_steps=10,
        obs_as_global_cond=True,
        use_pc_color=True,
        pointnet_type="pointnet",
        pointcloud_encoder_cfg=ConfigWrapper({
            "in_channels": 6,
            "out_channels": 64,
            "use_layernorm": True,
            "final_norm": "layernorm",
            "normal_channel": False,
        }),
    )

    return planner, policy


def load_checkpoint(planner, policy, checkpoint_path, device):
    """Load weights from a training checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Load task encoder weights
    planner.task_encoder.load_state_dict(ckpt["task_encoder_state_dict"])
    print(f"Loaded task encoder (epoch {ckpt['epoch']}, step {ckpt['global_step']})")

    # Load policy weights (includes normalizer)
    policy.load_state_dict(ckpt["policy_state_dict"])
    print(f"Loaded policy")

    planner.to(device)
    planner.eval()
    policy.to(device)
    policy.eval()

    return ckpt["epoch"], ckpt["global_step"]


@torch.no_grad()
def compute_latent(planner, rgb_image, instruction, device):
    """
    Compute task latent from a single RGB image + instruction via live VLM.

    Args:
        rgb_image: (H, W, 3) uint8 numpy array
        instruction: str

    Returns:
        latent: (512,) float32 tensor on device
    """
    plan_output = planner.plan(
        image=rgb_image,
        instruction=instruction,
        training=False,
        return_encoder_loss=False,
    )
    return plan_output["latent"].to(device=device, dtype=torch.float32)


@torch.no_grad()
def run_episode(planner, policy, env, instruction, device,
                n_obs_steps=2, max_steps=200,
                latent_update_interval=3):
    """
    Run a single evaluation episode.

    Returns:
        frames: list of (H, W, 3) uint8 RGB images for video
        total_reward: float
        success: bool
        info: dict with final env info
    """
    obs = env.reset()
    frames = []
    total_reward = 0.0
    success = False

    # Observation buffer (last n_obs_steps observations) 
    obs_buffer = deque(maxlen=n_obs_steps)

    # First observation: duplicate to fill the buffer
    processed = _process_obs(obs)
    for _ in range(n_obs_steps):
        obs_buffer.append(processed)

    # Compute initial latent from first frame 
    rgb_for_vlm = obs["image"].transpose(1, 2, 0) if obs["image"].shape[0] == 3 else obs["image"]
    rgb_for_vlm = rgb_for_vlm.astype(np.uint8)
    latent = compute_latent(planner, rgb_for_vlm, instruction, device)

    step_count = 0
    action_queue = []

    while step_count < max_steps:
        # Record frame 
        frame = env.get_rgb()
        if frame.shape[0] == 3:
            frame = frame.transpose(1, 2, 0)
        frames.append(frame.astype(np.uint8))

        # Update latent periodically 
        if step_count > 0 and step_count % latent_update_interval == 0:
            rgb_for_vlm = obs["image"].transpose(1, 2, 0) if obs["image"].shape[0] == 3 else obs["image"]
            rgb_for_vlm = rgb_for_vlm.astype(np.uint8)
            latent = compute_latent(planner, rgb_for_vlm, instruction, device)

        # Predict new action chunk if queue is empty 
        if len(action_queue) == 0:
            obs_dict = _build_policy_input(obs_buffer, latent, device)
            result = policy.predict_action(obs_dict)
            actions = result["action"][0].cpu().numpy()  # (n_action_steps, action_dim)
            action_queue = list(actions)

        # Execute one action 
        action = action_queue.pop(0)
        obs, reward, done, env_info = env.step(action)
        total_reward += reward

        # Track success
        if env_info.get("success", False):
            success = True

        # Update observation buffer
        processed = _process_obs(obs)
        obs_buffer.append(processed)

        step_count += 1
        if done:
            break

    # Capture final frame
    frame = env.get_rgb()
    if frame.shape[0] == 3:
        frame = frame.transpose(1, 2, 0)
    frames.append(frame.astype(np.uint8))

    return frames, total_reward, success, env_info


def _process_obs(obs):
    """Extract and format observation arrays for the policy buffer."""
    return {
        "point_cloud": obs["point_cloud"].astype(np.float32),
        "agent_pos":   obs["agent_pos"].astype(np.float32),
    }


def _build_policy_input(obs_buffer, latent, device):
    """
    Stack the observation buffer into batched tensors for the policy.

    The policy expects:
        point_cloud: (B, T, 1024, 6)
        agent_pos:   (B, T, 9)
        task_latent: (B, T, 512)

    where B=1, T=n_obs_steps.
    """
    obs_list = list(obs_buffer)
    T = len(obs_list)

    point_clouds = np.stack([o["point_cloud"] for o in obs_list], axis=0)  # (T, 1024, 6)
    agent_pos = np.stack([o["agent_pos"] for o in obs_list], axis=0)       # (T, 9)

    obs_dict = {
        "point_cloud": torch.from_numpy(point_clouds).unsqueeze(0).to(device),    # (1, T, 1024, 6)
        "agent_pos":   torch.from_numpy(agent_pos).unsqueeze(0).to(device),       # (1, T, 9)
        "task_latent": latent.unsqueeze(0).unsqueeze(0).expand(1, T, -1).to(device),  # (1, T, 512)
    }
    return obs_dict


def save_video(frames, path, fps=10):
    """Save list of RGB frames as mp4 video."""
    path = str(path)
    if not path.endswith(".mp4"):
        path += ".mp4"
    imageio.mimsave(path, frames, fps=fps)
    print(f"  Saved video: {path} ({len(frames)} frames)")


def evaluate(args):
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model 
    print("Building model...")
    planner, policy = build_model(device, load_vlm=True)

    # Load checkpoint 
    epoch, step = load_checkpoint(planner, policy, args.checkpoint, device)

    # Results tracking 
    all_results = {}

    for task_name in args.tasks:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {task_name}")
        print(f"{'='*60}")

        instruction = TASK_INSTRUCTIONS.get(
            task_name,
            f"Complete the {task_name.replace('-', ' ')} task."
        )
        print(f"  Instruction: {instruction}")

        # Create environment 
        env = MetaWorldEnv(
            task_name=task_name,
            device=args.device,
            use_point_crop=True,
            num_points=1024,
        )

        task_results = []
        task_dir = output_dir / task_name
        task_dir.mkdir(exist_ok=True)

        for ep_idx in range(args.episodes_per_task):
            print(f"\n  Episode {ep_idx + 1}/{args.episodes_per_task}")

            frames, reward, success, info = run_episode(
                planner=planner,
                policy=policy,
                env=env,
                instruction=instruction,
                device=device,
                n_obs_steps=2,
                max_steps=200,
                latent_update_interval=3,
            )

            result = {
                "episode": ep_idx,
                "reward": float(reward),
                "success": bool(success),
                "steps": len(frames) - 1,
            }
            task_results.append(result)

            status = "SUCCESS" if success else "FAIL"
            print(f"{status} | reward={reward:.2f} | steps={result['steps']}")

            # Save video
            video_path = task_dir / f"ep{ep_idx:03d}_{status.lower()}.mp4"
            save_video(frames, video_path, fps=10)

        # Task summary 
        n_success = sum(r["success"] for r in task_results)
        mean_reward = np.mean([r["reward"] for r in task_results])
        success_rate = n_success / len(task_results)

        print(f"\n {task_name} Summary:")
        print(f"Success rate: {n_success}/{len(task_results)} ({success_rate:.0%})")
        print(f"Mean reward:  {mean_reward:.2f}")

        all_results[task_name] = {
            "success_rate": success_rate,
            "mean_reward": float(mean_reward),
            "episodes": task_results,
        }

    # Save results JSON 
    results_path = output_dir / "results.json"
    meta = {
        "checkpoint": str(args.checkpoint),
        "epoch": epoch,
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "tasks": all_results,
    }
    with open(results_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print(f"\n{'='*60}")
    print("  OVERALL SUMMARY")
    print(f"{'='*60}")
    for task_name, res in all_results.items():
        print(f"  {task_name:20s}  success={res['success_rate']:.0%}  reward={res['mean_reward']:.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DuPLO-VLA on MetaWorld")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best.ckpt or latest.ckpt")
    parser.add_argument("--tasks", nargs="+", default=["pick-place", "basketball"],
                        help="MetaWorld task names to evaluate")
    parser.add_argument("--episodes_per_task", type=int, default=10,
                        help="Number of evaluation episodes per task")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory for videos and results JSON")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device")
    args = parser.parse_args()

    evaluate(args)