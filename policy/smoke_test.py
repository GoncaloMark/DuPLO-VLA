"""
Usage:
    python smoke_test.py --checkpoint ./checkpoints/best.ckpt --device cuda:0
"""

import os
import sys
import argparse
import traceback
from collections import deque

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "policy"))
sys.path.insert(0, os.path.join(REPO_ROOT, "policy", "diffusion_policy_3d"))


# Colored pass/fail helpers — makes it obvious at a glance which stage died.
def _ok(msg): print(f"\033[92m[PASS]\033[0m {msg}")
def _fail(msg): print(f"\033[91m[FAIL]\033[0m {msg}")
def _info(msg): print(f"\033[94m[INFO]\033[0m {msg}")


def stage(name):
    """Decorator: prints stage header, catches exceptions with full traceback,
    returns (success, result)."""
    def deco(fn):
        def wrapper(*args, **kwargs):
            print(f"\n=== {name} ===")
            try:
                result = fn(*args, **kwargs)
                _ok(name)
                return True, result
            except Exception as e:
                _fail(f"{name}: {type(e).__name__}: {e}")
                traceback.print_exc()
                return False, None
        return wrapper
    return deco


@stage("Imports")
def test_imports():
    import torch, transformers, mujoco_py, imageio
    import pytorch3d
    from transformers import Qwen3VLForConditionalGeneration
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    _info(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    _info(f"transformers={transformers.__version__} pytorch3d={pytorch3d.__version__}")


@stage("Build model (VLM load is the slow step)")
def test_build(device):
    from inference import build_model
    planner, policy = build_model(device, load_vlm=True)
    n_vlm    = sum(p.numel() for p in planner.vlm.parameters())
    n_enc    = sum(p.numel() for p in planner.task_encoder.parameters())
    n_policy = sum(p.numel() for p in policy.parameters())
    _info(f"VLM params     : {n_vlm/1e9:.2f}B  (frozen)")
    _info(f"Task encoder   : {n_enc/1e6:.2f}M")
    _info(f"Policy (DP3)   : {n_policy/1e6:.2f}M")
    return planner, policy


@stage("Load checkpoint")
def test_checkpoint(planner, policy, checkpoint_path, device):
    from inference import load_checkpoint
    epoch, step = load_checkpoint(planner, policy, checkpoint_path, device)
    _info(f"Loaded checkpoint from epoch {epoch}, step {step}")


@stage("Create MetaWorld env + reset")
def test_env():
    from diffusion_policy_3d.env.metaworld.metaworld_wrapper import MetaWorldEnv
    env = MetaWorldEnv(task_name="pick-place", device="cuda:0",
                       use_point_crop=True, num_points=1024)
    obs = env.reset()
    # Check every key the eval path expects
    expected = {
        "point_cloud": (1024, 6),
        "agent_pos":   (9,),
        "image":       (3, 128, 128),
    }
    for key, shape in expected.items():
        assert key in obs, f"obs missing key '{key}'"
        got = obs[key].shape
        assert got == shape, f"{key}: expected {shape}, got {got}"
        _info(f"obs['{key}'].shape = {got}  OK")

    rgb = env.get_rgb()
    assert rgb.shape == (128, 128, 3) or rgb.shape == (3, 128, 128), f"bad rgb shape {rgb.shape}"
    _info(f"env.get_rgb().shape = {rgb.shape}")
    return env, obs


@stage("VLM latent computation (single-image live path)")
def test_latent(planner, obs, device):
    from inference import compute_latent
    # handles both CHW and HWC
    rgb = obs["image"].transpose(1, 2, 0) if obs["image"].shape[0] == 3 else obs["image"]
    rgb = rgb.astype(np.uint8)
    latent = compute_latent(planner, rgb,
                            "Pick up the cube and place it at the target location.",
                            device)
    assert latent.shape == (512,), f"latent shape {latent.shape} != (512,)"
    _info(f"latent.shape={tuple(latent.shape)}  dtype={latent.dtype}  "
          f"norm={latent.norm().item():.3f}  "
          f"min={latent.min().item():.3f}  max={latent.max().item():.3f}")
    # Sanity: latent shouldn't be all zeros (would mean dead encoder)
    assert latent.abs().max() > 1e-4, "latent is effectively zero — encoder likely broken"
    return latent


@stage("Policy forward pass (predict_action)")
def test_policy(policy, obs, latent, device, n_obs_steps=2):
    from inference import _process_obs, _build_policy_input
    buf = deque(maxlen=n_obs_steps)
    for _ in range(n_obs_steps):
        buf.append(_process_obs(obs))
    obs_dict = _build_policy_input(buf, latent, device)
    for k, v in obs_dict.items():
        _info(f"  input '{k}': {tuple(v.shape)} {v.dtype}")
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
    actions = result["action"][0].cpu().numpy()
    # Expected (n_action_steps=8, action_dim=4)
    assert actions.shape == (8, 4), f"action shape {actions.shape} != (8, 4)"
    assert np.isfinite(actions).all(), "action contains NaN/Inf"
    _info(f"actions.shape={actions.shape}  range=[{actions.min():.3f}, {actions.max():.3f}]")


@stage("Closed-loop: 5 env steps end-to-end")
def test_rollout(planner, policy, env, obs, device):
    from inference import run_episode
    frames, reward, success, info = run_episode(
        planner=planner, policy=policy, env=env,
        instruction="Pick up the cube and place it at the target location.",
        device=device,
        n_obs_steps=2, max_steps=5, latent_update_interval=3,
    )
    assert len(frames) >= 5, f"only got {len(frames)} frames"
    _info(f"rolled {len(frames)} frames  reward={reward:.3f}  success={success}")


@stage("GPU memory summary")
def test_memory():
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1e9
    _info(f"VRAM used: {used:.2f} / {total/1e9:.2f} GB")
    # A6000 has 48GB; full stack should fit in <15GB.
    if used > 40:
        _fail(f"VRAM usage {used:.1f}GB is alarmingly high — check for duplicate VLM copies")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)

    results = {}

    results["imports"], _ = test_imports()
    if not results["imports"]: sys.exit(1)

    results["build"], built = test_build(device)
    if not results["build"]: sys.exit(1)
    planner, policy = built

    results["checkpoint"], _ = test_checkpoint(planner, policy, args.checkpoint, device)
    if not results["checkpoint"]: sys.exit(1)

    results["env"], env_out = test_env()
    if not results["env"]: sys.exit(1)
    env, obs = env_out

    results["latent"], latent = test_latent(planner, obs, device)
    if not results["latent"]: sys.exit(1)

    results["policy"], _ = test_policy(policy, obs, latent, device)
    if not results["policy"]: sys.exit(1)

    results["rollout"], _ = test_rollout(planner, policy, env, obs, device)

    results["memory"], _ = test_memory()

    print("\n" + "=" * 50)
    print("  SMOKE TEST SUMMARY")
    print("=" * 50)
    for name, ok in results.items():
        status = "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"
        print(f"  {name:15s} {status}")
    print("=" * 50)

    if not all(results.values()):
        sys.exit(1)
    print("\nAll systems go — safe to run full evaluate.py")


if __name__ == "__main__":
    main()
