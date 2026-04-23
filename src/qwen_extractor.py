"""
Offline VLM feature extraction for LIBERO with dual-rate storage.

The VLM runs at a lower frequency ("slow stream") than the action controller
("fast stream"). This matches System-1/System-2 robotics architectures where
a slow planner produces latents that a fast action policy re-uses across
multiple action steps.

Output layout (one group per demo):

    libero_<suite>_features.h5
    ├── demo_0/
    │   ├── # ---- Fast stream (action-rate, T frames) ----
    │   ├── agentview_image       (T, H, W, 3)   uint8
    │   ├── eye_in_hand_image     (T, H, W, 3)   uint8
    │   ├── state                 (T, proprio)   float32
    │   ├── action                (T, action)    float32
    │   ├── vlm_frame_idx         (T,)           int32
    │   │       for each action frame, index into the slow stream
    │   │       (always points to the most recent past VLM frame)
    │   ├── # ---- Slow stream (VLM-rate, T_slow frames) ----
    │   ├── vlm_source_idx        (T_slow,)      int32
    │   │       which action-frame each VLM sample was computed from
    │   ├── agentview_hidden      (T_slow, num_layers, L_text, D) uint16 (bf16 view)
    │   ├── wrist_hidden          (T_slow, num_layers, L_text, D) uint16 (bf16 view)
    │   ├── text_mask             (T_slow, L_text)  bool
    │   └── attrs: { instruction, episode_id, num_frames, vlm_stride, ... }
    ├── demo_1/
    │   ...
    └── attrs: {
          suite, model_name, vlm_hidden_dim, num_layers,
          layer_indices, extracted_cameras, vlm_stride,
        }

Design notes:
    * VLM runs every `stride` frames. At stride=4 with 20 Hz control that's 5 Hz.
    * `vlm_frame_idx[t]` gives the dataloader the slow-stream index to use
        at action frame t. Always points to the most recent *past* VLM frame
        (no future leakage) — this matches how inference will work.
    * Frame 0 always gets a VLM forward so `vlm_frame_idx[0]` is valid.
    * Both cameras extracted via separate VLM passes (one fused representation
        per camera). Later you can use either or both at training time by loading
        the right key.
    * Hidden states stored as bfloat16 via uint16 bit-view. Same byte count as
        fp16 but preserves bf16's wider dynamic range.
    * Resume logic: skips demos already present with all expected datasets.
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from vlm.vlm import VisualTaskPlanner


# --------------------------------------------------------------------------- #
# bfloat16 <-> uint16 view helpers (h5py has no native bf16 support)
# --------------------------------------------------------------------------- #
def bf16_to_uint16(t: torch.Tensor) -> np.ndarray:
    """Bit-cast a bfloat16 tensor to uint16 so h5py can store it."""
    assert t.dtype == torch.bfloat16, f"expected bfloat16, got {t.dtype}"
    return t.contiguous().view(torch.uint16).cpu().numpy()


# Corresponding reader (include this pattern in your dataloader):
#
#   arr = f["demo_0/agentview_hidden"][vlm_idx]       # uint16
#   t   = torch.from_numpy(arr).view(torch.bfloat16)  # back to bf16


# --------------------------------------------------------------------------- #
# LIBERO instruction lookup — more robust than a single attrs.get() call
# --------------------------------------------------------------------------- #
def get_instruction(demo_f: h5py.File, demo_id: str, file_name: str) -> str:
    """
    LIBERO stores the language instruction in a few possible places depending
    on the suite and version. Try them in order; fall back to filename.
    """
    attrs = demo_f["data"].attrs
    if "problem_info" in attrs:
        try:
            info = json.loads(attrs["problem_info"])
            if "language_instruction" in info:
                return info["language_instruction"]
        except (json.JSONDecodeError, TypeError):
            pass

    for key in ("language_instruction", "language", "instruction"):
        if key in attrs:
            val = attrs[key]
            return val.decode() if isinstance(val, bytes) else str(val)

    return file_name.replace("_demo.hdf5", "").replace("_", " ")


# --------------------------------------------------------------------------- #
# Slow/fast stream planning
# --------------------------------------------------------------------------- #
def plan_slow_indices(num_frames: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a demo with num_frames action-rate frames and a VLM stride,
    return:
        vlm_source_idx:  (T_slow,)  — which action frames the VLM runs on.
                         Always includes frame 0; then 0 + stride, 0 + 2*stride, ...
        vlm_frame_idx:   (num_frames,) — for each action frame t, the index
                         into the slow stream whose source is the most recent
                         past action frame.

    Example (num_frames=10, stride=4):
        vlm_source_idx = [0, 4, 8]
        vlm_frame_idx  = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
            action frame 0 uses slow-stream 0 (computed on frame 0)
            action frame 3 uses slow-stream 0 (still the most recent past VLM)
            action frame 4 uses slow-stream 1 (computed on frame 4)
            action frame 9 uses slow-stream 2 (computed on frame 8)
    """
    vlm_source_idx = np.arange(0, num_frames, stride, dtype=np.int32)
    # For each action frame t, find the largest slow-stream index whose
    # source frame is <= t. Equivalent to floor(t / stride).
    vlm_frame_idx = (np.arange(num_frames) // stride).astype(np.int32)
    return vlm_source_idx, vlm_frame_idx


# --------------------------------------------------------------------------- #
# Per-camera VLM feature extraction for one demo's slow-stream frames
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_slow_stream(
    planner: VisualTaskPlanner,
    images: np.ndarray,           # (T_slow, H, W, 3) uint8 — already subsampled
    instruction: str,
    batch_size: int,
    layer_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the VLM on the slow-stream frames, sampling the requested layers.
    Returns:
        sampled:   (T_slow, num_layers, L_text, D) bfloat16
        mask:      (T_slow, L_text) bool
    """
    T_slow = len(images)
    per_batch_hidden = []
    per_batch_mask = []

    for i in range(0, T_slow, batch_size):
        end = min(i + batch_size, T_slow)
        img_batch = [images[j] for j in range(i, end)]
        text_batch = [instruction] * (end - i)

        all_h, mask = planner.extract_features_batch(img_batch, text_batch)
        # all_h is a list of length (num_vlm_layers + 1). Pick the ones we want.
        sampled = torch.stack([all_h[idx] for idx in layer_indices], dim=1)
        # sampled: (B, num_layers, L_text, D) in bf16 already

        per_batch_hidden.append(sampled.to(torch.bfloat16))
        per_batch_mask.append(mask.bool())

    hidden = torch.cat(per_batch_hidden, dim=0)   # (T_slow, num_layers, L_text, D)
    mask = torch.cat(per_batch_mask, dim=0)       # (T_slow, L_text)
    return hidden, mask


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="./libero_goal")
    parser.add_argument("--out_path", default="libero_goal_features.h5")
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--vlm_stride", type=int, default=4,
                        help="Run VLM every N action frames. LIBERO is 20 Hz; "
                             "stride=4 -> 5 Hz VLM rate.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Slow-stream frames per VLM forward.")
    parser.add_argument("--layer_indices", type=int, nargs="+",
                        default=[8, 16, 24, 32])
    parser.add_argument("--cameras", nargs="+",
                        default=["agentview_image", "eye_in_hand_image"])
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    demo_files = sorted(f for f in os.listdir(dataset_path) if f.endswith(".hdf5"))
    print(f"Found {len(demo_files)} demo files in {dataset_path}")
    print(f"VLM stride: {args.vlm_stride} "
          f"(~{20 / args.vlm_stride:.1f} Hz at LIBERO's 20 Hz control)")

    # Build planner once
    planner = VisualTaskPlanner(load_vlm=True, freeze_vlm=True)
    planner.eval()
    vlm_dim = planner.vlm.config.hidden_size

    # Open output
    out_mode = "a" if Path(args.out_path).exists() else "w"
    out_f = h5py.File(args.out_path, out_mode)

    # Root-level attrs
    out_f.attrs["suite"] = args.suite
    out_f.attrs["model_name"] = "Qwen/Qwen3-VL-4B-Instruct"
    out_f.attrs["vlm_hidden_dim"] = vlm_dim
    out_f.attrs["num_layers"] = len(args.layer_indices)
    out_f.attrs["layer_indices"] = np.array(args.layer_indices)
    out_f.attrs["extracted_cameras"] = [c.encode() for c in args.cameras]
    out_f.attrs["vlm_stride"] = args.vlm_stride

    # Flat work list across all demo files
    work = []
    for file_name in demo_files:
        file_path = dataset_path / file_name
        with h5py.File(file_path, "r") as demo_f:
            for demo_id in demo_f["data"].keys():
                work.append((file_name, demo_id))

    print(f"Total demos: {len(work)}")
    global_ep_id = 0
    total_slow = 0
    total_fast = 0

    for file_name, demo_id in tqdm(work, desc="Demos"):
        group_name = f"demo_{global_ep_id}"

        # Resume: skip if already extracted with all expected datasets
        expected_keys = {"state", "action", "text_mask",
                         "vlm_frame_idx", "vlm_source_idx"}
        expected_keys.update(c for c in args.cameras)
        expected_keys.update(
            f"{c.replace('_image', '')}_hidden" for c in args.cameras
        )
        if group_name in out_f:
            if expected_keys.issubset(set(out_f[group_name].keys())):
                global_ep_id += 1
                continue
            else:
                del out_f[group_name]   # partial write — redo

        file_path = dataset_path / file_name
        with h5py.File(file_path, "r") as demo_f:
            obs_grp = demo_f[f"data/{demo_id}/obs"]
            instruction = get_instruction(demo_f, demo_id, file_name)

            # Fast-stream data (everything at action rate)
            states  = obs_grp["robot0_proprio-state"][:].astype(np.float32)
            actions = demo_f[f"data/{demo_id}/actions"][:].astype(np.float32)
            T = len(states)

            # Slow/fast index mapping
            vlm_source_idx, vlm_frame_idx = plan_slow_indices(T, args.vlm_stride)
            T_slow = len(vlm_source_idx)
            total_slow += T_slow
            total_fast += T

            camera_arrays = {}        # cam_key -> uint8 (T, H, W, 3)
            camera_hidden = {}        # short_key -> bf16 (T_slow, num_layers, L, D)
            camera_mask = None        # shared across cameras (text is the same)

            for cam in args.cameras:
                if cam not in obs_grp:
                    raise KeyError(
                        f"{cam!r} not in {file_name}/{demo_id}. "
                        f"Available: {list(obs_grp.keys())}"
                    )

                # Full-rate images for the fast stream (stored for later use)
                imgs_fast = obs_grp[cam][:]          # (T, H, W, 3)
                camera_arrays[cam] = imgs_fast

                # Subsampled images for the slow-stream VLM pass
                imgs_slow = imgs_fast[vlm_source_idx]   # (T_slow, H, W, 3)

                hidden, mask = extract_slow_stream(
                    planner, imgs_slow, instruction,
                    batch_size=args.batch_size,
                    layer_indices=args.layer_indices,
                )
                short_key = cam.replace("_image", "")   # agentview / eye_in_hand
                camera_hidden[short_key] = hidden
                if camera_mask is None:
                    camera_mask = mask

        # Write this demo's group
        grp = out_f.create_group(group_name)
        grp.attrs["instruction"] = instruction
        grp.attrs["episode_id"] = global_ep_id
        grp.attrs["num_frames"] = T
        grp.attrs["num_slow_frames"] = T_slow
        grp.attrs["vlm_stride"] = args.vlm_stride
        grp.attrs["source_file"] = file_name
        grp.attrs["source_demo"] = demo_id

        # Fast-stream datasets
        for cam, imgs in camera_arrays.items():
            grp.create_dataset(
                cam, data=imgs,
                chunks=(1,) + imgs.shape[1:],
                compression="gzip", compression_opts=4,
            )
        grp.create_dataset("state", data=states)
        grp.create_dataset("action", data=actions)
        grp.create_dataset("vlm_frame_idx", data=vlm_frame_idx)

        # Slow-stream datasets
        grp.create_dataset("vlm_source_idx", data=vlm_source_idx)
        for short_key, hidden in camera_hidden.items():
            uint16_view = bf16_to_uint16(hidden)
            grp.create_dataset(
                f"{short_key}_hidden",
                data=uint16_view,
                chunks=(1,) + uint16_view.shape[1:],
                compression="gzip", compression_opts=4,
            )
        grp.create_dataset(
            "text_mask",
            data=camera_mask.cpu().numpy(),
            chunks=(1,) + tuple(camera_mask.shape[1:]),
        )

        # Periodic flush so a crash doesn't lose everything
        if global_ep_id % 10 == 0:
            out_f.flush()

        global_ep_id += 1

    out_f.close()
    print(f"\nDone. Wrote {global_ep_id} demos to {args.out_path}")
    print(f"Action-rate frames:   {total_fast:,}")
    print(f"VLM-rate frames:      {total_slow:,} "
          f"({100 * total_slow / max(total_fast, 1):.1f}% of action rate)")


if __name__ == "__main__":
    main()
