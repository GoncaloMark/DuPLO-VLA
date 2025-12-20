import zarr
import numpy as np
import os
import shutil
from termcolor import cprint

INPUT_ZARRS = [
    "../data/metaworld_pick-place_expert_augmented.zarr",
    "../data/metaworld_door-open_expert_augmented.zarr",
    "../data/metaworld_drawer-open_expert_augmented.zarr",
    "../data/metaworld_button-press_expert_augmented.zarr",
    "../data/metaworld_reach_expert_augmented.zarr",
]

OUTPUT_ZARR = "../data/metaworld_all_tasks_expert_augmented.zarr"

DATA_KEYS = [
    "img",
    "state",
    "full_state",
    "point_cloud",
    "depth",
    "action",
    "instruction",
]

def main():
    if os.path.exists(OUTPUT_ZARR):
        cprint(f"Removing existing {OUTPUT_ZARR}", "red")
        shutil.rmtree(OUTPUT_ZARR)  
    
    data_acc = {k: [] for k in DATA_KEYS}
    episode_ends_acc = []
    step_offset = 0
    
    task_stats = []
    
    for path in INPUT_ZARRS:
        cprint(f"\nLoading {path}", "cyan")
        
        if not os.path.exists(path):
            cprint(f"  ✗ File not found: {path}", "red")
            continue
        
        root = zarr.open(path, mode="r")
        data_grp = root["data"]
        meta_grp = root["meta"]
        
        task_name = os.path.basename(path).replace("metaworld_", "").replace("_expert_augmented.zarr", "")
        
        num_steps = len(data_grp["action"])
        num_episodes = len(meta_grp["episode_ends"])
        
        cprint(f"  Task: {task_name}", "yellow")
        cprint(f"  Episodes: {num_episodes}, Steps: {num_steps}", "yellow")
        
        for k in DATA_KEYS:
            data_acc[k].append(data_grp[k][:])
        
        episode_ends = meta_grp["episode_ends"][:]
        episode_ends_offset = episode_ends + step_offset
        episode_ends_acc.append(episode_ends_offset)
        
        step_offset += num_steps
        
        task_stats.append({
            'task': task_name,
            'episodes': num_episodes,
            'steps': num_steps
        })
    
    cprint("\nConcatenating data...", "cyan")
    data_cat = {}
    for k in DATA_KEYS:
        if len(data_acc[k]) == 0:
            cprint(f"  ✗ No data for key: {k}", "red")
            continue
        
        if k == "instruction":
            data_cat[k] = np.concatenate(data_acc[k], axis=0)
        else:
            data_cat[k] = np.concatenate(data_acc[k], axis=0)
        
        cprint(f"  {k}: {data_cat[k].shape}, dtype: {data_cat[k].dtype}", "green")
    
    episode_ends_cat = np.concatenate(episode_ends_acc, axis=0)
    
    cprint(f"\nWriting to {OUTPUT_ZARR}...", "cyan")
    root_out = zarr.open(OUTPUT_ZARR, mode="w")
    data_out = root_out.create_group("data")
    meta_out = root_out.create_group("meta")
    
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    
    for k, arr in data_cat.items():
        chunks = (100,) + arr.shape[1:]
        
        if k == "instruction":
            data_out.create_dataset(
                k,
                data=arr.astype(str),  
                chunks=chunks,
                dtype=str,
                compressor=compressor,
            )
        else:
            data_out.create_dataset(
                k,
                data=arr,
                chunks=chunks,
                dtype=arr.dtype,
                compressor=compressor,
            )
    
    meta_out.create_dataset(
        "episode_ends",
        data=episode_ends_cat,
        dtype="int64",
        compressor=compressor,
    )
    
    cprint("\n" + "="*80, "green")
    cprint("Concatenation Complete!", "green")
    cprint("="*80, "green")
    
    cprint("\nPer-Task Statistics:", "yellow")
    for stat in task_stats:
        cprint(f"  {stat['task']:20s} - {stat['episodes']:3d} episodes, {stat['steps']:5d} steps", "cyan")
    
    cprint(f"\nCombined Dataset:", "yellow")
    cprint(f"  Total episodes: {len(episode_ends_cat)}", "green")
    cprint(f"  Total steps: {len(data_cat['action'])}", "green")
    cprint(f"  Unique instructions: {len(np.unique(data_cat['instruction']))}", "green")
    
    cprint(f"\nData Shapes:", "yellow")
    for k, arr in data_cat.items():
        cprint(f"  {k:15s}: {arr.shape}", "cyan")
    
    cprint(f"\n✓ Saved to {OUTPUT_ZARR}", "green")
    cprint("="*80 + "\n", "green")

if __name__ == "__main__":
    main()