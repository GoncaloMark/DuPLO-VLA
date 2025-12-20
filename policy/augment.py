import argparse
import os
import zarr
import numpy as np
from termcolor import cprint
import cv2
from pathlib import Path
import shutil

def apply_lighting_variation(img, variation_type='random'):
    """Apply lighting variations to image"""
    img_float = img.astype(np.float32) / 255.0
    
    if variation_type == 'random':
        variation_type = np.random.choice(['brighten', 'darken', 'contrast', 'gamma'])
    
    if variation_type == 'brighten':
        factor = np.random.uniform(1.2, 1.5)
        img_float = np.clip(img_float * factor, 0, 1)
    
    elif variation_type == 'darken':
        factor = np.random.uniform(0.5, 0.8)
        img_float = np.clip(img_float * factor, 0, 1)
    
    elif variation_type == 'contrast':
        factor = np.random.uniform(0.7, 1.3)
        mean = img_float.mean()
        img_float = np.clip((img_float - mean) * factor + mean, 0, 1)
    
    elif variation_type == 'gamma':
        gamma = np.random.uniform(0.7, 1.3)
        img_float = np.power(img_float, gamma)
    
    return (img_float * 255).astype(np.uint8)

def apply_background_variation(img, variation_type='random'):
    """Apply background variations"""
    if variation_type == 'random':
        variation_type = np.random.choice(['noise', 'blur', 'color_shift'])
    
    if variation_type == 'noise':
        noise = np.random.normal(0, np.random.uniform(5, 15), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    elif variation_type == 'blur':
        kernel_size = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    elif variation_type == 'color_shift':
        shift = np.random.randint(-20, 20, size=3)
        img = np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)
    
    return img

def apply_combined_augmentation(img):
    """Apply both lighting and background variations"""
    img = apply_lighting_variation(img, 'random')
    img = apply_background_variation(img, 'random')
    return img

def augment_zarr_dataset(input_path, output_path, augment_ratio=0.2, seed=42):
    """
    Augment a zarr dataset by creating variations of a subset of data
    
    Args:
        input_path: Path to input zarr file
        output_path: Path to output zarr file
        augment_ratio: Ratio of data to augment (0.2 = 20%)
        seed: Random seed
    """
    np.random.seed(seed)
    
    cprint(f"Loading data from {input_path}", "cyan")
    input_root = zarr.open(input_path, mode='r')
    
    img_data = input_root['data']['img'][:]
    state_data = input_root['data']['state'][:]
    full_state_data = input_root['data']['full_state'][:]
    point_cloud_data = input_root['data']['point_cloud'][:]
    depth_data = input_root['data']['depth'][:]
    action_data = input_root['data']['action'][:]
    instruction_data = input_root['data']['instruction'][:]
    episode_ends = input_root['meta']['episode_ends'][:]
    
    total_timesteps = len(img_data)
    num_episodes = len(episode_ends)
    
    cprint(f"Original dataset: {num_episodes} episodes, {total_timesteps} timesteps", "yellow")
    
    num_episodes_to_augment = int(num_episodes * augment_ratio)
    cprint(f"Will augment {num_episodes_to_augment} episodes ({augment_ratio*100:.0f}%)", "yellow")
    
    episodes_to_augment = np.random.choice(num_episodes, num_episodes_to_augment, replace=False)
    episodes_to_augment = sorted(episodes_to_augment)
    
    cprint(f"Selected episodes: {episodes_to_augment[:10]}{'...' if len(episodes_to_augment) > 10 else ''}", "cyan")
    
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    
    augmented_imgs = []
    augmented_states = []
    augmented_full_states = []
    augmented_point_clouds = []
    augmented_depths = []
    augmented_actions = []
    augmented_instructions = []
    new_episode_ends = []
    
    current_count = 0
    
    for ep_idx in range(num_episodes):
        start_idx = episode_starts[ep_idx]
        end_idx = episode_ends[ep_idx]
        
        ep_imgs = img_data[start_idx:end_idx]
        ep_states = state_data[start_idx:end_idx]
        ep_full_states = full_state_data[start_idx:end_idx]
        ep_point_clouds = point_cloud_data[start_idx:end_idx]
        ep_depths = depth_data[start_idx:end_idx]
        ep_actions = action_data[start_idx:end_idx]
        ep_instructions = instruction_data[start_idx:end_idx]
        
        ep_length = end_idx - start_idx
        
        augmented_imgs.append(ep_imgs)
        augmented_states.append(ep_states)
        augmented_full_states.append(ep_full_states)
        augmented_point_clouds.append(ep_point_clouds)
        augmented_depths.append(ep_depths)
        augmented_actions.append(ep_actions)
        augmented_instructions.extend(ep_instructions)
        
        current_count += ep_length
        new_episode_ends.append(current_count)
        
        if ep_idx in episodes_to_augment:
            aug_imgs = np.array([apply_combined_augmentation(img) for img in ep_imgs])
            
            augmented_imgs.append(aug_imgs)
            augmented_states.append(ep_states)
            augmented_full_states.append(ep_full_states)
            augmented_point_clouds.append(ep_point_clouds)
            augmented_depths.append(ep_depths)
            augmented_actions.append(ep_actions)
            augmented_instructions.extend(ep_instructions)
            
            current_count += ep_length
            new_episode_ends.append(current_count)
            
            if ep_idx < 3:  
                cprint(f"  Augmented episode {ep_idx}", "green")
    
    # Stack 
    cprint("Stacking augmented data...", "cyan")
    final_imgs = np.concatenate(augmented_imgs, axis=0)
    final_states = np.concatenate(augmented_states, axis=0)
    final_full_states = np.concatenate(augmented_full_states, axis=0)
    final_point_clouds = np.concatenate(augmented_point_clouds, axis=0)
    final_depths = np.concatenate(augmented_depths, axis=0)
    final_actions = np.concatenate(augmented_actions, axis=0)
    final_instructions = np.array(augmented_instructions, dtype='object')
    final_episode_ends = np.array(new_episode_ends)
    
    cprint(f"Saving augmented data to {output_path}", "cyan")
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    
    output_root = zarr.group(output_path)
    output_data = output_root.create_group('data')
    output_meta = output_root.create_group('meta')
    
    # Save with compression
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    
    img_chunk_size = (100, final_imgs.shape[1], final_imgs.shape[2], final_imgs.shape[3])
    state_chunk_size = (100, final_states.shape[1])
    full_state_chunk_size = (100, final_full_states.shape[1])
    point_cloud_chunk_size = (100, final_point_clouds.shape[1], final_point_clouds.shape[2])
    depth_chunk_size = (100, final_depths.shape[1], final_depths.shape[2])
    action_chunk_size = (100, final_actions.shape[1])
    instruction_chunk_size = (100,)
    
    output_data.create_dataset('img', data=final_imgs, chunks=img_chunk_size, dtype='uint8', compressor=compressor)
    output_data.create_dataset('state', data=final_states, chunks=state_chunk_size, dtype='float32', compressor=compressor)
    output_data.create_dataset('full_state', data=final_full_states, chunks=full_state_chunk_size, dtype='float32', compressor=compressor)
    output_data.create_dataset('point_cloud', data=final_point_clouds, chunks=point_cloud_chunk_size, dtype='float32', compressor=compressor)
    output_data.create_dataset('depth', data=final_depths, chunks=depth_chunk_size, dtype='float32', compressor=compressor)
    output_data.create_dataset('action', data=final_actions, chunks=action_chunk_size, dtype='float32', compressor=compressor)
    output_data.create_dataset('instruction', data=final_instructions.astype(str), chunks=instruction_chunk_size, dtype=str, compressor=compressor)
    
    output_meta.create_dataset('episode_ends', data=final_episode_ends, dtype='int64', compressor=compressor)
    
    cprint(f'-'*50, 'cyan')
    cprint(f'Augmented dataset summary:', 'yellow')
    cprint(f'  Original episodes: {num_episodes}', 'green')
    cprint(f'  Augmented episodes: {num_episodes_to_augment}', 'green')
    cprint(f'  Total episodes: {len(final_episode_ends)}', 'green')
    cprint(f'  Original timesteps: {total_timesteps}', 'green')
    cprint(f'  Total timesteps: {len(final_imgs)}', 'green')
    cprint(f'  Images shape: {final_imgs.shape}', 'green')
    cprint(f'  Increase: {(len(final_imgs) - total_timesteps) / total_timesteps * 100:.1f}%', 'yellow')
    cprint(f'-'*50, 'cyan')

def main(args):
    input_path = args.input_zarr
    
    if args.output_zarr:
        output_path = args.output_zarr
    else:
        input_dir = Path(input_path).parent
        input_name = Path(input_path).stem
        output_path = str(input_dir / f"{input_name}_augmented.zarr")
    
    augment_zarr_dataset(
        input_path=input_path,
        output_path=output_path,
        augment_ratio=args.augment_ratio,
        seed=args.seed
    )
    
    cprint(f"Done! Augmented dataset saved to: {output_path}", "green")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment MetaWorld dataset with lighting and background variations")
    parser.add_argument('--input_zarr', type=str, required=True, help='Path to input zarr dataset')
    parser.add_argument('--output_zarr', type=str, default=None, help='Path to output zarr dataset (auto-generated if not provided)')
    parser.add_argument('--augment_ratio', type=float, default=0.2, help='Ratio of episodes to augment (default: 0.2 for 20%%)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    main(args)