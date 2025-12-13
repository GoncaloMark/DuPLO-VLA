if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np

from diffusion_policy_3d.dataset.metaworld_dataset import MetaworldDataset

from train_e2e import EndToEndRobotPolicy

def check_gradients(model):
    """Check if gradients are flowing"""
    grad_info = {}
    
    vlm_grads = [p.grad.norm().item() for p in model.planner.vlm.parameters() 
                 if p.grad is not None]
    latent_grads = [p.grad.norm().item() for p in model.planner.task_encoder.parameters() 
                    if p.grad is not None]
    policy_grads = [p.grad.norm().item() for p in model.policy.parameters() 
                    if p.grad is not None]
    
    grad_info['vlm'] = np.mean(vlm_grads) if vlm_grads else 0.0
    grad_info['latent'] = np.mean(latent_grads) if latent_grads else 0.0
    grad_info['policy'] = np.mean(policy_grads) if policy_grads else 0.0
    
    return grad_info


def run_overfitting_test():
    print("="*80)
    print("OVERFITTING TEST - Architecture Validation")
    print("="*80)
    print("Purpose: Verify model can memorize 5 episodes")
    print("Expected: Loss < 0.01 within 100-200 epochs")
    print("="*80 + "\n")
    
    task_name = "pick-place"
    device = "cuda:0"
    
    # TINY DATASET
    max_train_episodes = 5  
    batch_size = 16      
    num_workers = 2
    horizon = 16
    pad_before = 1
    pad_after = 7
    
    vlm_model_name = "Qwen/Qwen3-VL-8B-Instruct"
    latent_dim = 512
    use_point_crop = True
    condition_type = "film"
    use_down_condition = True
    use_mid_condition = True
    use_up_condition = True
    diffusion_step_embed_dim = 128
    down_dims = [512, 1024, 2048]
    crop_shape = [80, 80]
    encoder_output_dim = 64
    kernel_size = 5
    n_action_steps = 8
    n_obs_steps = 2
    n_groups = 8
    num_inference_steps = 10
    obs_as_global_cond = True
    use_pc_color = True
    pointnet_type = "pointnet"
    
    num_train_timesteps = 100
    beta_start = 0.0001
    beta_end = 0.02
    beta_schedule = "squaredcos_cap_v2"
    
    # LoRA config
    use_lora = True
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    
    # OPTIMIZED FOR OVERFITTING
    vlm_lr = 1e-4          # 2x higher 
    latent_lr = 5e-4       # 5x higher
    policy_lr = 5e-4       # 5x higher
    weight_decay = 0.0     # NO regularization (we want to overfit!)
    num_epochs = 500       # More epochs to see convergence
    
    log_dir = f'./logs/overfit_test_{task_name}'
    sample_every = 1  
    
    # Shape meta
    shape_meta = {
        'obs': {
            'point_cloud': {
                'shape': [1024, 6],
                'type': 'point_cloud'
            },
            'agent_pos': {
                'shape': [9],
                'type': 'low_dim'
            },
            'rgb_image': {
                'shape': [128, 128, 3],
                'type': 'rgb'
            }
        },
        'action': {
            'shape': [4]
        }
    }
    
    print("Configuration:")
    print(f"  Task: {task_name}")
    print(f"  Device: {device}")
    print(f"  Episodes: {max_train_episodes}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  VLM LR: {vlm_lr} (2x production)")
    print(f"  Policy LR: {policy_lr} (5x production)")
    print(f"  Weight decay: {weight_decay} (disabled for overfitting)")
    print(f"  Use LoRA: {use_lora}")
    print()
    
    print("Loading dataset...")
    train_dataset = MetaworldDataset(
        zarr_path=f'/data/home/g.marques/storage/DuPLO-VLA/data/metaworld_pick-place_expert.zarr',
        horizon=horizon,
        pad_before=pad_before,
        pad_after=pad_after,
        seed=42,
        val_ratio=0.0,  
        max_train_episodes=max_train_episodes
    )
    print(f"Dataset: {len(train_dataset)} samples from {max_train_episodes} episodes\n")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=False
    )
    
    print("Creating model...")
    model = EndToEndRobotPolicy(
        vlm_model_name=vlm_model_name,
        latent_dim=latent_dim,
        shape_meta=shape_meta,
        use_point_crop=use_point_crop,
        condition_type=condition_type,
        use_down_condition=use_down_condition,
        use_mid_condition=use_mid_condition,
        use_up_condition=use_up_condition,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=down_dims,
        crop_shape=crop_shape,
        encoder_output_dim=encoder_output_dim,
        horizon=horizon,
        kernel_size=kernel_size,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        n_groups=n_groups,
        num_inference_steps=num_inference_steps,
        obs_as_global_cond=obs_as_global_cond,
        use_pc_color=use_pc_color,
        pointnet_type=pointnet_type,
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = model.to(device)
    print(f"Policy device: {next(model.policy.parameters()).device}")
    print(f"Obs encoder device: {next(model.policy.obs_encoder.parameters()).device}")
    print(f"Extractor device: {next(model.policy.obs_encoder.extractor.parameters()).device}")
    print(f"Planner VLM device: {next(model.planner.vlm.parameters()).device}") 
    # Count parameters
    vlm_params = sum(p.numel() for p in model.planner.vlm.parameters())
    latent_params = sum(p.numel() for p in model.planner.task_encoder.parameters())
    policy_params = sum(p.numel() for p in model.policy.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model created:")
    print(f"    VLM: {vlm_params:,} parameters")
    print(f"    Latent Encoder: {latent_params:,} parameters")
    print(f"    Policy: {policy_params:,} parameters")
    print(f"    Total: {total_params:,} parameters\n")
    
    normalizer = train_dataset.get_normalizer()
    normalizer.params_dict['task_latent'] = nn.ParameterDict({
        'mean': torch.zeros(latent_dim),
        'scale': torch.ones(latent_dim),
        'offset': torch.zeros(latent_dim)
    })
    model.policy.set_normalizer(normalizer)
    model.policy.normalizer = model.policy.normalizer.to(device)
    print(f"Normalizer device: {next(model.policy.normalizer.parameters()).device if hasattr(model.policy.normalizer, 'parameters') else 'no parameters'}") 
    print("Setting up optimizer...")
    param_groups = [
        {
            'params': [p for p in model.planner.vlm.parameters() if p.requires_grad],
            'lr': vlm_lr,
            'name': 'vlm'
        },
        {
            'params': model.planner.task_encoder.parameters(),
            'lr': latent_lr,
            'name': 'latent_encoder'
        },
        {
            'params': model.policy.parameters(),
            'lr': policy_lr,
            'name': 'policy'
        }
    ]
    
    optimizer = optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
        betas=(0.95, 0.999),
        eps=1e-8
    )
    print(f"Optimizer: AdamW with 3 parameter groups\n")
    
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True, parents=True)
    
    # Save one batch for testing
    train_sampling_batch = None
    
    print("Checking gradients on first batch...")
    batch = next(iter(train_loader))
    obs_dict = {}
    for k, v in batch['obs'].items():
        if isinstance(v, torch.Tensor):
            obs_dict[k] = v.to(device)
        else:
            obs_dict[k] = v

    actions = batch['action'].to(device)
    instructions = batch['obs']['instruction']
    
    train_sampling_batch = {
        'obs': obs_dict,
        'action': actions,
        'instruction': instructions
    }
 
    model.train()
    loss, _ = model(
        obs_dict=obs_dict,
        instruction_text=instructions,
        actions=actions,
        compute_loss=True
    )
    loss.backward()
    
    grad_info = check_gradients(model)
    
    print(f"Initial checks:")
    print(f"    Loss: {loss.item():.4f}")
    print(f"    VLM grad norm: {grad_info['vlm']:.6f}")
    print(f"    Latent grad norm: {grad_info['latent']:.6f}")
    print(f"    Policy grad norm: {grad_info['policy']:.6f}")
    
    if grad_info['vlm'] < 1e-8:
        print("WARNING: VLM gradients very small!")
    if grad_info['policy'] < 1e-8:
        print("WARNING: Policy gradients very small!")
    
    print()
    optimizer.zero_grad()
    
    print("="*80)
    print("Starting training...")
    print("Watch for: Loss should steadily decrease to < 0.01")
    print("="*80 + "\n")
    
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for _, batch in enumerate(pbar):
            obs_dict = {}
            for k, v in batch['obs'].items():
                if isinstance(v, torch.Tensor):
                    obs_dict[k] = v.to(device)
                else:
                    obs_dict[k] = v
            actions = batch['action'].to(device, non_blocking=True)
            instructions = batch['obs']['instruction']
            
            loss, _ = model(
                obs_dict=obs_dict,
                instruction_text=instructions,
                actions=actions,
                compute_loss=True
            )
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'step': global_step
            })
        
        epoch_loss /= num_batches
        
        if epoch % sample_every == 0:
            model.eval()
            
            torch.manual_seed(42)
            np.random.seed(42)
            
            with torch.no_grad():
                obs_dict = {}
                for k, v in train_sampling_batch['obs'].items():
                    if isinstance(v, torch.Tensor):
                        obs_dict[k] = v.to(device)
                    else:
                        obs_dict[k] = v

                gt_action = train_sampling_batch['action'].to(device)
                instructions = train_sampling_batch['instruction']
                
                all_preds = []
                for _ in range(10):
                    result = model(
                        obs_dict=obs_dict,
                        instruction_text=instructions,
                        actions=None,
                        compute_loss=False
                    )
                    all_preds.append(result['action_pred'])
                
                all_preds = torch.stack(all_preds)  
                mean_pred = all_preds.mean(dim=0)   
                pred_variance = all_preds.var(dim=0).mean().item()
                
                mse = torch.nn.functional.mse_loss(mean_pred, gt_action).item()
            
            print(f"\nEpoch {epoch:3d}")
            print(f"  Train Loss:      {epoch_loss:.6f}")
            print(f"  MSE (mean):      {mse:.6f}")
            print(f"  Pred variance:   {pred_variance:.6f}")
            
            if epoch_loss < 0.01 and mse < 0.01 and pred_variance < 0.001:
                print("\n" + "="*80)
                print("SUCCESS! Model has successfully overfit the dataset!")
                print(f"  Final Train Loss: {epoch_loss:.6f} < 0.01")
                print(f"  Final MSE: {mse:.6f} < 0.01")
                print(f"  Pred Variance: {pred_variance:.6f} < 0.001")
                print("="*80)
                print("\nArchitecture validation complete!")
                print("You can now proceed to full-scale training.")
                print("="*80 + "\n")
                break
    
    print("\n" + "="*80)
    print("OVERFITTING TEST COMPLETE")
    print("="*80)
    print()
    
    if epoch_loss < 0.01:
        print("SUCCESS: Loss < 0.01")
        print("  Architecture can memorize data correctly!")
        print("  Gradients are flowing properly.")
        print("  Ready for full-scale training!")
    elif epoch_loss < 0.1:
        print("PARTIAL SUCCESS: Loss < 0.1 but > 0.01")
        print("  Model is learning but may need more epochs.")
    else:
        print("NEEDS MORE TRAINING: Loss > 0.1")
        print("  Model is still learning. Consider more epochs.")
    
    print("="*80 + "\n")
    print(f"Logs saved to: {log_dir}")
    print()

if __name__ == "__main__":
    run_overfitting_test()
