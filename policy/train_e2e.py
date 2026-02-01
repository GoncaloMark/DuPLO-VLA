import signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import copy
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

from policy.vlm.latent_encoder import TemporalConsistencyLoss
from vlm.vlm import VisualTaskPlanner
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.dataset.metaworld_dataset import MetaworldDataset

class MetaWrapper:
    def __init__(self, data):
        self.data = data
    def __getitem__(self, key):
        return self.data[key]
    def __repr__(self):
        return f"Wrapper({self.data})"

def prepare_meta(data):
    if isinstance(data, dict):
        if 'shape' in data: 
            return MetaWrapper(data)
        else:
            return {k: prepare_meta(v) for k, v in data.items()}
    return data

class ConfigWrapper(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class EndToEndRobotPolicy(nn.Module):
    """
    Dual-process VLA combining System 2 (VLM planner) with System 1 (DP3 policy)
    """
    def __init__(
        self,
        vlm_model_name="Qwen/Qwen3-VL-8B-Instruct",
        latent_dim=512,
        shape_meta=None,
        # DP3 parameters
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
        use_pc_color=False,
        pointnet_type="pointnet",
        # Noise scheduler
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
    ):
        super().__init__()
        
        # System 2
        self.planner = VisualTaskPlanner(
            model_name=vlm_model_name,
            freeze_vlm=True,
            latent_dim=latent_dim
        )

        self.planner.vlm.print_trainable_parameters()

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="sample"
            )
        
        # shape_meta add task_latent
        self.shape_meta = shape_meta.copy()
        self.shape_meta['obs']['task_latent'] = {
            'shape': [latent_dim],
            'type': 'low_dim'
        }
        
        self.shape_meta = prepare_meta(self.shape_meta)

        # System 1
        self.policy = DP3(
            shape_meta=self.shape_meta,
            noise_scheduler=noise_scheduler,
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
            pointcloud_encoder_cfg = ConfigWrapper({'in_channels': 6,'out_channels': encoder_output_dim,'use_layernorm': True,'final_norm': 'layernorm','normal_channel': False})
        )
        
        self.latent_dim = latent_dim
        self.n_obs_steps = n_obs_steps

        self.temporal_consistency_loss = TemporalConsistencyLoss(weight=0.1)

    @staticmethod
    def scale_grad(x, scale):
        y = x
        if x.requires_grad:
            def hook(grad):
                return grad * scale
            y.register_hook(hook)
        return y

    def forward(
        self,
        obs_dict,
        instruction_text,
        actions=None,
        compute_loss=True,
        dp3_grad_scale=1.0,
        latent_update_mask=None,  # (B, T) or (T,)
        latent_group_id=None
    ):
        B = obs_dict['point_cloud'].shape[0]
        T = obs_dict['point_cloud'].shape[1]
        
        # Handle default masks if not provided
        if latent_update_mask is None:
            # Default: update every 3 timesteps
            latent_update_mask = torch.zeros(T, dtype=torch.bool, device=next(self.parameters()).device)
            latent_update_mask[::3] = True
            latent_group_id = torch.arange(T, device=latent_update_mask.device) // 3
        
        # Ensure masks are on correct device
        device = next(self.parameters()).device
        if isinstance(latent_update_mask, torch.Tensor):
            latent_update_mask = latent_update_mask.to(device)
        if isinstance(latent_group_id, torch.Tensor):
            latent_group_id = latent_group_id.to(device)
        
        # Handle batched vs unbatched masks
        if latent_update_mask.ndim == 1:
            # (T,) -> (B, T)
            latent_update_mask = latent_update_mask.unsqueeze(0).expand(B, -1)
        if latent_group_id.ndim == 1:
            # (T,) -> (B, T)
            latent_group_id = latent_group_id.unsqueeze(0).expand(B, -1)
        
        # System 2: Compute latents only for update timesteps
        # Collect unique update timesteps across batch
        update_timesteps = latent_update_mask[0].nonzero(as_tuple=True)[0]  # Assume same schedule for all batch items
        num_updates = len(update_timesteps)
        
        if num_updates > 0:
            # Extract RGB images only for update timesteps
            # obs_dict['rgb_image']: (B, T, H, W, C)
            update_images_list = []
            for t_idx in update_timesteps:
                update_images_list.append(obs_dict['rgb_image'][:, t_idx])  # (B, H, W, C)
            
            update_images = torch.stack(update_images_list, dim=1)  # (B, num_updates, H, W, C)
            
            # Flatten batch and time for VLM processing
            update_images_flat = update_images.reshape(B * num_updates, *update_images.shape[2:])
            
            # Repeat instructions for each update
            update_instructions = [inst for inst in instruction_text for _ in range(num_updates)]
            
            # Run System 2 (VLM + Encoder)
            plan_output = self.planner.plan(
                image=update_images_flat,
                instruction=update_instructions,
                training=self.training,
                return_reconstruction_loss=True
            )
            
            # Reshape computed latents: (B*num_updates, latent_dim) -> (B, num_updates, latent_dim)
            computed_latents = plan_output['latent'].view(B, num_updates, -1)
            
            # Expand latents to all timesteps using group IDs
            task_latents_expanded = torch.zeros(
                B, T, self.latent_dim,
                device=computed_latents.device,
                dtype=computed_latents.dtype
            )
            
            clamped_group_ids = latent_group_id.clamp(0, num_updates - 1)
            batch_indices = torch.arange(B, device=computed_latents.device)[:, None].expand(B, T)
            task_latents_expanded = computed_latents[batch_indices, clamped_group_ids]
            
            reconstruction_loss = plan_output['reconstruction_loss']
            reconstruction_loss_dict = plan_output.get('reconstruction_loss_dict', {})
            
        else:
            # No updates in this batch (shouldn't happen normally)
            task_latents_expanded = torch.zeros(B, T, self.latent_dim, device=device)
            reconstruction_loss = torch.tensor(0.0, device=device)
            reconstruction_loss_dict = {}
        
        # Apply gradient scaling from DP3 to encoder
        if self.training:
            task_latents_expanded = self.scale_grad(task_latents_expanded, scale=0.1)
        
        task_latents_expanded = task_latents_expanded.to(dtype=torch.float32)
        
        # Prepare observation dict with task latents
        obs_dict_with_task = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in obs_dict.items()
            if k not in ['instruction', 'rgb_image']
        }
        obs_dict_with_task['task_latent'] = task_latents_expanded

        # System 1: Diffusion Policy
        if compute_loss and actions is not None:
            batch = {
                'obs': obs_dict_with_task,
                'action': actions.to(device)
            }

            diff_loss, loss_dict = self.policy.compute_loss(batch)
            diff_loss = diff_loss * dp3_grad_scale
            
            # NEW: Compute temporal consistency loss
            temporal_loss, temporal_loss_dict = self.temporal_consistency_loss(
                latents=task_latents_expanded,
                update_mask=latent_update_mask
            )
            
            # Total loss
            total_loss = diff_loss + reconstruction_loss + temporal_loss
            
            # Logging
            loss_dict['diffusion_loss'] = diff_loss.item() / max(dp3_grad_scale, 1e-8)
            if reconstruction_loss_dict:
                loss_dict.update(reconstruction_loss_dict)
            loss_dict.update(temporal_loss_dict)
            loss_dict['dp3_grad_scale'] = dp3_grad_scale
            loss_dict['num_vlm_calls'] = num_updates  # Track efficiency

            return total_loss, loss_dict
        else:
            # Inference mode
            result = self.policy.predict_action(obs_dict_with_task)
            return result

class EndToEndTrainer:
    def __init__(
        self,
        model: EndToEndRobotPolicy,
        train_dataset: MetaworldDataset,
        val_dataset: MetaworldDataset,
        # Learning rates
        latent_lr=1e-4,
        policy_lr=1e-4,
        weight_decay=1e-6,
        # Training config
        batch_size=64,
        num_workers=8,
        gradient_accumulate_every=1,
        # EMA config
        use_ema=True,
        ema_update_after_step=0,
        ema_inv_gamma=1.0,
        ema_power=0.75,
        ema_min_value=0.0,
        ema_max_value=0.9999,
        # LR scheduler
        lr_scheduler_type="cosine",
        lr_warmup_steps=500,
        num_epochs=3000,
        # Phased training config
        pre_alignment_epochs=50,      # Phase 1: encoder only with reconstruction
        warmup_epochs=50,             # Phase 2: gradual DP3 introduction
        # Logging and checkpointing
        log_dir="./logs/e2e_training",
        checkpoint_every=200,
        val_every=1,
        sample_every=5,
        device="cuda:0",
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.num_epochs = num_epochs
        
        # Phased training configuration
        self.pre_alignment_epochs = pre_alignment_epochs
        self.warmup_epochs = warmup_epochs
        self.total_warmup_epochs = pre_alignment_epochs + warmup_epochs
        
        # Setup datasets and dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=False
        )
        
        # normalizer
        normalizer = train_dataset.get_normalizer()
        normalizer.params_dict['task_latent'] = nn.ParameterDict({
            'mean': torch.zeros(self.model.latent_dim),
            'std': torch.ones(self.model.latent_dim)
        })

        self.model.policy.set_normalizer(normalizer)
        
        self.optimizer = self._setup_optimizer(
            latent_lr=latent_lr,
            policy_lr=policy_lr,
            weight_decay=weight_decay
        )
        
        self.lr_scheduler = self._setup_lr_scheduler(
            lr_scheduler_type=lr_scheduler_type,
            lr_warmup_steps=lr_warmup_steps,
            num_training_steps=(len(self.train_loader) * num_epochs) // gradient_accumulate_every
        )
        
        self.use_ema = use_ema
        self.ema_model = None
        self.ema = None
        if use_ema:
            shared_vlm = self.model.planner.vlm
            self.model.planner.vlm = None

            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.to(device)
            self.ema_model.policy.set_normalizer(normalizer)

            self.model.planner.vlm = shared_vlm
            self.ema_model.planner.vlm = shared_vlm
            
            self.ema = EMAModel(
                model=self.ema_model,
                update_after_step=ema_update_after_step,
                inv_gamma=ema_inv_gamma,
                power=ema_power,
                min_value=ema_min_value,
                max_value=ema_max_value
            )
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))

        self.checkpoint_dir = self.log_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_test_score = float('-inf')
        
        self.checkpoint_every = checkpoint_every
        self.val_every = val_every
        self.sample_every = sample_every
        
        self.train_sampling_batch = None

        signal.signal(signal.SIGTERM, self._emergency_save)

    def _emergency_save(self, signum, frame):
        print("\n[!] SIGTERM received. Saving emergency checkpoint...")
        self.save_checkpoint(tag='emergency_shutdown')
        sys.exit(0)
        
    def _setup_optimizer(self, latent_lr, policy_lr, weight_decay):
        """Setup optimizer with different learning rates for different components"""
        param_groups = [
            {
                'params': self.model.planner.task_encoder.parameters(),
                'lr': latent_lr,
                'name': 'latent_encoder'
            },
            {
                'params': self.model.policy.parameters(),
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
        
        return optimizer
    
    def _setup_lr_scheduler(self, lr_scheduler_type, lr_warmup_steps, num_training_steps):
        """Setup learning rate scheduler"""
        from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
        
        scheduler = get_scheduler(
            lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=self.global_step - 1
        )
        
        return scheduler
    
    def get_training_phase(self, epoch):
        """Determine current training phase"""
        if epoch < self.pre_alignment_epochs:
            return "pre_alignment"
        elif epoch < self.total_warmup_epochs:
            return "warmup"
        else:
            return "main_training"
    
    def get_dp3_grad_scale(self, epoch):
        """Get DP3 gradient scale for current epoch"""
        phase = self.get_training_phase(epoch)
        
        if phase == "pre_alignment":
            # Phase 1: No DP3 gradients at all
            return 0.0
        elif phase == "warmup":
            # Phase 2: Linear warmup from 0 to 1
            warmup_progress = (epoch - self.pre_alignment_epochs) / self.warmup_epochs
            return warmup_progress
        else:
            # Phase 3: Full DP3 gradients
            return 1.0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        
        phase = self.get_training_phase(epoch)
        dp3_grad_scale = self.get_dp3_grad_scale(epoch)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [{phase}]")
        
        for batch_idx, batch in enumerate(pbar):
            obs_dict = {}
            for k, v in batch['obs'].items():
                if isinstance(v, torch.Tensor):
                    obs_dict[k] = v.to(self.device)
                else:
                    obs_dict[k] = v
            actions = batch['action'].to(self.device, non_blocking=True)
            instructions = batch['obs']['instruction']
            
            if self.train_sampling_batch is None:
                self.train_sampling_batch = {
                    'obs': obs_dict,
                    'action': actions,
                    'instruction': instructions,
                    'latent_update_mask': batch['latent_update_mask'].to(self.device),  
                    'latent_group_id': batch['latent_group_id'].to(self.device),
                }

            latent_update_mask = batch['latent_update_mask'].to(self.device)
            latent_group_id = batch['latent_group_id'].to(self.device)
            
            loss, loss_dict = self.model(
                obs_dict=obs_dict,
                instruction_text=instructions,
                actions=actions,
                compute_loss=True,
                dp3_grad_scale=dp3_grad_scale,
                latent_update_mask=latent_update_mask,
                latent_group_id=latent_group_id
            )
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulate_every
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulate_every == 0:
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.writer.add_scalar('Charts/Gradient_Norm', total_norm, self.global_step)

                if self.global_step % 100 == 0:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and 'policy' in name: 
                            self.writer.add_histogram(f'Gradients/{name}', param.grad, self.global_step)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                
                if self.use_ema:
                    self.ema.step(self.model)
                
                # Logging
                self.global_step += 1
                epoch_loss += loss.item() * self.gradient_accumulate_every

                self.writer.add_scalar('Train/Total_Loss', loss.item() * self.gradient_accumulate_every, self.global_step)
                self.writer.add_scalar('Train/DP3_Grad_Scale', dp3_grad_scale, self.global_step)
                self.writer.add_scalar('Train/LR_Policy', self.optimizer.param_groups[1]['lr'], self.global_step)
                
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'Loss_Components/{k}', v, self.global_step)
                    if k not in epoch_metrics:
                        epoch_metrics[k] = 0.0
                    epoch_metrics[k] += v
                
                pbar.set_postfix({
                    'loss': f"{loss.item() * self.gradient_accumulate_every:.4f}",
                    'phase': phase,
                    'dp3_scale': f"{dp3_grad_scale:.2f}",
                    'step': self.global_step,
                    'lr': f"{self.lr_scheduler.get_last_lr()[0]:.2e}"
                })
        
        num_batches = len(self.train_loader) // self.gradient_accumulate_every
        epoch_loss /= num_batches
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        epoch_metrics['phase'] = phase
        
        return epoch_loss, epoch_metrics
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        policy = self.ema_model if self.use_ema else self.model
        policy.eval()
        
        val_loss = 0.0
        val_metrics = {}
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            obs_dict = {}
            for k, v in batch['obs'].items():
                if isinstance(v, torch.Tensor):
                    obs_dict[k] = v.to(self.device)
                else:
                    obs_dict[k] = v
            actions = batch['action'].to(self.device, non_blocking=True)
            instructions = batch['obs']['instruction']

            latent_update_mask = batch['latent_update_mask'].to(self.device)
            latent_group_id = batch['latent_group_id'].to(self.device)
            
            # Always use full gradients for validation
            loss, loss_dict = policy(
                obs_dict=obs_dict,
                instruction_text=instructions,
                actions=actions,
                compute_loss=True,
                dp3_grad_scale=1.0,  # Full scale for validation
                latent_update_mask=latent_update_mask,
                latent_group_id=latent_group_id
            )
            
            val_loss += loss.item()
            self.writer.add_scalar('Val/Total_Loss', val_loss, self.global_step)

            for k, v in loss_dict.items():
                self.writer.add_scalar(f'Val_Metrics/{k}', v, self.global_step)
                if k not in val_metrics:
                    val_metrics[k] = 0.0
                val_metrics[k] += v
        
        val_loss /= len(self.val_loader)
        val_metrics = {k: v / len(self.val_loader) for k, v in val_metrics.items()}
        
        return val_loss, val_metrics
    
    @torch.no_grad()
    def sample_actions(self):
        """Sample actions on training batch to check policy"""
        policy = self.ema_model if self.use_ema else self.model
        policy.eval()
        
        batch = self.train_sampling_batch
        obs_dict = {}
        for k, v in batch['obs'].items():
            if isinstance(v, torch.Tensor):
                obs_dict[k] = v.to(self.device)
            else:
                obs_dict[k] = v
        gt_action = batch['action'].to(self.device, non_blocking=True)
        instructions = batch['obs']['instruction']

        latent_update_mask = batch.get('latent_update_mask')
        latent_group_id = batch.get('latent_group_id')
        
        if latent_update_mask is not None:
            latent_update_mask = latent_update_mask.to(self.device)
        if latent_group_id is not None:
            latent_group_id = latent_group_id.to(self.device)

        result = policy(
            obs_dict=obs_dict,
            instruction_text=instructions,
            actions=None,
            compute_loss=False,
            latent_update_mask=latent_update_mask,
            latent_group_id=latent_group_id
        )
        
        pred_action = result['action_pred']
        mse = torch.nn.functional.mse_loss(pred_action, gt_action)

        self.writer.add_scalar('Analysis/Action_MSE', mse.item(), self.global_step)
        
        return {'train_action_mse_error': mse.item()}
    
    def save_checkpoint(self, tag="latest", is_best=False):
        """Save checkpoint WITHOUT VLM weights"""

        deploy_model = self.ema_model if self.use_ema else self.model

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,

            "task_encoder_state_dict": deploy_model.planner.task_encoder.state_dict(),
            "policy_state_dict": deploy_model.policy.state_dict(),

            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),

            "best_val_loss": self.best_val_loss,
            "best_test_score": self.best_test_score,
        }

        checkpoint_path = self.checkpoint_dir / f"{tag}.ckpt"
        torch.save(checkpoint, checkpoint_path)

        deploy_dir = self.log_dir / "deployment"
        deploy_dir.mkdir(exist_ok=True)

        planner_path = deploy_dir / f"system2_latent_encoder_{tag}.pth"
        torch.save(
            {
                "state_dict": deploy_model.planner.task_encoder.state_dict(),
                "latent_dim": deploy_model.latent_dim,
                "model_name": "LatentTaskEncoder",
                "epoch": self.epoch,
                "global_step": self.global_step,
            },
            planner_path,
        )

        policy_path = deploy_dir / f"system1_policy_{tag}.pth"
        torch.save(
            {
                "state_dict": deploy_model.policy.state_dict(),
                "normalizer": (
                    deploy_model.policy.normalizer.state_dict()
                    if hasattr(deploy_model.policy, "normalizer")
                    else None
                ),
                "model_name": "DP3 Policy",
                "epoch": self.epoch,
                "global_step": self.global_step,
            },
            policy_path,
        )

        config_path = deploy_dir / f"config_{tag}.pth"
        torch.save(
            {
                "latent_dim": deploy_model.latent_dim,
                "n_obs_steps": deploy_model.n_obs_steps,
                "epoch": self.epoch,
                "global_step": self.global_step,
            },
            config_path,
        )

        if is_best:
            best_checkpoint_path = self.checkpoint_dir / "best.ckpt"
            torch.save(checkpoint, best_checkpoint_path)

            torch.save(torch.load(planner_path), deploy_dir / "system2_latent_encoder_best.pth")
            torch.save(torch.load(policy_path), deploy_dir / "system1_policy_best.pth")
            torch.save(torch.load(config_path), deploy_dir / "config_best.pth")

            print("\n" + "=" * 80)
            print("Saved best checkpoint (NO VLM WEIGHTS)")
            print(f"  Checkpoint: {best_checkpoint_path}")
            print(f"  Latent encoder: {deploy_dir / 'system2_latent_encoder_best.pth'}")
            print(f"  DP3 policy: {deploy_dir / 'system1_policy_best.pth'}")
            print("=" * 80 + "\n")

        return str(checkpoint_path)

    def load_checkpoint(self, path):
        """Load checkpoint WITHOUT touching the VLM"""

        checkpoint = torch.load(path, map_location=self.device)

        self.model.planner.task_encoder.load_state_dict(
            checkpoint["task_encoder_state_dict"]
        )
        self.model.policy.load_state_dict(
            checkpoint["policy_state_dict"]
        )

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_test_score = checkpoint.get("best_test_score", float("-inf"))

        print(f"Loaded checkpoint from {path}")
        print(f"  Epoch: {self.epoch}")
        print(f"  Global step: {self.global_step}")
        print("  VLM weights were NOT loaded (frozen & external)")

        return self.epoch

    def train(self, resume_from=None):
        """Main training loop"""
        start_epoch = 0
        
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from) + 1
        
        print(f"\n{'='*80}")
        print(f"Starting End-to-End VLA Training with Phased Strategy")
        print(f"{'='*80}")
        print(f"Epochs: {start_epoch} -> {self.num_epochs}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Gradient accumulation: {self.gradient_accumulate_every}")
        print(f"Use EMA: {self.use_ema}")
        print(f"\nPhased Training Schedule:")
        print(f"  Phase 1 (Pre-alignment):  Epochs 0-{self.pre_alignment_epochs} (DP3 frozen)")
        print(f"  Phase 2 (Warmup):         Epochs {self.pre_alignment_epochs}-{self.total_warmup_epochs} (DP3 gradual)")
        print(f"  Phase 3 (Main Training):  Epochs {self.total_warmup_epochs}+ (DP3 full)")
        print(f"{'='*80}\n")
        
        for epoch in range(start_epoch, self.num_epochs):
            self.epoch = epoch
            
            # Training
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # phase-specific info
            phase = train_metrics.get('phase', 'unknown')
            dp3_scale = train_metrics.get('dp3_grad_scale', 1.0)
            print(f"\nEpoch {epoch} [{phase}] - Train Loss: {train_loss:.4f} (DP3 scale: {dp3_scale:.2f})")
            
            step_log = {
                'epoch': epoch,
                'train_loss': train_loss,
                **train_metrics
            }
            
            # Validation
            if (epoch % self.val_every) == 0:
                val_loss, val_metrics = self.validate()
                print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")
                step_log['val_loss'] = val_loss
                step_log.update({f'val_{k}': v for k, v in val_metrics.items()})
                
                # best validation 
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
            
            if (epoch % self.sample_every) == 0:
                sample_metrics = self.sample_actions()
                step_log.update(sample_metrics)
                print(f"Epoch {epoch} - Action MSE: {sample_metrics['train_action_mse_error']:.4f}")
            
            if (epoch % self.checkpoint_every) == 0:
                is_best = False
                if 'test_mean_score' in step_log:
                    is_best = step_log['test_mean_score'] >= self.best_test_score
                
                self.save_checkpoint(tag='latest', is_best=is_best)
            
            print(f"Step log: {step_log}")
        
        print("\n" + "="*80)
        print("Training completed!")
        print("="*80)

def main():
    device = "cuda:0"
    
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
    
    train_dataset = MetaworldDataset(
        zarr_path=f'data/metaworld_all_tasks_expert_augmented.zarr',
        horizon=16,
        pad_before=1,  # n_obs_steps - 1
        pad_after=7,   # n_action_steps - 1
        seed=42,
        val_ratio=0.02,
        max_train_episodes=90,
        latent_update_interval=3,      # 9Hz update at ~27Hz policy (assuming 16 horizon ~ 0.5s)
        randomize_update_interval=True
    )
    
    val_dataset = train_dataset.get_validation_dataset()

    model = EndToEndRobotPolicy(
        vlm_model_name="Qwen/Qwen3-VL-8B-Instruct",
        latent_dim=512,
        shape_meta=shape_meta,
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
        # Noise scheduler config
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
    )

    trainer = EndToEndTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        # Learning rates
        latent_lr=1e-4,
        policy_lr=1e-4,
        weight_decay=1e-6,
        # Training config
        batch_size=16,
        num_workers=8,
        gradient_accumulate_every=8,
        # EMA config
        use_ema=True,
        ema_update_after_step=0,
        ema_inv_gamma=1.0,
        ema_power=0.75,
        ema_min_value=0.0,
        ema_max_value=0.9999,
        # LR scheduler
        lr_scheduler_type="cosine",
        lr_warmup_steps=500,
        num_epochs=450,
        # Phased training config
        pre_alignment_epochs=5,   # Phase 1: Encoder learns with reconstruction only
        warmup_epochs=10,          # Phase 2: Gradual DP3 gradient introduction
        # Logging
        log_dir=f'./logs/e2e_vla',
        checkpoint_every=50,
        val_every=20,
        sample_every=40,
        device=device,
    )
    
    trainer.train(resume_from=None)

if __name__ == "__main__":
    main()
