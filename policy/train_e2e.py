import itertools
import signal
import sys
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
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

from vlm.latent_encoder import TemporalConsistencyLoss
from vlm.vlm import VisualTaskPlanner
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.dataset.metaworld_dataset import MetaworldDataset
from helpers import ConfigWrapper, prepare_meta


class EndToEndRobotPolicy(nn.Module):
    """
    Dual-process VLA:
        System 2 — VLM planner  (Qwen3-VL frozen + trainable task encoder)
        System 1 — DP3 policy   (UNet diffusion, conditioned on task latent)
    """
    def __init__(
        self,
        vlm_model_name="Qwen/Qwen3-VL-8B-Instruct",
        latent_dim=512,
        shape_meta=None,
        # DP3 params
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

        self.planner = VisualTaskPlanner(
            model_name=vlm_model_name,
            freeze_vlm=True,
            latent_dim=latent_dim,
            num_pooling_queries=16,       
            contrastive_weight=0.1,
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
            prediction_type="sample",
        )

        self.shape_meta = shape_meta.copy()
        self.shape_meta['obs']['task_latent'] = {
            'shape': [latent_dim],
            'type': 'low_dim',
        }
        self.shape_meta = prepare_meta(self.shape_meta)

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
            pointcloud_encoder_cfg=ConfigWrapper({
                'in_channels': 6,
                'out_channels': encoder_output_dim,
                'use_layernorm': True,
                'final_norm': 'layernorm',
                'normal_channel': False,
            }),
        )

        self.latent_dim  = latent_dim
        self.n_obs_steps = n_obs_steps
        self.temporal_consistency_loss = TemporalConsistencyLoss(weight=0.1)

    @staticmethod
    def scale_grad(x, scale):
        """Hook to scale gradients flowing back through the latent."""
        if x.requires_grad and scale != 1.0:
            def hook(grad):
                return grad * scale
            x.register_hook(hook)
        return x

    def forward(
        self,
        obs_dict,
        instruction_text,
        actions=None,
        compute_loss=True,
        dp3_grad_scale=1.0,
        latent_update_mask=None,
        latent_group_id=None,
    ):
        B      = obs_dict['point_cloud'].shape[0]
        T      = obs_dict['point_cloud'].shape[1]
        device = next(self.parameters()).device

        if latent_update_mask is None:
            latent_update_mask = torch.zeros(T, dtype=torch.bool, device=device)
            latent_update_mask[::3] = True
            latent_group_id = torch.arange(T, device=device) // 3

        latent_update_mask = latent_update_mask.to(device)
        latent_group_id    = latent_group_id.to(device)

        if latent_update_mask.ndim == 1:
            latent_update_mask = latent_update_mask.unsqueeze(0).expand(B, -1)
        if latent_group_id.ndim == 1:
            latent_group_id = latent_group_id.unsqueeze(0).expand(B, -1)

        # Use batch-item 0's schedule (all items share same mask from dataset)
        update_timesteps = latent_update_mask[0].nonzero(as_tuple=True)[0]
        num_updates      = len(update_timesteps)

        if num_updates > 0:
            # Gather images: (B, num_updates, H, W, C)
            update_images = torch.stack(
                [obs_dict['rgb_image'][:, t] for t in update_timesteps], dim=1
            )

            # Flatten to (B * num_updates, H, W, C)
            # Layout: [b0t0, b0t1, ..., b0tN, b1t0, b1t1, ...]
            update_images_flat = update_images.reshape(B * num_updates, *update_images.shape[2:])

            # images_flat[i * num_updates + j] = batch item i, update step j
            # so instructions must also be indexed as [i * num_updates + j] = instructions[i]
            update_instructions = [
                instruction_text[i]
                for i in range(B)
                for _ in range(num_updates)
            ]

            plan_output = self.planner.plan(
                image=update_images_flat,
                instruction=update_instructions,
                training=self.training,
                return_reconstruction_loss=True,
            )

            # Reshape: (B * num_updates, latent_dim) → (B, num_updates, latent_dim)
            computed_latents = plan_output['latent'].view(B, num_updates, -1)

            # Expand latents to all T timesteps via group IDs
            clamped_group_ids = latent_group_id.clamp(0, num_updates - 1)
            batch_indices     = torch.arange(B, device=device)[:, None].expand(B, T)
            task_latents_expanded = computed_latents[batch_indices, clamped_group_ids]  # (B, T, latent_dim)

            reconstruction_loss      = plan_output['reconstruction_loss']
            reconstruction_loss_dict = plan_output.get('reconstruction_loss_dict', {})

        else:
            task_latents_expanded    = torch.zeros(B, T, self.latent_dim, device=device)
            reconstruction_loss      = torch.tensor(0.0, device=device)
            reconstruction_loss_dict = {}

        if self.training:
            task_latents_expanded = self.scale_grad(task_latents_expanded, scale=0.1)

        task_latents_expanded = task_latents_expanded.to(dtype=torch.float32)

        obs_dict_with_task = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in obs_dict.items()
            if k not in ('instruction', 'rgb_image')
        }
        obs_dict_with_task['task_latent'] = task_latents_expanded

        if compute_loss and actions is not None:
            batch = {'obs': obs_dict_with_task, 'action': actions.to(device)}
            diff_loss, loss_dict = self.policy.compute_loss(batch)
            diff_loss = diff_loss * dp3_grad_scale

            temporal_loss, temporal_loss_dict = self.temporal_consistency_loss(
                latents=task_latents_expanded,
                update_mask=latent_update_mask,
            )

            total_loss = diff_loss + reconstruction_loss + temporal_loss

            loss_dict['diffusion_loss']   = diff_loss.item() / max(dp3_grad_scale, 1e-8)
            loss_dict['dp3_grad_scale']   = dp3_grad_scale
            loss_dict['num_vlm_calls']    = num_updates
            if reconstruction_loss_dict:
                loss_dict.update(reconstruction_loss_dict)
            loss_dict.update(temporal_loss_dict)

            return total_loss, loss_dict

        else:
            return self.policy.predict_action(obs_dict_with_task)

class EndToEndTrainer:
    def __init__(
        self,
        model: EndToEndRobotPolicy,
        train_dataset: MetaworldDataset,
        val_dataset: MetaworldDataset,
        latent_lr=3e-4,
        policy_lr=1e-4,
        weight_decay=1e-6,
        # Training
        batch_size=32,
        num_workers=8,
        gradient_accumulate_every=4,
        # EMA
        use_ema=True,
        ema_update_after_step=0,
        ema_inv_gamma=1.0,
        ema_power=0.75,
        ema_min_value=0.0,
        ema_max_value=0.9999,
        # LR scheduler
        lr_scheduler_type="cosine",
        lr_warmup_steps=500,
        num_epochs=1000,
        # Phased training  
        pre_alignment_epochs=30,    # Phase 1: reconstruction + contrastive only
        warmup_epochs=50,           # Phase 2: linear DP3 ramp
        # Logging / checkpointing
        log_dir="./logs/e2e_training",
        checkpoint_every=100,
        val_every=5,                
        sample_every=20,
        device="cuda:0",
    ):
        self.model    = model.to(device)
        self.device   = torch.device(device)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.num_epochs = num_epochs

        self.pre_alignment_epochs = pre_alignment_epochs
        self.warmup_epochs        = warmup_epochs
        self.total_warmup_epochs  = pre_alignment_epochs + warmup_epochs

        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
        self.global_step   = 0

        self.log_dir        = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.writer         = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        self.checkpoint_dir = self.log_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.epoch          = 0
        self.best_val_loss  = float('inf')
        self.best_test_score = float('-inf')

        self.checkpoint_every = checkpoint_every
        self.val_every        = val_every
        self.sample_every     = sample_every
        self.train_sampling_batch = None

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=False,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=False,
        )

        normalizer = train_dataset.get_normalizer()
        normalizer.params_dict['task_latent'] = nn.ParameterDict({
            'mean':   torch.zeros(self.model.latent_dim),
            'scale':  torch.ones(self.model.latent_dim),
            'offset': torch.zeros(self.model.latent_dim),
        })
        self.model.policy.set_normalizer(normalizer)
        self.model.policy.normalizer = self.model.policy.normalizer.to(device)

        self.optimizer = self._setup_optimizer(
            latent_lr=latent_lr,
            policy_lr=policy_lr,
            weight_decay=weight_decay,
        )
        self.lr_scheduler = self._setup_lr_scheduler(
            lr_scheduler_type=lr_scheduler_type,
            lr_warmup_steps=lr_warmup_steps,
            num_training_steps=(len(self.train_loader) * num_epochs) // gradient_accumulate_every,
        )

        self.use_ema   = use_ema
        self.ema_model = None
        self.ema       = None
        if use_ema:
            self._setup_ema(normalizer, device, ema_update_after_step,
                            ema_inv_gamma, ema_power, ema_min_value, ema_max_value)

        signal.signal(signal.SIGTERM, self._emergency_save)

    def _setup_ema(self, normalizer, device, update_after_step, inv_gamma, power, min_value, max_value):
        """
        Deep-copy everything except the VLM (too large), then re-attach the
        shared VLM reference after copy. Safer than None-swapping.
        """
        vlm_ref = self.model.planner.vlm
        # Temporarily replace VLM with a tiny placeholder so deepcopy is cheap
        self.model.planner.vlm = None
        self.ema_model = copy.deepcopy(self.model)
        self.model.planner.vlm = vlm_ref          # restore on original

        self.ema_model.to(device)
        self.ema_model.planner.vlm = vlm_ref      # share the same VLM object
        self.ema_model.policy.set_normalizer(normalizer)
        self.ema_model.policy.normalizer = self.ema_model.policy.normalizer.to(device)

        self.ema = EMAModel(
            model=self.ema_model,
            update_after_step=update_after_step,
            inv_gamma=inv_gamma,
            power=power,
            min_value=min_value,
            max_value=max_value,
        )

    def _emergency_save(self, signum, frame):
        print("\n[!] SIGTERM received. Saving emergency checkpoint...")
        self.save_checkpoint(tag='emergency_shutdown')
        sys.exit(0)

    def _setup_optimizer(self, latent_lr, policy_lr, weight_decay):
        """
        Two param groups:
            - task_encoder: higher LR (learning from scratch atop frozen VLM)
            - policy:       standard LR
        """
        return optim.AdamW(
            [
                {'params': self.model.planner.task_encoder.parameters(), 'lr': latent_lr,  'name': 'latent_encoder'},
                {'params': self.model.policy.parameters(), 'lr': policy_lr,  'name': 'policy'},
            ],
            weight_decay=weight_decay,
            betas=(0.95, 0.999),
            eps=1e-8,
        )

    def _setup_lr_scheduler(self, lr_scheduler_type, lr_warmup_steps, num_training_steps):
        from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
        return get_scheduler(
            lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=self.global_step - 1,
        )

    def get_training_phase(self, epoch):
        if epoch < self.pre_alignment_epochs:
            return "pre_alignment"
        elif epoch < self.total_warmup_epochs:
            return "warmup"
        return "main_training"

    def get_dp3_grad_scale(self, epoch):
        phase = self.get_training_phase(epoch)
        if phase == "pre_alignment":
            return 0.0
        elif phase == "warmup":
            progress = (epoch - self.pre_alignment_epochs) / max(self.warmup_epochs, 1)
            return float(progress)
        return 1.0

    def _get_latent_lr_multiplier(self, epoch):
        """
        Reduce latent encoder LR once DP3 takes over — prevents the encoder
        from oscillating under the stronger action-loss signal.
        """
        phase = self.get_training_phase(epoch)
        if phase == "pre_alignment":
            return 1.0   # full LR during alignment phase
        elif phase == "warmup":
            return 0.5
        return 0.2       # fine-tune mode during main training

    def _update_latent_lr(self, epoch):
        multiplier = self._get_latent_lr_multiplier(epoch)
        base_lr = self.optimizer.param_groups[0]['initial_lr'] \
            if 'initial_lr' in self.optimizer.param_groups[0] \
            else 3e-4
        self.optimizer.param_groups[0]['lr'] = base_lr * multiplier

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss    = 0.0
        epoch_metrics = {}

        phase          = self.get_training_phase(epoch)
        dp3_grad_scale = self.get_dp3_grad_scale(epoch)
        self._update_latent_lr(epoch)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [{phase}]")

        for batch_idx, batch in enumerate(pbar):
            obs_dict = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch['obs'].items()
            }
            actions      = batch['action'].to(self.device, non_blocking=True)
            instructions = batch['obs']['instruction']

            if self.train_sampling_batch is None:
                self.train_sampling_batch = {
                    'obs': obs_dict, 'action': actions, 'instruction': instructions,
                    'latent_update_mask': batch['latent_update_mask'].to(self.device),
                    'latent_group_id':    batch['latent_group_id'].to(self.device),
                }

            latent_update_mask = batch['latent_update_mask'].to(self.device)
            latent_group_id    = batch['latent_group_id'].to(self.device)

            loss, loss_dict = self.model(
                obs_dict=obs_dict,
                instruction_text=instructions,
                actions=actions,
                compute_loss=True,
                dp3_grad_scale=dp3_grad_scale,
                latent_update_mask=latent_update_mask,
                latent_group_id=latent_group_id,
            )

            (loss / self.gradient_accumulate_every).backward()

            if (batch_idx + 1) % self.gradient_accumulate_every == 0:
                total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.writer.add_scalar('Charts/Gradient_Norm', total_norm, self.global_step)

                if self.global_step % 100 == 0:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and 'policy' in name:
                            self.writer.add_histogram(
                                f'Gradients/{name}', param.grad, self.global_step)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

                if self.use_ema:
                    self.ema.step(self.model)

                self.global_step += 1
                epoch_loss += loss.item()

                self.writer.add_scalar('Train/Total_Loss',    loss.item(), self.global_step)
                self.writer.add_scalar('Train/DP3_Grad_Scale', dp3_grad_scale, self.global_step)
                self.writer.add_scalar('Train/LR_Encoder',
                    self.optimizer.param_groups[0]['lr'], self.global_step)
                self.writer.add_scalar('Train/LR_Policy',
                    self.optimizer.param_groups[1]['lr'], self.global_step)

                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'Loss_Components/{k}', v, self.global_step)
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v

                pbar.set_postfix({
                    'loss':     f"{loss.item():.4f}",
                    'phase':    phase,
                    'dp3':      f"{dp3_grad_scale:.2f}",
                    'step':     self.global_step,
                    'lr_enc':   f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })

        n = max(len(self.train_loader) // self.gradient_accumulate_every, 1)
        epoch_metrics = {k: v / n for k, v in epoch_metrics.items()}
        epoch_metrics['phase'] = phase
        return epoch_loss / n, epoch_metrics

    @torch.no_grad()
    def validate(self):
        policy = self.ema_model if self.use_ema else self.model
        policy.eval()

        val_loss    = 0.0
        val_metrics = {}
        val_batches = list(itertools.islice(self.val_loader, 100))

        for batch in tqdm(val_batches, desc="Validating"):
            obs_dict = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch['obs'].items()
            }
            actions      = batch['action'].to(self.device, non_blocking=True)
            instructions = batch['obs']['instruction']
            latent_update_mask = batch['latent_update_mask'].to(self.device)
            latent_group_id    = batch['latent_group_id'].to(self.device)

            loss, loss_dict = policy(
                obs_dict=obs_dict,
                instruction_text=instructions,
                actions=actions,
                compute_loss=True,
                dp3_grad_scale=1.0,
                latent_update_mask=latent_update_mask,
                latent_group_id=latent_group_id,
            )

            val_loss += loss.item()
            for k, v in loss_dict.items():
                val_metrics[k] = val_metrics.get(k, 0.0) + v

        n = max(len(val_batches), 1)
        val_metrics = {k: v / n for k, v in val_metrics.items()}

        self.writer.add_scalar('Val/Total_Loss', val_loss / n, self.global_step)
        for k, v in val_metrics.items():
            self.writer.add_scalar(f'Val_Metrics/{k}', v, self.global_step)

        return val_loss / n, val_metrics

    @torch.no_grad()
    def sample_actions(self):
        policy = self.ema_model if self.use_ema else self.model
        policy.eval()

        batch = self.train_sampling_batch
        obs_dict = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch['obs'].items()
        }
        gt_action    = batch['action'].to(self.device)
        instructions = batch['obs']['instruction']

        result = policy(
            obs_dict=obs_dict,
            instruction_text=instructions,
            actions=None,
            compute_loss=False,
            latent_update_mask=batch.get('latent_update_mask'),
            latent_group_id=batch.get('latent_group_id'),
        )

        mse = nn.functional.mse_loss(result['action_pred'], gt_action)
        self.writer.add_scalar('Analysis/Action_MSE', mse.item(), self.global_step)
        return {'train_action_mse_error': mse.item()}

    def save_checkpoint(self, tag="latest", is_best=False):
        """Save checkpoint WITHOUT VLM weights (VLM is loaded separately at deploy time)."""
        deploy_model = self.ema_model if self.use_ema else self.model

        checkpoint = {
            "epoch":               self.epoch,
            "global_step":         self.global_step,
            "task_encoder_state_dict": deploy_model.planner.task_encoder.state_dict(),
            "policy_state_dict":       deploy_model.policy.state_dict(),
            "optimizer_state_dict":    self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "best_val_loss":           self.best_val_loss,
            "best_test_score":         self.best_test_score,
        }

        ckpt_path = self.checkpoint_dir / f"{tag}.ckpt"
        torch.save(checkpoint, ckpt_path)

        deploy_dir = self.log_dir / "deployment"
        deploy_dir.mkdir(exist_ok=True)

        planner_path = deploy_dir / f"system2_latent_encoder_{tag}.pth"
        torch.save({
            "state_dict":  deploy_model.planner.task_encoder.state_dict(),
            "latent_dim":  deploy_model.latent_dim,
            "model_name":  "LatentTaskEncoder",
            "epoch":       self.epoch,
            "global_step": self.global_step,
        }, planner_path)

        policy_path = deploy_dir / f"system1_policy_{tag}.pth"
        torch.save({
            "state_dict":  deploy_model.policy.state_dict(),
            "normalizer":  deploy_model.policy.normalizer.state_dict() if hasattr(deploy_model.policy, "normalizer") else None,
            "model_name":  "DP3 Policy",
            "epoch":       self.epoch,
            "global_step": self.global_step,
        }, policy_path)

        config_path = deploy_dir / f"config_{tag}.pth"
        torch.save({
            "latent_dim":  deploy_model.latent_dim,
            "n_obs_steps": deploy_model.n_obs_steps,
            "epoch":       self.epoch,
            "global_step": self.global_step,
        }, config_path)

        if is_best:
            for src, stem in [(planner_path, "system2_latent_encoder_best.pth"), (policy_path,  "system1_policy_best.pth"), (config_path,  "config_best.pth")]:
                torch.save(torch.load(src), deploy_dir / stem)
            torch.save(checkpoint, self.checkpoint_dir / "best.ckpt")
            print(f"\n{'='*60}\nSaved best checkpoint (no VLM weights)\n{'='*60}\n")

        return str(ckpt_path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.planner.task_encoder.load_state_dict(ckpt["task_encoder_state_dict"])
        self.model.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
        self.epoch           = ckpt["epoch"]
        self.global_step     = ckpt["global_step"]
        self.best_val_loss   = ckpt["best_val_loss"]
        self.best_test_score = ckpt.get("best_test_score", float("-inf"))
        print(f"Loaded checkpoint: epoch={self.epoch}, step={self.global_step}")
        return self.epoch

    def train(self, resume_from=None):
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from) + 1

        print(f"\n{'='*70}")
        print("End-to-End VLA Training  (A6000 schedule)")
        print(f"{'='*70}")
        print(f"Epochs {start_epoch} → {self.num_epochs}  |  device: {self.device}")
        print(f"Phase 1 pre-alignment : 0 – {self.pre_alignment_epochs}  (DP3 frozen)")
        print(f"Phase 2 warmup        : {self.pre_alignment_epochs} – {self.total_warmup_epochs}  (DP3 ramp)")
        print(f"Phase 3 main training : {self.total_warmup_epochs}+  (DP3 full)")
        print(f"{'='*70}\n")

        for epoch in range(start_epoch, self.num_epochs):
            self.epoch = epoch

            train_loss, train_metrics = self.train_epoch(epoch)
            phase = train_metrics.get('phase', '?')
            dp3s  = self.get_dp3_grad_scale(epoch)
            print(f"Epoch {epoch:4d} [{phase}]  train_loss={train_loss:.4f}  dp3_scale={dp3s:.2f}")

            if epoch % self.val_every == 0:
                val_loss, val_metrics = self.validate()
                print(f"             val_loss={val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

            if epoch % self.sample_every == 0 and self.train_sampling_batch is not None:
                s = self.sample_actions()
                print(f"             action_mse={s['train_action_mse_error']:.4f}")

            if epoch % self.checkpoint_every == 0:
                self.save_checkpoint(tag='latest')

        print("\nTraining complete.")

def main():
    device = "cuda:0"

    shape_meta = {
        'obs': {
            'point_cloud': {'shape': [1024, 6],    'type': 'point_cloud'},
            'agent_pos':   {'shape': [9],           'type': 'low_dim'},
            'rgb_image':   {'shape': [128, 128, 3], 'type': 'rgb'},
        },
        'action': {'shape': [4]},
    }

    train_dataset = MetaworldDataset(
        zarr_path='data/metaworld_all_tasks_expert_augmented.zarr',
        horizon=16,
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.02,
        max_train_episodes=50,
        max_val_episodes=10,
        latent_update_interval=3,
        randomize_update_interval=True,
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
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
    )

    trainer = EndToEndTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        latent_lr=3e-4,
        policy_lr=1e-4,
        weight_decay=1e-6,
        batch_size=32,
        num_workers=8,
        gradient_accumulate_every=4,
        # EMA
        use_ema=True,
        ema_update_after_step=0,
        ema_inv_gamma=1.0,
        ema_power=0.75,
        ema_min_value=0.0,
        ema_max_value=0.9999,
        # Schedule
        lr_scheduler_type="cosine",
        lr_warmup_steps=500,
        num_epochs=1000,               
        pre_alignment_epochs=30,       # meaningful alignment 
        warmup_epochs=50,              
        # Logging
        log_dir='./logs/e2e_vla',
        checkpoint_every=100,
        val_every=5,                   
        sample_every=20,
        device=device,
    )

    trainer.train(resume_from=None)

if __name__ == "__main__":
    main()