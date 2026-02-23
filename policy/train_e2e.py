import itertools
import signal
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import copy
import numpy as np
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
            contrastive_weight=0.01,   
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
        if x.requires_grad and scale != 1.0:
            def hook(grad):
                return grad * scale
            x.register_hook(hook)
        return x

    def forward(
        self,
        obs_dict,
        instruction_text,
        task_names,
        episode_ids,
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

        update_timesteps = latent_update_mask[0].nonzero(as_tuple=True)[0]
        num_updates      = len(update_timesteps)

        if num_updates > 0:
            update_images = torch.stack(
                [obs_dict['rgb_image'][:, t] for t in update_timesteps], dim=1
            )
            update_images_flat = update_images.reshape(B * num_updates, *update_images.shape[2:])

            update_instructions = [
                instruction_text[i]
                for i in range(B)
                for _ in range(num_updates)
            ]
            update_task_names = [
                task_names[i]
                for i in range(B)
                for _ in range(num_updates)
            ] if task_names is not None else None

            update_episode_ids = [
                episode_ids[i]
                for i in range(B)
                for _ in range(num_updates)
            ] if episode_ids is not None else None

            plan_output = self.planner.plan(
                image=update_images_flat,
                instruction=update_instructions,
                task_names=update_task_names,
                episode_ids=update_episode_ids,
                training=self.training,
                return_reconstruction_loss=True,
            )

            computed_latents      = plan_output['latent'].view(B, num_updates, -1)
            clamped_group_ids     = latent_group_id.clamp(0, num_updates - 1)
            batch_indices         = torch.arange(B, device=device)[:, None].expand(B, T)
            task_latents_expanded = computed_latents[batch_indices, clamped_group_ids]

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
            if k not in ('instruction', 'rgb_image', 'task_name', 'episode_id')
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

def _to_list(x):
    """Safely convert tensor or list of task_names / episode_ids to plain list."""
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return list(x)


@torch.no_grad()
def _collect_latents(model, loader, device, num_batches=30):
    """
    Collect (latents, task_names, episode_ids) using only the first timestep
    image per sample — avoids the full temporal forward pass and is fast.

    Returns:
        latents:     np.ndarray  (N, latent_dim)
        task_names:  List[str]   (N,)
        episode_ids: List[int]   (N,)
    """
    model.eval()
    all_latents, all_tasks, all_eps = [], [], []

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        images       = batch['obs']['rgb_image'][:, 0].to(device)   # (B, H, W, C)
        instructions = _to_list(batch['obs']['instruction'])
        task_names   = _to_list(batch['obs']['task_name'])
        episode_ids  = _to_list(batch['obs']['episode_id'])

        plan = model.planner.plan(
            image=images,
            instruction=instructions,
            task_names=task_names,
            episode_ids=episode_ids,
            training=False,
            return_reconstruction_loss=False,
        )

        all_latents.append(plan['latent'].cpu().float().numpy())
        all_tasks.extend(task_names)
        all_eps.extend(episode_ids)

    return np.concatenate(all_latents, axis=0), all_tasks, all_eps


@torch.no_grad()
def check_latent_health(model, loader, device, writer=None, global_step=0, num_batches=20):
    """
    Checks for the three most common early failure modes:
        Collapse  — pairwise cosine > 0.9  (all latents identical)
        Dead dims — many dims with std < 0.01
        Explosion — norms growing unbounded
    Takes ~1 min on val set.
    """
    latents, _, _ = _collect_latents(model, loader, device, num_batches)
    t = torch.from_numpy(latents)

    mean_norm   = t.norm(dim=-1).mean().item()
    std_per_dim = t.std(dim=0)
    dead_dims   = (std_per_dim < 0.01).sum().item()
    mean_std    = std_per_dim.mean().item()

    # Subsample to keep NxN manageable
    sub   = t[:256] if len(t) > 256 else t
    sub_n = F.normalize(sub, dim=-1)
    cos   = (sub_n @ sub_n.T)
    cos.fill_diagonal_(0.0)
    pairwise_cos = cos.mean().item()

    results = {
        'mean_norm':       mean_norm,
        'mean_std':        mean_std,
        'dead_dims':       dead_dims,
        'pairwise_cosine': pairwise_cos,
    }

    print(f"\n── Latent Health ──────────────────────────────────")
    print(f"  Mean norm:          {mean_norm:8.3f}   (expect 1-10)")
    print(f"  Mean std / dim:     {mean_std:8.4f}   (collapse if <0.01)")
    print(f"  Dead dims:          {dead_dims:8d} / {latents.shape[1]}  (expect 0)")
    print(f"  Mean pairwise cos:  {pairwise_cos:8.3f}   (collapse if >0.9)")
    print(f"────────────────────────────────────────────────────\n")

    if writer:
        for k, v in results.items():
            writer.add_scalar(f'Diagnostics/health_{k}', v, global_step)

    return results

@torch.no_grad()
def within_task_sensitivity(model, loader, device, writer=None, global_step=0, num_batches=30):
    """
    Variance ratio test: does the latent vary within a task?

    within_var / across_var < 0.3  - encoder reads the image              
    within_var / across_var > 0.5  - encoder ignores image, only text     
    """
    latents, task_names, _ = _collect_latents(model, loader, device, num_batches)
    t = torch.from_numpy(latents)

    task_latents: dict = {}
    for lat, task in zip(t, task_names):
        task_latents.setdefault(task, []).append(lat)

    within_vars, task_means = [], []
    for task, lats in task_latents.items():
        if len(lats) < 2:
            continue
        stacked = torch.stack(lats)
        within_vars.append(stacked.var(dim=0).mean().item())
        task_means.append(stacked.mean(dim=0))

    if not within_vars or len(task_means) < 2:
        print("  [sensitivity] Not enough task diversity — try more batches or tasks")
        return {}

    across_var  = torch.stack(task_means).var(dim=0).mean().item()
    mean_within = float(np.mean(within_vars))
    ratio       = mean_within / (across_var + 1e-8)
    status      = '✓ reads image' if ratio < 0.3 else '✗ ignoring image'

    results = {'within_var': mean_within, 'across_var': across_var, 'ratio': ratio}

    print(f"\n--- Within-Task Visual Sensitivity ------------------------------")
    print(f"  Mean within-task var:  {mean_within:.4f}")
    print(f"  Across-task var:       {across_var:.4f}")
    print(f"  Ratio (within/across): {ratio:.3f}   → {status}")
    print(f"-----------------------------------------------------\n")

    if writer:
        for k, v in results.items():
            writer.add_scalar(f'Diagnostics/sensitivity_{k}', v, global_step)

    return results


@torch.no_grad()
def visualize_attention(model, loader, device, writer=None, global_step=0):
    """
    Shows what each Q-Pooler query attends to: image tokens vs text tokens.
    Works with the basic QPooler — uses positional heuristic since we don't
    have explicit token-type ids (Qwen3-VL packs image patches first).

    Healthy: diverse split, some queries image-dominant, some text-dominant.
    Unhealthy: all queries identical → QPooler hasn't specialised.
    """
    model.eval()

    # Grab a single sample to keep it cheap
    batch = next(iter(loader))
    images       = batch['obs']['rgb_image'][:1, 0].to(device)
    instructions = _to_list(batch['obs']['instruction'])[:1]
    task_names   = _to_list(batch['obs']['task_name'])[:1]
    episode_ids  = _to_list(batch['obs']['episode_id'])[:1]

    plan = model.planner.plan(
        image=images,
        instruction=instructions,
        task_names=task_names,
        episode_ids=episode_ids,
        training=False,
        return_attention_weights=True,
        return_reconstruction_loss=False,
    )

    enc_out = plan.get('encoder_output', {})
    attn    = enc_out.get('attention_weights', None)  # (1, heads, queries, seq_len)

    if attn is None:
        print("  [attention] No weights returned — ensure return_attention_weights "
              "is plumbed through plan() → task_encoder.forward()")
        return

    # Mean over heads, first sample → (num_queries, seq_len)
    attn    = attn[0].mean(dim=0).cpu()   # (Q, seq_len)
    seq_len = attn.shape[-1]

    # Heuristic split: Qwen3-VL places image patch tokens first.
    # 128×128 px → ceil(128/14)^2 = 10^2 = 100 patches (approx).
    # Adjust img_frac if you know your exact VLM token layout.
    img_frac = 0.40
    img_end  = max(1, int(seq_len * img_frac))
    img_attn  = attn[:, :img_end].sum(dim=-1)
    text_attn = attn[:, img_end:].sum(dim=-1)

    img_dom = (img_attn > text_attn).sum().item()
    print(f"\n--- Q-Pooler Attention  (seq_len={seq_len}, img_end≈{img_end}) ------")
    print(f"  {'Query':>6}  {'→ Image':>9}  {'→ Text':>9}  Dominant")
    for q in range(attn.shape[0]):
        dom = 'IMAGE' if img_attn[q] > text_attn[q] else 'TEXT '
        print(f"  {q:>6}  {img_attn[q].item():>9.3f}  {text_attn[q].item():>9.3f}  {dom}")
    print(f"  → {img_dom}/{attn.shape[0]} queries image-dominant")
    print(f"---------------------------------------------------------\n")

    if writer:
        for q in range(attn.shape[0]):
            frac = img_attn[q].item() / (img_attn[q].item() + text_attn[q].item() + 1e-8)
            writer.add_scalar(f'Diagnostics/attn_img_frac/q{q:02d}', frac, global_step)


def run_diagnostics(model, val_loader, device,
                    writer, global_step, epoch):
    """
    Runs all four probes and returns a go/no-go bool.
    Hard-stops training only when called at the end of pre-alignment.
    """
    print(f"\n{'='*62}")
    print(f"  DIAGNOSTICS — epoch {epoch}  (global step {global_step})")
    print(f"{'='*62}")

    health      = check_latent_health(model, val_loader, device, writer, global_step)
    sensitivity = within_task_sensitivity(model, val_loader, device, writer, global_step)
    visualize_attention(model, val_loader, device, writer, global_step)

    print(f"---- Go / No-Go -----------------------------------------")
    go, checks = True, []

    if health.get('pairwise_cosine', 0) > 0.9:
        checks.append('FAIL: latent collapse — pairwise cos > 0.9')
        go = False
    else:
        checks.append(f'pairwise cos = {health.get("pairwise_cosine", 0):.3f}')

    if health.get('dead_dims', 0) > 100:
        checks.append(f'FAIL: {health["dead_dims"]} dead dimensions')
        go = False
    else:
        checks.append(f'dead dims = {health.get("dead_dims", 0)}')

    if sensitivity:
        r = sensitivity.get('ratio', 0)
        if r > 0.5:
            checks.append(f'FAIL: within/across ratio {r:.2f} > 0.5 (ignoring image)')
            go = False
        else:
            checks.append(f'within/across ratio = {r:.2f}')

    for c in checks:
        print(f'  {c}')

    verdict = 'Proceed to DP3 warmup' if go else 'Fix encoder before DP3'
    print(f'\n  Verdict: {verdict}')
    print(f"------------------------------------------------------------\n")

    if writer:
        writer.add_scalar('Diagnostics/go_nogo', float(go), global_step)

    return go

class EndToEndTrainer:
    def __init__(
        self,
        model: EndToEndRobotPolicy,
        train_dataset: MetaworldDataset,
        val_dataset: MetaworldDataset,
        latent_lr=3e-4,
        policy_lr=1e-4,
        weight_decay=1e-6,
        batch_size=32,
        num_workers=8,
        gradient_accumulate_every=4,
        use_ema=True,
        ema_update_after_step=0,
        ema_inv_gamma=1.0,
        ema_power=0.75,
        ema_min_value=0.0,
        ema_max_value=0.9999,
        lr_scheduler_type="cosine",
        lr_warmup_steps=500,
        num_epochs=1000,
        pre_alignment_epochs=30,
        warmup_epochs=50,
        diagnostic_epochs=(5, 15),   # also auto-runs at pre_alignment_epochs - 1
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

        # Always include the final pre-alignment epoch in the diagnostic schedule
        self.diagnostic_epochs = set(diagnostic_epochs) | {pre_alignment_epochs - 1}

        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
        self.global_step   = 0

        self.log_dir        = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.writer         = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        self.checkpoint_dir = self.log_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.epoch           = 0
        self.best_val_loss   = float('inf')
        self.best_test_score = float('-inf')

        self.checkpoint_every = checkpoint_every
        self.val_every        = val_every
        self.sample_every     = sample_every
        self.train_sampling_batch = None

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers,
            shuffle=True, pin_memory=True, persistent_workers=False,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=num_workers,
            shuffle=False, pin_memory=True, persistent_workers=False,
        )

        normalizer = train_dataset.get_normalizer()
        normalizer.params_dict['task_latent'] = nn.ParameterDict({
            'mean':   torch.zeros(self.model.latent_dim),
            'scale':  torch.ones(self.model.latent_dim),
            'offset': torch.zeros(self.model.latent_dim),
        })
        self.model.policy.set_normalizer(normalizer)
        self.model.policy.normalizer = self.model.policy.normalizer.to(device)

        self.optimizer       = self._setup_optimizer(latent_lr, policy_lr, weight_decay)
        self._base_latent_lr = latent_lr
        self.lr_scheduler    = self._setup_lr_scheduler(
            lr_scheduler_type, lr_warmup_steps,
            (len(self.train_loader) * num_epochs) // gradient_accumulate_every,
        )

        self.use_ema   = use_ema
        self.ema_model = None
        self.ema       = None
        if use_ema:
            self._setup_ema(normalizer, device, ema_update_after_step,
                            ema_inv_gamma, ema_power, ema_min_value, ema_max_value)

        signal.signal(signal.SIGTERM, self._emergency_save)

    def _setup_ema(self, normalizer, device, update_after_step,
                   inv_gamma, power, min_value, max_value):
        vlm_ref = self.model.planner.vlm
        self.model.planner.vlm = None
        self.ema_model = copy.deepcopy(self.model)
        self.model.planner.vlm = vlm_ref

        self.ema_model.to(device)
        self.ema_model.planner.vlm = vlm_ref
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
        return optim.AdamW(
            [
                {'params': self.model.planner.task_encoder.parameters(),
                 'lr': latent_lr, 'name': 'latent_encoder'},
                {'params': self.model.policy.parameters(),
                 'lr': policy_lr, 'name': 'policy'},
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
        phase = self.get_training_phase(epoch)
        if phase == "pre_alignment":
            return 1.0
        elif phase == "warmup":
            return 0.5
        return 0.2

    def _update_latent_lr(self, epoch):
        multiplier = self._get_latent_lr_multiplier(epoch)
        self.optimizer.param_groups[0]['lr'] = self._base_latent_lr * multiplier

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
            instructions = _to_list(batch['obs']['instruction'])
            task_names   = _to_list(batch['obs']['task_name'])
            episode_ids  = _to_list(batch['obs']['episode_id'])

            if self.train_sampling_batch is None:
                self.train_sampling_batch = {
                    'obs':              obs_dict,
                    'action':           actions,
                    'instructions':     instructions,
                    'task_names':       task_names,
                    'episode_ids':      episode_ids,
                    'latent_update_mask': batch['latent_update_mask'].to(self.device),
                    'latent_group_id':    batch['latent_group_id'].to(self.device),
                }

            latent_update_mask = batch['latent_update_mask'].to(self.device)
            latent_group_id    = batch['latent_group_id'].to(self.device)

            loss, loss_dict = self.model(
                obs_dict=obs_dict,
                instruction_text=instructions,
                task_names=task_names,
                episode_ids=episode_ids,
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

                self.writer.add_scalar('Train/Total_Loss',     loss.item(),    self.global_step)
                self.writer.add_scalar('Train/DP3_Grad_Scale', dp3_grad_scale, self.global_step)
                self.writer.add_scalar('Train/LR_Encoder',
                    self.optimizer.param_groups[0]['lr'], self.global_step)
                self.writer.add_scalar('Train/LR_Policy',
                    self.optimizer.param_groups[1]['lr'], self.global_step)

                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'Loss_Components/{k}', v, self.global_step)
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v

                pbar.set_postfix({
                    'loss':   f"{loss.item():.4f}",
                    'phase':  phase,
                    'dp3':    f"{dp3_grad_scale:.2f}",
                    'step':   self.global_step,
                    'lr_enc': f"{self.optimizer.param_groups[0]['lr']:.2e}",
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
            instructions = _to_list(batch['obs']['instruction'])
            task_names   = _to_list(batch['obs']['task_name'])
            episode_ids  = _to_list(batch['obs']['episode_id'])
            latent_update_mask = batch['latent_update_mask'].to(self.device)
            latent_group_id    = batch['latent_group_id'].to(self.device)

            loss, loss_dict = policy(
                obs_dict=obs_dict,
                instruction_text=instructions,
                task_names=task_names,
                episode_ids=episode_ids,
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

        batch        = self.train_sampling_batch
        obs_dict     = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch['obs'].items()
        }
        gt_action    = batch['action'].to(self.device)
        instructions = batch['instructions']
        task_names   = batch['task_names']
        episode_ids  = batch['episode_ids']

        result = policy(
            obs_dict=obs_dict,
            instruction_text=instructions,
            task_names=task_names,
            episode_ids=episode_ids,
            actions=None,
            compute_loss=False,
            latent_update_mask=batch.get('latent_update_mask'),
            latent_group_id=batch.get('latent_group_id'),
        )

        mse = nn.functional.mse_loss(result['action_pred'], gt_action)
        self.writer.add_scalar('Analysis/Action_MSE', mse.item(), self.global_step)
        return {'train_action_mse_error': mse.item()}

    def save_checkpoint(self, tag="latest", is_best=False):
        deploy_model = self.ema_model if self.use_ema else self.model

        checkpoint = {
            "epoch":                   self.epoch,
            "global_step":             self.global_step,
            "task_encoder_state_dict": deploy_model.planner.task_encoder.state_dict(),
            "policy_state_dict":       deploy_model.policy.state_dict(),
            "optimizer_state_dict":    self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "best_val_loss":           self.best_val_loss,
            "best_test_score":         self.best_test_score,
        }

        torch.save(checkpoint, self.checkpoint_dir / f"{tag}.ckpt")

        deploy_dir = self.log_dir / "deployment"
        deploy_dir.mkdir(exist_ok=True)

        torch.save({
            "state_dict":  deploy_model.planner.task_encoder.state_dict(),
            "latent_dim":  deploy_model.latent_dim,
            "epoch":       self.epoch,
            "global_step": self.global_step,
        }, deploy_dir / f"system2_latent_encoder_{tag}.pth")

        torch.save({
            "state_dict":  deploy_model.policy.state_dict(),
            "normalizer":  deploy_model.policy.normalizer.state_dict()
                           if hasattr(deploy_model.policy, "normalizer") else None,
            "epoch":       self.epoch,
            "global_step": self.global_step,
        }, deploy_dir / f"system1_policy_{tag}.pth")

        torch.save({
            "latent_dim":  deploy_model.latent_dim,
            "n_obs_steps": deploy_model.n_obs_steps,
            "epoch":       self.epoch,
        }, deploy_dir / f"config_{tag}.pth")

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.ckpt")
            print(f"\n{'='*60}\nSaved best checkpoint\n{'='*60}\n")

        return str(self.checkpoint_dir / f"{tag}.ckpt")

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
        print(f"Diagnostics at epochs : {sorted(self.diagnostic_epochs)}")
        print(f"{'='*70}\n")

        for epoch in range(start_epoch, self.num_epochs):
            self.epoch = epoch

            train_loss, train_metrics = self.train_epoch(epoch)
            phase = train_metrics.get('phase', '?')
            dp3s  = self.get_dp3_grad_scale(epoch)
            print(f"Epoch {epoch:4d} [{phase}]  train_loss={train_loss:.4f}  dp3_scale={dp3s:.2f}")

            if epoch in self.diagnostic_epochs:
                go = run_diagnostics(
                    model=self.model,
                    val_loader=self.val_loader,
                    device=self.device,
                    writer=self.writer,
                    global_step=self.global_step,
                    epoch=epoch,
                )
                # Only hard-stop at the transition boundary, not mid-phase
                if epoch == self.pre_alignment_epochs - 1 and not go:
                    print("\n[!] Encoder failed go/no-go at end of pre-alignment.")
                    print("    Inspect the diagnostics above before committing to 1000 epochs.")
                    self.save_checkpoint(tag='pre_alignment_failed')
                    return

            if epoch % self.val_every == 0:
                val_loss, _ = self.validate()
                print(f"             val_loss={val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(tag='best', is_best=True)

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
        use_ema=True,
        ema_update_after_step=0,
        ema_inv_gamma=1.0,
        ema_power=0.75,
        ema_min_value=0.0,
        ema_max_value=0.9999,
        lr_scheduler_type="cosine",
        lr_warmup_steps=500,
        num_epochs=1000,
        pre_alignment_epochs=30,
        warmup_epochs=50,
        diagnostic_epochs=(5, 15),   # also auto-fires at epoch 29
        log_dir='./logs/e2e_vla',
        checkpoint_every=100,
        val_every=5,
        sample_every=20,
        device=device,
    )

    trainer.train(resume_from=None)


if __name__ == "__main__":
    main()
