from typing import Dict
import torch
import numpy as np
import copy
from tqdm import tqdm
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

import zarr


class MetaworldDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            max_val_episodes=None,
            latent_update_interval=3,
            randomize_update_interval=True,
            use_precomputed_vlm=False,
            aws_key=None,
            aws_secret=None,
            ):
        super().__init__()

        # Small arrays go into RAM. VLM hidden states stay on disk.
        ram_keys = ['state', 'action', 'point_cloud', 'instruction',
                    'task_name', 'episode_id', 'img']

        self.use_precomputed_vlm = use_precomputed_vlm
        self._vlm_hs = None
        self._vlm_sl = None

        # -----------------------------------------------------
        # 1. Load small arrays into RAM via ReplayBuffer
        # -----------------------------------------------------
        if zarr_path.startswith("s3://"):
            zarr_root = zarr.open_group(
                store=zarr_path.replace("s3://", ""),
                mode='r',
                storage_options={
                    'key': aws_key,
                    'secret': aws_secret,
                    'client_kwargs': {'region_name': "eu-west-2"}
                },
            )
            self.replay_buffer = ReplayBuffer(root=zarr_root)
        else:
            self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=ram_keys)

        # -----------------------------------------------------
        # 2. Open VLM arrays lazily on disk (not loaded into RAM)
        # -----------------------------------------------------
        if use_precomputed_vlm:
            vlm_zarr = zarr.open(zarr_path, mode='r')
            self._vlm_hs = vlm_zarr['data/vlm_hidden_states']  # lazy, reads on access
            self._vlm_sl = vlm_zarr['data/vlm_seq_len']
            print(f"VLM features on disk (lazy): hs={self._vlm_hs.shape}, sl={self._vlm_sl.shape}")
            print(f"  RAM saved: ~{self._vlm_hs.nbytes / 1e9:.1f} GB kept on NVMe instead of RAM")

        # -----------------------------------------------------
        # 3. Setup Sampler & Masks
        # -----------------------------------------------------
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
        if max_val_episodes is not None:
            val_mask = downsample_mask(mask=val_mask, max_n=max_val_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.latent_update_interval = latent_update_interval
        self.randomize_update_interval = randomize_update_interval
        self.rng = np.random.RandomState(seed)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.train_mask = self.val_mask
        val_set.randomize_update_interval = False
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def get_sample_weights(self) -> torch.Tensor:
        """
        Inverse-frequency weights per sample for WeightedRandomSampler.
        Gives each task equal expected representation regardless of episode count.
        """
        task_names_flat = np.array([str(x) for x in self.replay_buffer['task_name'][:]])
        max_valid_idx = len(task_names_flat) - 1

        if isinstance(self.sampler.indices, np.ndarray):
            valid_mask = self.sampler.indices[:, 1] <= max_valid_idx
            self.sampler.indices = self.sampler.indices[valid_mask]
        else:
            self.sampler.indices = [row for row in self.sampler.indices if row[1] <= max_valid_idx]

        indices = self.sampler.indices
        sample_tasks = [task_names_flat[row[0]] for row in indices]

        counts: dict = {}
        for t in sample_tasks:
            counts[t] = counts.get(t, 0) + 1
        weights = torch.tensor([1.0 / counts[t] for t in sample_tasks], dtype=torch.float)
        return weights

    def _extract_scalar_string(self, value) -> str:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, list):
            value = value[0]
        return str(value)

    def _load_vlm_sequence(self, idx):
        """
        Load VLM hidden states lazily from disk for a single sample,
        replicating the same padding logic as SequenceSampler.sample_sequence.

        Returns:
            vlm_hs: (horizon, num_layers, max_seq_len, hidden_dim) float16
            vlm_sl: (horizon,) int32
        """
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.sampler.indices[idx]

        # Read only the needed frames from disk
        hs_slice = self._vlm_hs[buffer_start_idx:buffer_end_idx]  # (n, layers, seq, dim)
        sl_slice = self._vlm_sl[buffer_start_idx:buffer_end_idx]  # (n,)

        # Apply same padding as SequenceSampler
        if (sample_start_idx > 0) or (sample_end_idx < self.horizon):
            hs_padded = np.zeros(
                (self.horizon,) + hs_slice.shape[1:], dtype=hs_slice.dtype)
            sl_padded = np.zeros(self.horizon, dtype=sl_slice.dtype)

            if sample_start_idx > 0:
                hs_padded[:sample_start_idx] = hs_slice[0]
                sl_padded[:sample_start_idx] = sl_slice[0]
            if sample_end_idx < self.horizon:
                hs_padded[sample_end_idx:] = hs_slice[-1]
                sl_padded[sample_end_idx:] = sl_slice[-1]

            hs_padded[sample_start_idx:sample_end_idx] = hs_slice
            sl_padded[sample_start_idx:sample_end_idx] = sl_slice
            return hs_padded, sl_padded
        else:
            return np.asarray(hs_slice), np.asarray(sl_slice)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32)
        point_cloud = sample['point_cloud'].astype(np.float32)
        rgb_image = sample['img'].astype(np.uint8)

        instruction = self._extract_scalar_string(sample['instruction'])
        task_name = self._extract_scalar_string(sample['task_name'])
        episode_id = int(self._extract_scalar_string(sample['episode_id']))

        obs = {
            'point_cloud': point_cloud,
            'agent_pos': agent_pos,
            'instruction': instruction,
            'task_name': task_name,
            'rgb_image': rgb_image,
            'episode_id': episode_id,
        }

        return {'obs': obs, 'action': sample['action'].astype(np.float32)}

    def _create_latent_update_schedule(self, horizon):
        if self.randomize_update_interval:
            min_interval = max(1, self.latent_update_interval - 1)
            max_interval = self.latent_update_interval + 1
            interval = self.rng.randint(min_interval, max_interval + 1)
        else:
            interval = self.latent_update_interval

        latent_update_mask = np.zeros(horizon, dtype=bool)
        latent_update_mask[::interval] = True
        latent_group_id = np.arange(horizon) // interval

        return latent_update_mask, latent_group_id

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data   = self._sample_to_data(sample)

        if self.use_precomputed_vlm:
            vlm_hs, vlm_sl = self._load_vlm_sequence(idx)
            data['obs']['vlm_hidden_states'] = vlm_hs.astype(np.float16)
            data['obs']['vlm_seq_len'] = vlm_sl.astype(np.int32)

        latent_update_mask, latent_group_id = self._create_latent_update_schedule(self.horizon)

        torch_data: Dict = {}
        for k, v in data.items():
            if isinstance(v, dict):
                torch_data[k] = {}
                for kk, vv in v.items():
                    torch_data[k][kk] = torch.from_numpy(vv) if isinstance(vv, np.ndarray) else vv
            else:
                torch_data[k] = torch.from_numpy(v) if isinstance(v, np.ndarray) else v

        torch_data['latent_update_mask'] = torch.from_numpy(latent_update_mask)
        torch_data['latent_group_id'] = torch.from_numpy(latent_group_id)

        return torch_data
