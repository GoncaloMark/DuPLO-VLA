import numpy as np
import torch
from torch.utils.data import Sampler


class EpisodeBalancedBatchSampler(Sampler):
    """
    Yields batches of (episodes_per_batch × samples_per_episode) indices,
    where each batch is guaranteed to have multiple samples from each of
    several episodes.

    Args:
        dataset: MetaworldDataset. Must expose sampler.indices (the list of
            (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
            tuples) and replay_buffer['episode_id'].
        episodes_per_batch: how many distinct episodes per batch (N).
        samples_per_episode: how many sequence starts to draw per episode (K).
            Effective batch size is N * K.
        num_iters: number of batches per epoch. Typically set to
            len(dataset) // (N * K) to match roughly 1 epoch of data, but
            can be increased for more gradient updates per epoch.
        task_balance: if True, also stratify episodes across tasks so each
            batch has episodes from multiple tasks (critical for multi-task).
        seed: RNG seed for reproducibility.
    """

    def __init__(self, dataset, episodes_per_batch=32, samples_per_episode=8,
                 num_iters=None, task_balance=True, seed=42):
        self.dataset = dataset
        self.N = episodes_per_batch
        self.K = samples_per_episode
        self.task_balance = task_balance
        self.rng = np.random.RandomState(seed)

        # Build: for each episode_id, list of dataset indices (sampler indices)
        # that belong to that episode.
        episode_id_per_frame = np.asarray(
            [int(x) for x in dataset.replay_buffer['episode_id'][:]]
        )
        task_name_per_frame = np.asarray(
            [str(x) for x in dataset.replay_buffer['task_name'][:]]
        )

        # dataset.sampler.indices is a list/array of
        # (buffer_start, buffer_end, sample_start, sample_end).
        # The first column is the frame index where the sequence starts.
        sampler_indices = dataset.sampler.indices
        if isinstance(sampler_indices, np.ndarray):
            buffer_starts = sampler_indices[:, 0].astype(np.int64)
        else:
            buffer_starts = np.array([row[0] for row in sampler_indices], dtype=np.int64)

        # For each dataset index i, the episode it belongs to is the
        # episode of the frame at buffer_starts[i].
        frame_eps = episode_id_per_frame[buffer_starts]
        frame_tasks = task_name_per_frame[buffer_starts]

        # Group dataset indices by episode
        self.episode_to_indices = {}
        self.episode_to_task = {}
        for ds_idx, (ep, task) in enumerate(zip(frame_eps, frame_tasks)):
            ep = int(ep)
            self.episode_to_indices.setdefault(ep, []).append(ds_idx)
            self.episode_to_task[ep] = task

        self.all_episodes = np.array(sorted(self.episode_to_indices.keys()))

        # For task-balancing: group episodes by task
        self.task_to_episodes = {}
        for ep in self.all_episodes:
            task = self.episode_to_task[ep]
            self.task_to_episodes.setdefault(task, []).append(ep)

        self.all_tasks = sorted(self.task_to_episodes.keys())

        # Default num_iters: roughly one pass through the dataset
        if num_iters is None:
            num_iters = len(dataset) // (self.N * self.K)
        self.num_iters = num_iters

        print(f"[EpisodeBalancedBatchSampler] "
              f"{len(self.all_episodes)} episodes, "
              f"{len(self.all_tasks)} tasks, "
              f"{self.N} eps/batch × {self.K} samples/ep = "
              f"batch size {self.N * self.K}, "
              f"{self.num_iters} batches/epoch")

    def _sample_episodes(self):
        """
        Pick N episodes for a batch. If task_balance, spread across tasks.
        """
        if self.task_balance and len(self.all_tasks) > 1:
            # Distribute N episodes as evenly as possible across tasks
            per_task = self.N // len(self.all_tasks)
            remainder = self.N % len(self.all_tasks)
            chosen = []
            # Shuffle task order so the remainder isn't always given to the
            # same alphabetically-first tasks
            task_order = self.rng.permutation(self.all_tasks)
            for i, task in enumerate(task_order):
                k = per_task + (1 if i < remainder else 0)
                pool = self.task_to_episodes[task]
                # Sample without replacement if possible, with if not
                if k <= len(pool):
                    chosen.extend(self.rng.choice(pool, size=k, replace=False))
                else:
                    chosen.extend(self.rng.choice(pool, size=k, replace=True))
            return chosen
        else:
            replace = self.N > len(self.all_episodes)
            return self.rng.choice(self.all_episodes, size=self.N, replace=replace).tolist()

    def _sample_frames(self, episode):
        """
        Pick K sequence-start indices from within a given episode.
        Uses replacement if an episode has fewer than K valid starts (rare).
        """
        pool = self.episode_to_indices[int(episode)]
        if len(pool) >= self.K:
            return self.rng.choice(pool, size=self.K, replace=False).tolist()
        else:
            return self.rng.choice(pool, size=self.K, replace=True).tolist()

    def __iter__(self):
        for _ in range(self.num_iters):
            batch = []
            for ep in self._sample_episodes():
                batch.extend(self._sample_frames(ep))
            # Shuffle within batch so positions aren't grouped by episode
            # (matters if your model or loss ever looks at batch order)
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_iters