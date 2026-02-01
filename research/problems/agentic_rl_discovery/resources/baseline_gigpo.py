"""
GiGPO (Group-in-Group Policy Optimization) baseline implementation.

This is a reference implementation that achieves ~86% success rate on ALFWorld.
Based on verl-agent paper (NeurIPS 2025).
"""
import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List
import uuid


class Solution:
    """
    GiGPO: Group-in-Group Policy Optimization

    Key innovation: Two-level grouping for fine-grained credit assignment.
    1. Episode-level groups: Like GRPO, normalize across trajectories from same prompt
    2. Step-level groups: Cluster identical/similar observations across trajectories
       to compute step-level advantages
    """

    def solve(self, spec_path: str = None) -> "Solution":
        """Initialize GiGPO algorithm with default hyperparameters."""
        self.config = {
            "gamma": 0.95,
            "clip_ratio": 0.2,
            "use_kl_loss": True,
            "kl_loss_coef": 0.01,
            "step_advantage_w": 1.0,  # Weight for step-level advantages
            "mode": "mean_std_norm",  # or "mean_norm"
            "enable_similarity": False,  # Use exact match for step grouping
            "similarity_thresh": 0.95,  # For similarity-based grouping
        }
        return self

    def compute_advantage(
        self,
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        episode_index: np.ndarray,
        trajectory_index: np.ndarray,
        step_rewards: Optional[torch.Tensor] = None,
        anchor_observations: Optional[np.ndarray] = None,
        gamma: float = 0.95,
        epsilon: float = 1e-6,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GiGPO advantage computation with two-level grouping.

        Joint advantage = episode_advantage + step_advantage_w * step_advantage
        """
        remove_std = self.config.get("mode", "mean_std_norm") == "mean_norm"

        # 1. Compute episode-level advantages (like GRPO)
        episode_advantages = self._episode_norm_reward(
            token_level_rewards, response_mask, episode_index, trajectory_index,
            epsilon, remove_std
        )

        # 2. Compute step-level advantages (GiGPO innovation)
        step_advantages = torch.zeros_like(episode_advantages)
        if step_rewards is not None and anchor_observations is not None:
            # Build step-level groups from anchor observations
            step_group_uids = self._build_step_group(
                anchor_observations, episode_index
            )
            step_advantages = self._step_norm_reward(
                step_rewards, response_mask, step_group_uids, epsilon, remove_std
            )

        # 3. Combine advantages
        step_w = self.config.get("step_advantage_w", 1.0)
        advantages = episode_advantages + step_w * step_advantages

        return advantages, advantages

    def _episode_norm_reward(
        self,
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        episode_index: np.ndarray,
        trajectory_index: np.ndarray,
        epsilon: float,
        remove_std: bool
    ) -> torch.Tensor:
        """Compute episode-level normalized rewards (GRPO-style)."""
        response_length = token_level_rewards.shape[-1]
        scores = token_level_rewards.sum(dim=-1)

        id2score = defaultdict(list)
        seen_pairs = set()

        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                if (episode_index[i], trajectory_index[i]) in seen_pairs:
                    continue
                id2score[episode_index[i]].append(scores[i].item())
                # For multi-turn: count each trajectory once
                # seen_pairs.add((episode_index[i], trajectory_index[i]))

            id2mean = {}
            id2std = {}
            for idx, score_list in id2score.items():
                if len(score_list) == 1:
                    id2mean[idx] = 0.0
                    id2std[idx] = 1.0
                else:
                    id2mean[idx] = np.mean(score_list)
                    id2std[idx] = np.std(score_list) + epsilon

            normalized = torch.zeros_like(scores)
            for i in range(bsz):
                idx = episode_index[i]
                if remove_std:
                    normalized[i] = scores[i] - id2mean[idx]
                else:
                    normalized[i] = (scores[i] - id2mean[idx]) / id2std[idx]

            episode_advantages = normalized.unsqueeze(-1).expand(-1, response_length) * response_mask

        return episode_advantages

    def _build_step_group(
        self,
        anchor_obs: np.ndarray,
        episode_index: np.ndarray
    ) -> np.ndarray:
        """
        Group observations by episode and cluster identical observations.
        Assigns unique step_group_uid to each cluster.
        """
        step_group_uids = np.empty(len(anchor_obs), dtype=object)
        unique_indices = np.unique(episode_index)

        for idx in unique_indices:
            # Get observations for this episode group
            locs = np.where(episode_index == idx)[0]
            obs_group = anchor_obs[locs]

            # Cluster identical observations
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                # Use hashable representation
                obs_key = self._to_hashable(obs)
                clusters[obs_key].append(locs[i])

            # Assign unique UIDs to each cluster
            for obs_key, original_indices in clusters.items():
                uid = str(uuid.uuid4())
                for orig_idx in original_indices:
                    step_group_uids[orig_idx] = uid

        return step_group_uids

    def _to_hashable(self, x):
        """Convert observation to hashable type for clustering."""
        if isinstance(x, str):
            return x
        elif isinstance(x, (int, float, bool)):
            return x
        elif isinstance(x, np.ndarray):
            return tuple(x.flatten())
        elif isinstance(x, (list, tuple)):
            return tuple(self._to_hashable(e) for e in x)
        else:
            return str(x)

    def _step_norm_reward(
        self,
        step_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        step_group_uids: np.ndarray,
        epsilon: float,
        remove_std: bool
    ) -> torch.Tensor:
        """Compute step-level normalized rewards."""
        response_length = response_mask.shape[-1]
        scores = step_rewards.clone()

        id2score = defaultdict(list)
        id2mean = {}
        id2std = {}

        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[step_group_uids[i]].append(scores[i].item())

            for idx, score_list in id2score.items():
                if len(score_list) == 1:
                    id2mean[idx] = np.mean(score_list)
                    id2std[idx] = 1.0
                else:
                    id2mean[idx] = np.mean(score_list)
                    id2std[idx] = np.std(score_list) + epsilon

            normalized = torch.zeros_like(scores)
            for i in range(bsz):
                idx = step_group_uids[i]
                if remove_std:
                    normalized[i] = scores[i] - id2mean[idx]
                else:
                    normalized[i] = (scores[i] - id2mean[idx]) / id2std[idx]

            step_advantages = normalized.unsqueeze(-1).expand(-1, response_length) * response_mask

        return step_advantages

    def compute_policy_loss(
        self,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        clip_ratio: float = 0.2,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """PPO-style clipped policy loss (same as GRPO)."""
        ratio = torch.exp(log_prob - old_log_prob)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1 - clip_ratio, 1 + clip_ratio
        )

        loss_per_token = torch.max(pg_loss1, pg_loss2)
        loss = (loss_per_token * response_mask).sum() / (response_mask.sum() + 1e-8)

        with torch.no_grad():
            clip_frac = (
                (pg_loss2 > pg_loss1).float() * response_mask
            ).sum() / (response_mask.sum() + 1e-8)
            approx_kl = (
                (old_log_prob - log_prob) * response_mask
            ).sum() / (response_mask.sum() + 1e-8)

        metrics = {
            "clip_frac": clip_frac.item(),
            "approx_kl": approx_kl.item(),
        }

        return loss, metrics

    def assign_step_rewards(
        self,
        episode_reward: float,
        trajectory_length: int,
        step_observations: list = None,
        step_actions: list = None,
        **kwargs
    ) -> np.ndarray:
        """
        GiGPO uses discounted returns for step rewards.

        R_t = r_t + gamma * R_{t+1}

        With sparse episode reward, this gives:
        - Final step gets full reward
        - Earlier steps get gamma^(T-t) * reward
        """
        gamma = self.config.get("gamma", 0.95)
        rewards = np.zeros(trajectory_length)

        # Compute discounted returns from the end
        running_return = episode_reward
        for t in reversed(range(trajectory_length)):
            rewards[t] = running_return
            running_return = gamma * running_return

        return rewards
