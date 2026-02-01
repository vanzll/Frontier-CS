"""
GRPO (Group Relative Policy Optimization) baseline implementation.

This is a reference implementation that achieves ~70% success rate on ALFWorld.
Based on DeepSeek-R1 paper's GRPO algorithm.
"""
import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional


class Solution:
    """
    GRPO: Group Relative Policy Optimization

    Key idea: Normalize rewards within episode groups to compute relative advantages.
    This is a critic-free algorithm that uses group statistics as baseline.
    """

    def solve(self, spec_path: str = None) -> "Solution":
        """Initialize GRPO algorithm with default hyperparameters."""
        self.config = {
            "gamma": 0.95,
            "clip_ratio": 0.2,
            "use_kl_loss": True,
            "kl_loss_coef": 0.01,
            "norm_adv_by_std": True,  # Whether to normalize by std (original GRPO)
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
        GRPO advantage computation: normalize rewards within episode groups.

        For each trajectory, the advantage is:
            A = (R - mean(R_group)) / (std(R_group) + epsilon)

        where R_group contains all rewards from trajectories starting from
        the same initial state (same episode_index).
        """
        # Sum token-level rewards to get episode reward
        scores = token_level_rewards.sum(dim=-1)  # (batch,)

        # Group scores by episode
        id2score = defaultdict(list)
        for i in range(len(scores)):
            id2score[episode_index[i]].append(scores[i].item())

        # Compute group statistics
        id2mean = {}
        id2std = {}
        for idx, score_list in id2score.items():
            if len(score_list) == 1:
                # Single trajectory in group - no normalization possible
                id2mean[idx] = 0.0
                id2std[idx] = 1.0
            else:
                id2mean[idx] = np.mean(score_list)
                id2std[idx] = np.std(score_list) + epsilon

        # Normalize scores within groups
        normalized = torch.zeros_like(scores)
        for i in range(len(scores)):
            idx = episode_index[i]
            if self.config.get("norm_adv_by_std", True):
                normalized[i] = (scores[i] - id2mean[idx]) / id2std[idx]
            else:
                # Dr.GRPO variant: don't normalize by std
                normalized[i] = scores[i] - id2mean[idx]

        # Expand to token level
        advantages = normalized.unsqueeze(-1) * response_mask
        returns = advantages  # For GRPO, returns == advantages

        return advantages, returns

    def compute_policy_loss(
        self,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        clip_ratio: float = 0.2,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        PPO-style clipped policy loss.

        Loss = max(-A * ratio, -A * clip(ratio, 1-eps, 1+eps))

        where ratio = exp(log_prob - old_log_prob)
        """
        # Compute importance sampling ratio
        ratio = torch.exp(log_prob - old_log_prob)

        # Clipped surrogate objective
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1 - clip_ratio, 1 + clip_ratio
        )

        # Take the maximum (more conservative)
        loss_per_token = torch.max(pg_loss1, pg_loss2)

        # Aggregate loss (token-mean)
        loss = (loss_per_token * response_mask).sum() / (response_mask.sum() + 1e-8)

        # Compute metrics for logging
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
            "ratio_mean": ratio.mean().item(),
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
        Sparse reward assignment: only the final step gets the reward.

        This is the default for GRPO - no intermediate rewards.
        """
        rewards = np.zeros(trajectory_length)
        rewards[-1] = episode_reward
        return rewards
