import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        """Initialize the algorithm with hyperparameters."""
        self.config = {
            "gamma": 0.99,
            "clip_ratio": 0.2,
            "use_kl_loss": False,
            "kl_loss_coef": 0.001,
            "eps": 1e-8,
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
        gamma: float = 0.99,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GiGPO-style hierarchical grouping.
        Combines episode-level and step-level normalization.
        """
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device
        dtype = token_level_rewards.dtype
        
        # Compute sequence-level rewards (sum over tokens)
        sequence_rewards = (token_level_rewards * response_mask).sum(dim=-1)
        
        # Initialize outputs
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        
        # Step 1: Episode-level grouping (GRPO-style)
        unique_episodes = np.unique(episode_index)
        episode_advantages = torch.zeros(batch_size, device=device, dtype=dtype)
        
        for ep_idx in unique_episodes:
            mask = episode_index == ep_idx
            group_indices = np.where(mask)[0]
            group_rewards = sequence_rewards[group_indices]
            
            if len(group_rewards) > 1:
                mean_r = group_rewards.mean()
                std_r = group_rewards.std() + self.config["eps"]
                normalized = (group_rewards - mean_r) / std_r
            else:
                normalized = torch.zeros_like(group_rewards)
            
            for i, idx in enumerate(group_indices):
                episode_advantages[idx] = normalized[i]
        
        # Step 2: Step-level grouping (GiGPO-style) if anchor observations provided
        if anchor_observations is not None:
            episode_advantages = self._step_level_grouping(
                episode_advantages,
                sequence_rewards,
                anchor_observations,
                episode_index
            )
        
        # Broadcast advantages to token level
        for i in range(batch_size):
            token_count = response_mask[i].sum()
            if token_count > 0:
                advantages[i] = episode_advantages[i] * response_mask[i]
                returns[i] = sequence_rewards[i] * response_mask[i]
        
        return advantages, returns

    def _step_level_grouping(
        self,
        episode_advantages: torch.Tensor,
        sequence_rewards: torch.Tensor,
        anchor_observations: np.ndarray,
        episode_index: np.ndarray
    ) -> torch.Tensor:
        """Apply GiGPO step-level advantage refinement."""
        batch_size = episode_advantages.shape[0]
        device = episode_advantages.device
        dtype = episode_advantages.dtype
        
        # Create observation-based groups within episodes
        obs_groups = defaultdict(list)
        for i in range(batch_size):
            ep_idx = int(episode_index[i])
            if anchor_observations[i] is not None:
                obs_hash = hash(str(anchor_observations[i]))
            else:
                obs_hash = i
            obs_groups[(ep_idx, obs_hash)].append(i)
        
        refined_advantages = episode_advantages.clone()
        
        for group_key, indices in obs_groups.items():
            if len(indices) > 1:
                group_rewards = sequence_rewards[indices]
                mean_r = group_rewards.mean()
                std_r = group_rewards.std() + self.config["eps"]
                
                for idx in indices:
                    normalized = (sequence_rewards[idx] - mean_r) / std_r
                    refined_advantages[idx] = normalized
        
        return refined_advantages

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
        Compute PPO-style clipped policy loss with sequence-level weighting.
        """
        clip_ratio = self.config.get("clip_ratio", clip_ratio)
        
        # Importance sampling ratio
        log_ratio = log_prob - old_log_prob
        ratio = torch.exp(log_ratio)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        
        # Negative because we minimize
        policy_loss = -torch.min(surr1, surr2)
        
        # Masked mean
        valid_tokens = response_mask.sum() + self.config["eps"]
        loss = (policy_loss * response_mask).sum() / valid_tokens
        
        # Compute diagnostic metrics
        with torch.no_grad():
            clipped = (torch.abs(ratio - 1.0) > clip_ratio).float()
            clip_frac = (clipped * response_mask).sum() / valid_tokens
            
            approx_kl = (log_ratio * response_mask).sum() / valid_tokens
            entropy_approx = -(log_prob * response_mask).sum() / valid_tokens
        
        metrics = {
            "clip_frac": clip_frac.item(),
            "approx_kl": approx_kl.item(),
            "entropy": entropy_approx.item(),
            "ratio_mean": (ratio * response_mask).sum().item() / valid_tokens.item(),
            "ratio_max": (ratio * response_mask).max().item(),
        }
        
        # Optional KL regularization
        if self.config.get("use_kl_loss", False):
            ref_log_prob = kwargs.get("ref_log_prob", old_log_prob)
            kl_div = log_prob - ref_log_prob
            kl_loss = (kl_div * response_mask).sum() / valid_tokens
            loss = loss + self.config["kl_loss_coef"] * kl_loss
            metrics["kl_loss"] = kl_loss.item()
        
        return loss, metrics

    def assign_step_rewards(
        self,
        episode_reward: float,
        trajectory_length: int,
        step_observations: list,
        step_actions: list,
        **kwargs
    ) -> np.ndarray:
        """
        Distribute episode reward with exponentially increasing weights.
        Later steps that complete the task get more credit.
        """
        if trajectory_length == 0:
            return np.array([])
        
        step_rewards = np.zeros(trajectory_length)
        
        if abs(episode_reward) < 1e-8:
            return step_rewards
        
        gamma = self.config.get("gamma", 0.99)
        
        if episode_reward > 0:
            # Success: reward weighted toward final steps
            weights = np.array([
                gamma ** (trajectory_length - 1 - i) 
                for i in range(trajectory_length)
            ])
            weights = weights / (weights.sum() + 1e-8)
            step_rewards = weights * episode_reward
        else:
            # Failure: small penalty at last step
            step_rewards[-1] = episode_reward * 0.1
        
        return step_rewards