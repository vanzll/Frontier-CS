from typing import Tuple, Dict, Any, Optional
import torch
import numpy as np
from collections import defaultdict

class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm. Called once before training.
        
        Using Group Relative Policy Optimization (GRPO) configuration.
        """
        self.config = {
            "gamma": 1.0,           # No discounting for sparse goal tasks
            "clip_ratio": 0.2,       # Standard PPO clip
            "use_kl_loss": False,    # GRPO handles KL implicitly or via reward penalty if needed
            "kl_loss_coef": 0.0,
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
        gamma: float = 1.0,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Group Relative Policy Optimization (GRPO).
        Normalizes the cumulative returns within each episode group.
        """
        # Calculate episode return (sum of token rewards, assuming sparse reward at end)
        # token_level_rewards: (batch, seq_len)
        episode_returns = torch.sum(token_level_rewards, dim=1)  # (batch,)

        advantages = torch.zeros_like(token_level_rewards)
        
        # Get unique episode groups
        unique_episodes = np.unique(episode_index)
        
        for ep_id in unique_episodes:
            # Identify trajectories belonging to this episode prompt
            group_mask = (episode_index == ep_id)
            group_indices = torch.tensor(np.where(group_mask)[0], device=token_level_rewards.device)
            
            # Extract returns for this group
            group_returns = episode_returns[group_indices]
            
            # Compute GRPO advantage: (R - mean(R)) / std(R)
            if len(group_returns) > 1:
                mean = group_returns.mean()
                std = group_returns.std(unbiased=True)
                # Stabilize division
                std = torch.clamp(std, min=1e-8)
                group_adv = (group_returns - mean) / std
            else:
                # Undefined variance for single sample, set advantage to 0
                group_adv = torch.zeros_like(group_returns)
            
            # Assign calculated advantages to the corresponding rows
            # Broadcast scalar advantage to the sequence length dimension
            advantages[group_indices] = group_adv.unsqueeze(1)

        # Mask advantages for padding/invalid tokens
        advantages = advantages * response_mask
        
        # Expand returns to match shape (for consistency/logging)
        returns = episode_returns.unsqueeze(1).expand_as(token_level_rewards) * response_mask
        
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
        Compute PPO policy gradient loss with clipping.
        """
        # Probability ratio: pi_new / pi_old
        ratio = torch.exp(log_prob - old_log_prob)
        
        # Surrogate objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        
        # Maximizing objective is equivalent to minimizing negative loss
        loss_element = -torch.min(surr1, surr2)
        
        # Average loss over valid tokens
        valid_tokens_count = response_mask.sum()
        
        if valid_tokens_count > 0:
            loss = (loss_element * response_mask).sum() / valid_tokens_count
        else:
            loss = torch.tensor(0.0, device=log_prob.device, requires_grad=True)
            
        # Metrics for logging
        with torch.no_grad():
            # Calculate clipping fraction
            clipped_mask = (ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)
            clip_frac = (clipped_mask.float() * response_mask).sum() / (valid_tokens_count + 1e-8)
            
            # Approximate KL divergence
            approx_kl = ((old_log_prob - log_prob) * response_mask).sum() / (valid_tokens_count + 1e-8)
            
            # Mean advantage
            mean_adv = (advantages * response_mask).sum() / (valid_tokens_count + 1e-8)

        metrics = {
            "clip_frac": clip_frac.item(),
            "approx_kl": approx_kl.item(),
            "mean_advantage": mean_adv.item()
        }
        
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
        Distribute episode reward. For sparse tasks like ALFWorld,
        assign the full reward to the final step.
        """
        rewards = np.zeros(trajectory_length, dtype=np.float32)
        if trajectory_length > 0:
            rewards[-1] = episode_reward
        return rewards