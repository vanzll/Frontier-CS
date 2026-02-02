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
            "kl_loss_coef": 0.01,
            "advantage_norm_eps": 1e-8,
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
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GiGPO-style group normalization.
        
        Key insight: Normalize advantages within groups of trajectories that share
        the same initial state (episode group) to reduce variance while maintaining
        unbiased gradient estimates.
        """
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device
        dtype = token_level_rewards.dtype
        
        gamma = self.config.get("gamma", gamma)
        eps = self.config.get("advantage_norm_eps", 1e-8)
        
        # Use step_rewards if provided, otherwise use token_level_rewards
        if step_rewards is not None:
            rewards = step_rewards.to(device=device, dtype=dtype)
        else:
            rewards = token_level_rewards
        
        # Compute discounted returns using backward dynamic programming
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(batch_size, device=device, dtype=dtype)
        
        for t in range(seq_len - 1, -1, -1):
            mask_t = response_mask[:, t].to(dtype)
            running_return = rewards[:, t] + gamma * running_return
            returns[:, t] = running_return * mask_t
        
        # Initialize advantages
        advantages = torch.zeros_like(returns)
        
        # Group normalization within episode groups (GRPO/GiGPO core)
        unique_episodes = np.unique(episode_index)
        
        for ep_idx in unique_episodes:
            group_mask = episode_index == ep_idx
            indices = np.where(group_mask)[0]
            
            if len(indices) == 0:
                continue
                
            group_returns = returns[indices]
            group_resp_mask = response_mask[indices].to(dtype)
            
            # Compute statistics over the episode group
            valid_count = group_resp_mask.sum()
            
            if valid_count > 1:
                # Compute mean return within group
                mean_val = (group_returns * group_resp_mask).sum() / valid_count
                # Compute variance within group
                centered = group_returns - mean_val
                var_val = (centered.pow(2) * group_resp_mask).sum() / valid_count
                std_val = (var_val + eps).sqrt()
                
                # Normalize: advantage = (return - baseline) / std
                advantages[indices] = centered / std_val
            else:
                # Single sample or empty: use raw returns
                advantages[indices] = group_returns
        
        # Apply response mask
        advantages = advantages * response_mask.to(dtype)
        returns = returns * response_mask.to(dtype)
        
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
        Compute PPO-style clipped policy gradient loss.
        
        The clipped objective prevents destructively large policy updates
        while still allowing significant learning when the ratio is within bounds.
        """
        clip_ratio = self.config.get("clip_ratio", clip_ratio)
        
        # Compute importance sampling ratio
        ratio = torch.exp(log_prob - old_log_prob)
        
        # Clipped ratio for conservative updates
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        
        # PPO surrogate objectives (negative for minimization)
        loss_unclipped = -advantages * ratio
        loss_clipped = -advantages * clipped_ratio
        
        # Take pessimistic bound: max of losses = min of objectives
        policy_loss = torch.max(loss_unclipped, loss_clipped)
        
        # Apply mask and compute mean
        mask = response_mask.to(policy_loss.dtype)
        mask_sum = mask.sum().clamp(min=1.0)
        loss = (policy_loss * mask).sum() / mask_sum
        
        # Compute clip fraction for monitoring
        clipped = ((ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)).to(mask.dtype)
        clip_frac = (clipped * mask).sum() / mask_sum
        
        # Optional KL penalty for additional stability
        if self.config.get("use_kl_loss", False):
            # Approximate KL divergence
            approx_kl = 0.5 * ((log_prob - old_log_prob).pow(2) * mask).sum() / mask_sum
            loss = loss + self.config.get("kl_loss_coef", 0.01) * approx_kl
        
        metrics = {
            "clip_frac": clip_frac.item(),
            "policy_loss": loss.item(),
            "ratio_mean": ((ratio * mask).sum() / mask_sum).item(),
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
        Distribute episode reward to individual steps.
        
        Using terminal reward assignment: the full episode reward is assigned
        to the final step only. This is appropriate for sparse reward settings
        like ALFWorld where success/failure is determined at episode end.
        
        The group-normalized advantage computation handles credit assignment
        by comparing trajectories that share the same initial state.
        """
        step_rewards = np.zeros(trajectory_length, dtype=np.float32)
        
        if trajectory_length > 0:
            # Assign full episode reward to terminal step
            step_rewards[-1] = float(episode_reward)
        
        return step_rewards