import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional

class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm. Called once before training.
        """
        self.config = {
            "gamma": 1.0,            # Discount factor (1.0 for sparse episodic tasks)
            "clip_ratio": 0.2,       # PPO clip range
            "use_kl_loss": True,     # Add KL penalty
            "kl_loss_coef": 0.02,    # KL penalty coefficient
        }
        return self

    def compute_advantage(
        self,
        token_level_rewards: torch.Tensor,   # (batch, seq_len)
        response_mask: torch.Tensor,          # (batch, seq_len)
        episode_index: np.ndarray,            # (batch,)
        trajectory_index: np.ndarray,         # (batch,)
        step_rewards: Optional[torch.Tensor] = None,
        anchor_observations: Optional[np.ndarray] = None,
        gamma: float = 1.0,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns using Group Relative Policy Optimization (GRPO).
        Normalize returns at each token step relative to the episode group.
        """
        # Ensure input types
        rewards = token_level_rewards.float()
        mask = response_mask.float()
        bs, seq_len = rewards.shape
        device = rewards.device
        
        # 1. Compute Returns (Monte Carlo / GAE with lambda=1)
        # Iterate backwards to accumulate rewards
        returns = torch.zeros_like(rewards)
        running_returns = torch.zeros(bs, device=device)
        
        for t in reversed(range(seq_len)):
            # G_t = r_t + gamma * G_{t+1}
            running_returns = rewards[:, t] + gamma * running_returns
            returns[:, t] = running_returns
            
        # Apply mask
        returns = returns * mask
        
        # 2. Group-based Advantage Normalization
        # Normalize returns at each token step across trajectories in the same group
        advantages = torch.zeros_like(returns)
        unique_eps = np.unique(episode_index)
        
        for ep_id in unique_eps:
            # Indices for current group
            group_indices_np = np.where(episode_index == ep_id)[0]
            group_indices = torch.tensor(group_indices_np, device=device)
            
            # Extract group data: (GroupSize, SeqLen)
            grp_returns = returns[group_indices]
            grp_mask = mask[group_indices]
            
            # Compute statistics per token position (Mean, Std)
            # Count valid trajectories at each token position
            valid_counts = grp_mask.sum(dim=0)
            valid_counts_safe = torch.clamp(valid_counts, min=1.0)
            
            # Mean
            grp_sum = grp_returns.sum(dim=0)
            grp_mean = grp_sum / valid_counts_safe
            
            # Variance (masked)
            diff_sq = (grp_returns - grp_mean.unsqueeze(0)) ** 2
            diff_sq = diff_sq * grp_mask
            grp_var = diff_sq.sum(dim=0) / valid_counts_safe
            grp_std = torch.sqrt(grp_var + 1e-8)
            
            # Calculate Advantage: (R - Mean) / (Std + eps)
            grp_adv = (grp_returns - grp_mean.unsqueeze(0)) / (grp_std.unsqueeze(0) + 1e-8)
            
            # Mask invalid tokens
            grp_adv = grp_adv * grp_mask
            
            # Store results
            advantages.index_put_((group_indices,), grp_adv)
            
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
        Compute PPO policy gradient loss with optional KL penalty.
        """
        mask = response_mask.float()
        num_valid = mask.sum()
        
        # Policy Ratio
        ratio = torch.exp(log_prob - old_log_prob)
        
        # Surrogate Objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        
        # PPO Loss (Negative Min)
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = (policy_loss * mask).sum() / (num_valid + 1e-8)
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "adv_mean": (advantages * mask).sum().item() / (num_valid.item() + 1e-8)
        }
        
        # KL Penalty (Approximate KL: old_log_prob - log_prob)
        if self.config.get("use_kl_loss", False):
            kl = (old_log_prob - log_prob)
            kl_mean = (kl * mask).sum() / (num_valid + 1e-8)
            
            policy_loss = policy_loss + self.config["kl_loss_coef"] * kl_mean
            metrics["kl"] = kl_mean.item()
            
        return policy_loss, metrics

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
        Assigns the final episode reward to the last step (sparse reward).
        """
        rewards = np.zeros(trajectory_length, dtype=np.float32)
        if trajectory_length > 0:
            rewards[-1] = episode_reward
        return rewards