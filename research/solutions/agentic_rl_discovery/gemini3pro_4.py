from typing import Tuple, Dict, Any, Optional, List
import torch
import numpy as np
from collections import defaultdict
from verl.utils.torch_functional import masked_mean

class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm configuration.
        """
        self.config = {
            "gamma": 1.0,            # No discount for sparse episodic success rewards
            "clip_ratio": 0.2,       # PPO clip ratio
            "use_kl_loss": False,    # KL handling via PPO clipping
            "kl_loss_coef": 0.01,
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
        Compute advantages using GRPO (Group Relative Policy Optimization).
        Normalizes trajectory returns within the group of trajectories generated from the same prompt.
        """
        # 1. Compute Episode Returns (sum of rewards along sequence)
        # Assuming sparse rewards (0 everywhere except end), sum gives the episode outcome.
        # shape: (batch,)
        episode_returns = token_level_rewards.sum(dim=-1)

        # 2. Initialize output tensors
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        
        # 3. Group by episode_index (Prompt ID)
        unique_episodes = np.unique(episode_index)
        
        for ep_id in unique_episodes:
            # Find trajectories belonging to this episode group
            indices = np.where(episode_index == ep_id)[0]
            indices_torch = torch.from_numpy(indices).to(token_level_rewards.device)
            
            # Extract returns for this group
            group_returns = episode_returns[indices_torch]  # (group_size,)
            
            # GRPO Normalization: Advantage = (Return - Mean) / Std
            if len(group_returns) > 1:
                mean = group_returns.mean()
                std = group_returns.std()
                # Add epsilon for numerical stability
                group_adv = (group_returns - mean) / (std + 1e-8)
            else:
                # If only one trajectory, advantage is 0
                group_adv = torch.zeros_like(group_returns)
            
            # Broadcast scalar advantage to all tokens in the sequence
            # We unsqueeze to (group_size, 1) and broadcast to (group_size, seq_len)
            advantages[indices_torch] = group_adv.unsqueeze(1).to(advantages.dtype)
            
            # Store returns (expanded) for logging
            returns[indices_torch] = group_returns.unsqueeze(1).to(returns.dtype)

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
        Compute PPO-style policy gradient loss with clipping.
        """
        # Calculate ratio: pi_new / pi_old
        # log_prob and old_log_prob are (batch, seq_len)
        ratio = torch.exp(log_prob - old_log_prob)
        
        # Calculate surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        
        # PPO Loss: minimize -min(surr1, surr2)
        loss = -torch.min(surr1, surr2)
        
        # Average loss over valid response tokens
        policy_loss = masked_mean(loss, response_mask)
        
        # Compute metrics for monitoring
        with torch.no_grad():
            # Fraction of updates clipped
            is_clipped = (ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)
            clip_frac = masked_mean(is_clipped.float(), response_mask)
            
            # Approximate KL divergence: E[log_old - log_new]
            approx_kl = masked_mean(old_log_prob - log_prob, response_mask)
            
        metrics = {
            "policy_loss": policy_loss.item(),
            "clip_frac": clip_frac.item(),
            "approx_kl": approx_kl.item()
        }
        
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
        Assign episode reward to the final step of the trajectory.
        Intermediate steps get 0 reward.
        """
        rewards = np.zeros(trajectory_length, dtype=np.float32)
        if trajectory_length > 0:
            rewards[-1] = episode_reward
        return rewards