from typing import Tuple, Dict, Any, Optional, List
import torch
import numpy as np
from collections import defaultdict

class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm. Called once before training.
        """
        self.config = {
            "gamma": 0.99,
            "clip_ratio": 0.2,
            "use_kl_loss": False,
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
        gamma: float = 0.99,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns for policy update using GRPO logic.
        Normalizes trajectory returns within episode groups.
        """
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device

        # 1. Compute discounted returns (Cost-to-Go) for output
        # Used for value estimation or logging
        returns = torch.zeros_like(token_level_rewards)
        running_returns = torch.zeros(batch_size, device=device)
        
        # Backward pass for discounted returns
        for t in reversed(range(seq_len)):
            running_returns = token_level_rewards[:, t] + gamma * running_returns
            returns[:, t] = running_returns

        # 2. Compute Advantage using Group Relative Policy Optimization (GRPO)
        # We calculate the total return for each trajectory to serve as the scalar score
        # Since rewards are sparse/step-based, sum gives the trajectory performance
        trajectory_returns = token_level_rewards.sum(dim=1)  # (batch,)
        
        # Group trajectories by episode_index
        group_returns = defaultdict(list)
        group_indices = defaultdict(list)
        
        # Move to CPU for grouping logic (batch_size is small ~128)
        traj_returns_cpu = trajectory_returns.detach().cpu().numpy()
        
        for i in range(batch_size):
            eid = episode_index[i]
            group_returns[eid].append(traj_returns_cpu[i])
            group_indices[eid].append(i)
            
        advantages = torch.zeros_like(token_level_rewards)
        
        # Calculate normalized advantages per group
        for eid, rets in group_returns.items():
            rets_arr = np.array(rets)
            # Standard GRPO normalization
            mean = np.mean(rets_arr)
            std = np.std(rets_arr)
            
            # Handle single-sample groups or zero variance
            if std < 1e-8:
                std = 1.0
                
            # Broadcast scalar advantage to all tokens in the trajectory
            adv_values = (rets_arr - mean) / std
            
            indices = group_indices[eid]
            for idx, adv in zip(indices, adv_values):
                advantages[idx, :] = float(adv)
        
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
        # Mask out invalid tokens (padding/prompt)
        valid_mask = response_mask.bool()
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=log_prob.device, requires_grad=True), {}

        # Select valid tokens
        log_prob_valid = log_prob[valid_mask]
        old_log_prob_valid = old_log_prob[valid_mask]
        advantages_valid = advantages[valid_mask]
        
        # Calculate probability ratio
        ratio = torch.exp(log_prob_valid - old_log_prob_valid)
        
        # PPO Clipped Loss
        surr1 = ratio * advantages_valid
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_valid
        
        # Negative sign because we want to maximize objective (minimize loss)
        loss = -torch.min(surr1, surr2).mean()
        
        # Metrics
        with torch.no_grad():
            clip_frac = ((ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)).float().mean()
            approx_kl = (old_log_prob_valid - log_prob_valid).mean()
            
        metrics = {
            "policy_loss": loss.item(),
            "clip_frac": clip_frac.item(),
            "approx_kl": approx_kl.item(),
            "mean_advantage": advantages_valid.mean().item()
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
        Using sparse reward strategy: assign full reward to the final step.
        """
        rewards = np.zeros(trajectory_length, dtype=np.float32)
        if trajectory_length > 0:
            rewards[-1] = episode_reward
        return rewards