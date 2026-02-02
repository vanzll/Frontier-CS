from typing import Tuple, Dict, Any, Optional, List
import torch
import numpy as np
from collections import defaultdict

class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm. Called once before training.

        Set hyperparameters in self.config:
            self.config = {
                "gamma": 1.0,            # No discounting for sparse episodic tasks (GRPO)
                "clip_ratio": 0.2,       # PPO clip range
                "use_kl_loss": False,    # KL usually handled via reward or implicit in GRPO
                "kl_loss_coef": 0.0,
            }

        Returns: self
        """
        self.config = {
            "gamma": 1.0,
            "clip_ratio": 0.2,
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
        }
        return self

    def compute_advantage(
        self,
        token_level_rewards: torch.Tensor,   # (batch, seq_len)
        response_mask: torch.Tensor,          # (batch, seq_len)
        episode_index: np.ndarray,            # (batch,) - episode group IDs
        trajectory_index: np.ndarray,         # (batch,) - trajectory IDs within group
        step_rewards: Optional[torch.Tensor] = None,
        anchor_observations: Optional[np.ndarray] = None,
        gamma: float = 1.0,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns using Group Relative Policy Optimization (GRPO).
        
        We normalize the returns within each episode group (prompt group) to estimate
        the advantage of each trajectory relative to its peers.
        """
        # 1. Compute trajectory-level returns
        # For ALFWorld/Reasoning, reward is typically at the end or sparse.
        # Summing gives the total episode return.
        # token_level_rewards: (B, L)
        trajectory_returns = token_level_rewards.sum(dim=-1) # (B,)

        # 2. Group Normalization
        # We calculate Advantage = (Return - Mean(Group)) / (Std(Group) + epsilon)
        advantages = torch.zeros_like(trajectory_returns)
        
        # Identify unique episode groups
        unique_episodes = np.unique(episode_index)
        
        for ep_id in unique_episodes:
            # Find trajectories belonging to this prompt/episode
            group_mask = (episode_index == ep_id)
            batch_indices = torch.tensor(np.where(group_mask)[0], device=token_level_rewards.device)
            
            if len(batch_indices) == 0:
                continue
                
            group_scores = trajectory_returns[batch_indices]
            
            if len(batch_indices) > 1:
                mean = group_scores.mean()
                std = group_scores.std()
                # Normalize
                group_adv = (group_scores - mean) / (std + 1e-8)
            else:
                # If only one trajectory, advantage is 0
                group_adv = torch.zeros_like(group_scores)
            
            advantages[batch_indices] = group_adv

        # 3. Broadcast to token level
        # GRPO assigns the trajectory advantage to every valid response token
        advantages_seq = advantages.unsqueeze(-1) * response_mask
        
        # Returns (for logging or optional critic, though GRPO is actor-only)
        returns_seq = trajectory_returns.unsqueeze(-1) * response_mask

        return advantages_seq.detach(), returns_seq.detach()

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
        Compute policy gradient loss using PPO clipping.
        """
        # Calculate ratio pi_new / pi_old
        ratio = torch.exp(log_prob - old_log_prob)
        
        # Surrogate objectives
        # We maximize Objective, so we minimize -Objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        
        # Element-wise loss
        loss_elements = -torch.min(surr1, surr2)
        
        # Masking and Averaging
        # Only compute loss on valid response tokens
        loss_masked = loss_elements * response_mask
        num_valid = response_mask.sum()
        
        if num_valid > 0:
            loss = loss_masked.sum() / num_valid
        else:
            loss = loss_masked.sum() * 0.0 # Avoid NaN
            
        # Metrics for monitoring
        with torch.no_grad():
            # Calculate clipping fraction
            clipped = (ratio < (1.0 - clip_ratio)) | (ratio > (1.0 + clip_ratio))
            clip_frac = (clipped.float() * response_mask).sum() / (num_valid + 1e-8)
            
            # Approximate KL divergence (http://joschu.net/blog/kl-approx.html)
            # k1 = log_ratio - ratio + 1. Using simpler mean(log_old - log_new)
            approx_kl = ((old_log_prob - log_prob) * response_mask).sum() / (num_valid + 1e-8)
            
            avg_adv = (advantages * response_mask).sum() / (num_valid + 1e-8)

        metrics = {
            "clip_frac": clip_frac.item(),
            "approx_kl": approx_kl.item(),
            "avg_advantage": avg_adv.item()
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
        For sparse reasoning tasks, we assign the final outcome reward to the last step.
        """
        rewards = np.zeros(trajectory_length, dtype=np.float32)
        if trajectory_length > 0:
            rewards[-1] = episode_reward
        return rewards