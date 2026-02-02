import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional

class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm. Called once before training.

        Set hyperparameters in self.config:
            self.config = {
                "gamma": 0.95,           # Discount factor
                "clip_ratio": 0.2,       # PPO clip range
                "use_kl_loss": False,    # Add KL penalty
                "kl_loss_coef": 0.01,    # KL penalty coefficient
            }

        Returns: self
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
        Compute advantages and returns for policy update.
        """
        device = token_level_rewards.device
        batch_size, seq_len = token_level_rewards.shape

        # 1. Calculate token-level discounted returns.
        returns = torch.zeros_like(token_level_rewards)
        next_return = torch.zeros(batch_size, device=device)
        for t in reversed(range(seq_len)):
            next_return = token_level_rewards[:, t] + gamma * next_return
            next_return = next_return * response_mask[:, t].float()
            returns[:, t] = next_return

        # 2. Compute trajectory-level advantages using Group-wise Reward Policy Optimization (GRPO).
        traj_rewards = torch.sum(token_level_rewards, dim=-1)
        
        traj_advantages = torch.zeros_like(traj_rewards)
        unique_episode_indices = np.unique(episode_index)

        for idx in unique_episode_indices:
            group_mask = (episode_index == idx)
            group_traj_rewards = traj_rewards[group_mask]
            
            if len(group_traj_rewards) > 1:
                mean = group_traj_rewards.mean()
                std = group_traj_rewards.std()
                if std > 1e-8:
                    whitened_rewards = (group_traj_rewards - mean) / std
                    traj_advantages[group_mask] = whitened_rewards
        
        # 3. Broadcast trajectory-level advantages to all response tokens.
        advantages = traj_advantages.unsqueeze(1).expand_as(returns)
        advantages = advantages * response_mask.float()
        
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
        Compute policy gradient loss.
        """
        ratio = torch.exp(log_prob - old_log_prob)

        surr1 = ratio * advantages
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        surr2 = clipped_ratio * advantages

        loss = -torch.min(surr1, surr2)

        masked_loss = loss * response_mask.float()
        num_response_tokens = response_mask.sum()

        if num_response_tokens > 0:
            total_loss = masked_loss.sum() / num_response_tokens
        else:
            total_loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

        with torch.no_grad():
            if num_response_tokens > 0:
                clip_mask = (torch.abs(ratio - 1.0) > clip_ratio) & response_mask
                clip_frac = clip_mask.sum() / num_response_tokens

                approx_kl = ((ratio - 1) - (log_prob - old_log_prob)) * response_mask.float()
                approx_kl_mean = approx_kl.sum() / num_response_tokens
                
                masked_advantages = torch.masked_select(advantages, response_mask)
                advantages_mean = masked_advantages.mean()
            else:
                clip_frac = torch.tensor(0.0, device=loss.device)
                approx_kl_mean = torch.tensor(0.0, device=loss.device)
                advantages_mean = torch.tensor(0.0, device=loss.device)

        metrics = {
            "clip_frac": clip_frac.item(),
            "approx_kl": approx_kl_mean.item(),
            "advantages": advantages_mean.item(),
        }

        return total_loss, metrics

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
        """
        step_rewards = np.zeros(trajectory_length, dtype=np.float32)
        if episode_reward > 0 and trajectory_length > 0:
            step_rewards[-1] = episode_reward
        return step_rewards