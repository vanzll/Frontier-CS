import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional

# verl-agent framework might provide these utilities.
# A fallback is included for robustness in case they are not in the path.
try:
    from verl.utils.torch_functional import masked_mean
except ImportError:
    def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean of a tensor over a masked region.
        """
        # Ensure mask sum is not zero to avoid division by zero
        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            return torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
        return (tensor * mask).sum() / mask_sum.clamp(min=1e-8)

class Solution:
    """
    Implementation of a GiGPO-style reinforcement learning algorithm.
    This algorithm uses Proximal Policy Optimization (PPO) with group-wise
    advantage normalization, inspired by the GiGPO and GRPO papers, which have
    shown strong performance in training LLM agents.
    """
    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm and set hyperparameters.
        """
        self.config = {
            "gamma": 0.99,           # Discount factor for future rewards
            "clip_ratio": 0.2,       # PPO clipping parameter for policy updates
            "use_kl_loss": True,     # Use KL divergence penalty for stabilization
            "kl_loss_coef": 0.01,    # Coefficient for the KL penalty
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
        gamma: float = 0.95,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes advantages and returns using group-wise normalization (GiGPO-style).
        Advantages are calculated by normalizing returns within groups of trajectories
        that share the same starting state and intermediate state representations.
        """
        device = token_level_rewards.device
        batch_size, seq_len = token_level_rewards.shape

        # 1. Aggregate token-level rewards to get a reward per step.
        # In this framework, the step reward is typically placed at the EOS token.
        step_level_rewards = torch.sum(token_level_rewards, dim=-1)

        # 2. Compute discounted returns for each trajectory.
        returns = torch.zeros_like(step_level_rewards)
        
        trajectories = defaultdict(list)
        for i in range(batch_size):
            traj_id = (episode_index[i], trajectory_index[i])
            trajectories[traj_id].append(i)

        for traj_id in trajectories:
            indices = trajectories[traj_id]
            traj_rewards = step_level_rewards[indices]
            
            discounted_return = 0.0
            traj_returns = torch.zeros_like(traj_rewards)
            for t in reversed(range(len(traj_rewards))):
                discounted_return = traj_rewards[t] + gamma * discounted_return
                traj_returns[t] = discounted_return
            
            returns[indices] = traj_returns

        # 3. Compute advantages via group-wise normalization (whitening).
        # Groups are defined by episode and anchor observations (GiGPO)
        # or just episodes if anchors are not available (GRPO fallback).
        if anchor_observations is not None and len(anchor_observations) == batch_size:
            unique_anchors, anchor_ids_np = np.unique(anchor_observations, return_inverse=True)
            max_episodes = (episode_index.max() + 1) if len(episode_index) > 0 else 1
            group_ids_np = anchor_ids_np * max_episodes + episode_index
        else:
            group_ids_np = episode_index

        group_ids = torch.from_numpy(group_ids_np).to(device)
        unique_group_ids, group_indices = torch.unique(group_ids, return_inverse=True)
        num_groups = len(unique_group_ids)

        # Efficiently compute group-wise mean and std using scatter_add
        group_means = torch.zeros(num_groups, device=device, dtype=returns.dtype).scatter_add_(0, group_indices, returns)
        group_counts = torch.zeros(num_groups, device=device, dtype=torch.long).scatter_add_(0, group_indices, torch.ones_like(group_indices, dtype=torch.long))
        
        safe_counts = group_counts.float().clamp(min=1)
        group_means /= safe_counts

        return_means_per_item = group_means[group_indices]
        
        group_vars = torch.zeros(num_groups, device=device, dtype=returns.dtype).scatter_add_(0, group_indices, (returns - return_means_per_item)**2)
        group_vars /= safe_counts
        group_stds = torch.sqrt(group_vars + 1e-8)

        return_stds_per_item = group_stds[group_indices]
        
        advantages = (returns - return_means_per_item) / return_stds_per_item
        
        # For groups with a single member, std is 0, advantage is NaN. Set to 0.
        single_item_mask = (group_counts[group_indices] <= 1)
        advantages[single_item_mask] = 0.0

        # 4. Broadcast step-level values to token-level and apply response mask.
        advantages = advantages.unsqueeze(1).expand(-1, seq_len) * response_mask
        returns = returns.unsqueeze(1).expand(-1, seq_len) * response_mask

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
        Computes the PPO-clip policy loss, with an optional KL penalty.
        """
        log_ratio = log_prob - old_log_prob
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        
        policy_loss = -torch.min(surr1, surr2)
        
        loss = masked_mean(policy_loss, response_mask)

        with torch.no_grad():
            is_clipped = (ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)
            clip_frac = masked_mean(is_clipped.float(), response_mask).item()
            
            metrics = {
                "clip_frac": clip_frac,
                "advantages": masked_mean(advantages, response_mask).item(),
                "ratio": masked_mean(ratio, response_mask).item(),
                "policy_loss": masked_mean(policy_loss, response_mask).item()
            }

        if self.config.get("use_kl_loss", False):
            kl_loss_coef = self.config.get("kl_loss_coef", 0.01)
            approx_kl = (ratio - 1) - log_ratio
            kl_penalty = kl_loss_coef * masked_mean(approx_kl, response_mask)
            loss += kl_penalty
            metrics["kl_penalty"] = kl_penalty.item()

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
        Assigns rewards to the steps of a trajectory. For goal-oriented tasks
        like ALFWorld, a sparse reward at the end of a successful episode is
        a robust and effective strategy.
        """
        step_rewards = np.zeros(trajectory_length, dtype=np.float32)
        
        if episode_reward > 0:
            step_rewards[-1] = episode_reward
            
        return step_rewards