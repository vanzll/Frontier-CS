import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Compute masked mean of tensor."""
    if mask is None:
        return x.mean(dim=dim)
    mask = mask.float()
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp(min=1e-8)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1e-8)


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm with GiGPO-style configuration.
        Uses group normalization at both episode and step levels.
        """
        self.config = {
            "gamma": 0.99,
            "clip_ratio": 0.2,
            "use_kl_loss": False,
            "kl_loss_coef": 0.001,
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
        GiGPO-style advantage computation with episode-level and step-level grouping.
        
        Key innovations:
        1. Episode-level grouping (GRPO): Normalize within trajectories from same initial state
        2. Step-level grouping (GiGPO): Use anchor observations to group similar states
        """
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device
        dtype = token_level_rewards.dtype
        
        # Compute sequence-level returns from token rewards
        seq_rewards = (token_level_rewards * response_mask).sum(dim=1)
        
        # Initialize returns and advantages
        returns = seq_rewards.unsqueeze(1).expand(-1, seq_len).clone()
        advantages = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        
        # Step 1: Episode-level grouping (GRPO-style normalization)
        unique_episodes = np.unique(episode_index)
        episode_advantages = torch.zeros(batch_size, device=device, dtype=dtype)
        
        for ep_idx in unique_episodes:
            ep_mask = (episode_index == ep_idx)
            indices = np.where(ep_mask)[0]
            
            if len(indices) > 1:
                group_rewards = seq_rewards[indices]
                mean_r = group_rewards.mean()
                std_r = group_rewards.std()
                
                if std_r > 1e-8:
                    norm_rewards = (group_rewards - mean_r) / std_r
                else:
                    norm_rewards = group_rewards - mean_r
                
                for i, idx in enumerate(indices):
                    episode_advantages[idx] = norm_rewards[i]
            else:
                idx = indices[0]
                episode_advantages[idx] = seq_rewards[idx]
        
        # Step 2: Step-level grouping using anchor observations (GiGPO enhancement)
        if anchor_observations is not None and len(anchor_observations) > 0:
            # Use anchor observations to create step-level groups
            # Group by hashing the anchor observation string
            anchor_groups = defaultdict(list)
            
            for i, anchor in enumerate(anchor_observations):
                if anchor is not None:
                    # Convert to string for grouping
                    key = str(anchor) if not isinstance(anchor, str) else anchor
                    anchor_groups[key].append(i)
            
            # Normalize within each anchor group
            step_advantages = episode_advantages.clone()
            
            for key, indices in anchor_groups.items():
                if len(indices) > 1:
                    group_adv = episode_advantages[indices]
                    mean_adv = group_adv.mean()
                    std_adv = group_adv.std()
                    
                    if std_adv > 1e-8:
                        for idx in indices:
                            step_advantages[idx] = (episode_advantages[idx] - mean_adv) / std_adv
            
            # Use step-level advantages
            for i in range(batch_size):
                advantages[i] = step_advantages[i]
        else:
            # Fall back to episode-level advantages only
            for i in range(batch_size):
                advantages[i] = episode_advantages[i]
        
        # Apply response mask
        advantages = advantages * response_mask
        returns = returns * response_mask
        
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
        PPO-style clipped policy loss with additional stability measures.
        """
        # Compute importance sampling ratio
        ratio = torch.exp(log_prob - old_log_prob)
        
        # Clamp ratio for numerical stability
        ratio = torch.clamp(ratio, 0.0, 10.0)
        
        # PPO clipping
        ratio_clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        
        # Surrogate objectives
        surr1 = ratio * advantages
        surr2 = ratio_clipped * advantages
        
        # PPO loss: minimize negative of minimum (pessimistic bound)
        policy_loss = -torch.min(surr1, surr2)
        
        # Average over valid tokens
        loss = masked_mean(policy_loss, response_mask)
        
        # Compute diagnostic metrics
        with torch.no_grad():
            clip_frac = masked_mean(
                ((ratio - 1.0).abs() > clip_ratio).float(),
                response_mask
            ).item()
            
            # Approximate KL divergence (more stable version)
            log_ratio = log_prob - old_log_prob
            approx_kl = masked_mean(
                (torch.exp(log_ratio) - 1) - log_ratio,
                response_mask
            ).item()
            
            # Policy entropy approximation
            entropy = masked_mean(-log_prob, response_mask).item()
        
        metrics = {
            "clip_frac": clip_frac,
            "approx_kl": approx_kl,
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
            "entropy": entropy,
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
        Assign episode reward with exponentially increasing weights toward the end.
        
        This encourages the model to focus on later steps that are closer to
        achieving the goal, while still providing some signal for earlier steps.
        """
        step_rewards = np.zeros(trajectory_length)
        
        if trajectory_length > 0 and episode_reward != 0:
            # Exponentially increasing weights toward the end
            # Later steps get more credit as they're closer to task completion
            weights = np.exp(np.linspace(-2, 0, trajectory_length))
            weights = weights / weights.sum()  # Normalize
            step_rewards = weights * episode_reward
        
        return step_rewards