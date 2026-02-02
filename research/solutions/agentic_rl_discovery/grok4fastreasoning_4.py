import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List

try:
    from verl.utils.torch_functional import masked_mean, masked_whiten
except ImportError:
    def masked_mean(tensor, mask, dim=None):
        if dim is None:
            return (tensor * mask).sum() / mask.sum().clamp(min=1e-8)
        return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1e-8)
    
    def masked_whiten(tensor, mask, dim=None, unbiased=True):
        if dim is None:
            mean = masked_mean(tensor, mask)
            var = masked_mean((tensor - mean) ** 2, mask)
        else:
            mean = masked_mean(tensor, mask, dim).unsqueeze(dim)
            var = masked_mean((tensor - mean) ** 2, mask, dim).unsqueeze(dim)
        return (tensor - mean) / (var.sqrt() + 1e-8)


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
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
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device
        dtype = token_level_rewards.dtype
        
        rewards_to_use = token_level_rewards
        if step_rewards is not None:
            if isinstance(step_rewards, np.ndarray):
                step_rewards = torch.from_numpy(step_rewards).to(device=device, dtype=dtype)
            if step_rewards.dim() == 1:
                step_rewards = step_rewards.unsqueeze(0).expand(batch_size, -1)
            if step_rewards.shape[1] != seq_len:
                padded = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
                min_len = min(step_rewards.shape[1], seq_len)
                padded[:, :min_len] = step_rewards[:, :min_len]
                step_rewards = padded
            rewards_to_use = step_rewards
        
        returns = self._compute_discounted_returns(rewards_to_use, response_mask, gamma)
        
        trajectory_returns = (returns * response_mask).sum(dim=1) / response_mask.sum(dim=1).clamp(min=1)
        
        advantages = torch.zeros_like(returns)
        
        unique_episodes = np.unique(episode_index)
        for ep_idx in unique_episodes:
            ep_mask = episode_index == ep_idx
            ep_indices = np.where(ep_mask)[0]
            
            if len(ep_indices) <= 1:
                group_returns = trajectory_returns[ep_indices]
                baseline = group_returns.mean()
                for idx in ep_indices:
                    advantages[idx] = (returns[idx] - baseline) * response_mask[idx]
                continue
            
            group_returns = trajectory_returns[ep_indices]
            mean_return = group_returns.mean()
            std_return = group_returns.std()
            
            if std_return < 1e-8:
                for idx in ep_indices:
                    advantages[idx] = torch.zeros_like(returns[idx])
            else:
                for idx in ep_indices:
                    norm_adv = (trajectory_returns[idx] - mean_return) / (std_return + 1e-8)
                    advantages[idx] = norm_adv * response_mask[idx]
        
        if anchor_observations is not None:
            advantages = self._apply_step_level_normalization(
                advantages, returns, response_mask, 
                episode_index, anchor_observations
            )
        
        return advantages, returns
    
    def _compute_discounted_returns(
        self, 
        rewards: torch.Tensor, 
        mask: torch.Tensor, 
        gamma: float
    ) -> torch.Tensor:
        batch_size, seq_len = rewards.shape
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(batch_size, device=rewards.device, dtype=rewards.dtype)
        
        for t in reversed(range(seq_len)):
            running_return = rewards[:, t] + gamma * running_return * mask[:, t]
            returns[:, t] = running_return
        
        return returns
    
    def _apply_step_level_normalization(
        self,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor,
        episode_index: np.ndarray,
        anchor_observations: np.ndarray
    ) -> torch.Tensor:
        batch_size = advantages.shape[0]
        
        obs_groups = defaultdict(list)
        for i in range(batch_size):
            ep_idx = episode_index[i]
            if anchor_observations[i] is not None:
                if isinstance(anchor_observations[i], (list, np.ndarray)):
                    obs_key = (ep_idx, tuple(anchor_observations[i]) if hasattr(anchor_observations[i], '__iter__') else anchor_observations[i])
                else:
                    obs_key = (ep_idx, str(anchor_observations[i]))
            else:
                obs_key = (ep_idx, i)
            obs_groups[obs_key].append(i)
        
        for obs_key, indices in obs_groups.items():
            if len(indices) <= 1:
                continue
            
            group_advs = torch.stack([advantages[idx].sum() / mask[idx].sum().clamp(min=1) for idx in indices])
            mean_adv = group_advs.mean()
            std_adv = group_advs.std()
            
            if std_adv > 1e-8:
                for idx in indices:
                    traj_mean_adv = advantages[idx].sum() / mask[idx].sum().clamp(min=1)
                    norm_factor = (traj_mean_adv - mean_adv) / (std_adv + 1e-8)
                    advantages[idx] = norm_factor * mask[idx]
        
        return advantages
    
    def compute_policy_loss(
        self,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        clip_ratio: float = 0.2,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        ratio = torch.exp(log_prob - old_log_prob)
        
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        
        surr1 = -advantages * ratio
        surr2 = -advantages * clipped_ratio
        policy_loss = torch.max(surr1, surr2)
        
        loss = masked_mean(policy_loss, response_mask)
        
        with torch.no_grad():
            clipped = (torch.abs(ratio - 1.0) > clip_ratio).float()
            clip_frac = masked_mean(clipped, response_mask).item()
            
            log_ratio = log_prob - old_log_prob
            approx_kl = masked_mean((torch.exp(log_ratio) - 1) - log_ratio, response_mask).item()
            
            entropy_bonus = masked_mean(-log_prob, response_mask).item()
        
        metrics = {
            "clip_frac": clip_frac,
            "approx_kl": approx_kl,
            "policy_loss": loss.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
            "advantage_mean": masked_mean(advantages, response_mask).item(),
            "entropy_estimate": entropy_bonus,
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
        step_rewards = np.zeros(trajectory_length, dtype=np.float32)
        
        if trajectory_length == 0:
            return step_rewards
        
        if episode_reward > 0:
            step_rewards[-1] = episode_reward
        else:
            step_rewards[-1] = episode_reward
        
        return step_rewards