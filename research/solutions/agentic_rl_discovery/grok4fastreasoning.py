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
    
    def masked_whiten(tensor, mask, dim=None, eps=1e-8):
        mean = masked_mean(tensor, mask, dim=dim)
        if dim is not None:
            mean = mean.unsqueeze(dim)
        centered = (tensor - mean) * mask
        var = masked_mean(centered ** 2, mask, dim=dim)
        if dim is not None:
            var = var.unsqueeze(dim)
        return centered / (var.sqrt() + eps)


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm with GiGPO-inspired hyperparameters.
        """
        self.config = {
            "gamma": 0.99,
            "clip_ratio": 0.2,
            "use_kl_loss": False,
            "kl_loss_coef": 0.01,
            "use_step_level_grouping": True,
            "advantage_norm_eps": 1e-8,
            "reward_decay": 0.95,
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
        Compute advantages using GiGPO-style group-relative policy optimization.
        
        Key ideas:
        1. Group trajectories by episode (same initial state)
        2. Optionally group by step-level anchor observations
        3. Compute returns and normalize within groups
        """
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device
        
        # Use step rewards if available, otherwise use token-level rewards
        if step_rewards is not None:
            # Expand step rewards to token level (assign to last token of each step)
            rewards = step_rewards.to(device)
        else:
            rewards = token_level_rewards
        
        # Compute per-trajectory returns (sum of rewards)
        trajectory_returns = (rewards * response_mask).sum(dim=1)  # (batch,)
        
        # Group-relative advantage computation (GRPO/GiGPO style)
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        
        # Convert to numpy for grouping
        episode_idx_np = np.asarray(episode_index)
        unique_episodes = np.unique(episode_idx_np)
        
        for ep_idx in unique_episodes:
            ep_mask = episode_idx_np == ep_idx
            ep_indices = np.where(ep_mask)[0]
            
            if len(ep_indices) <= 1:
                # Single trajectory, no relative advantage
                idx = ep_indices[0]
                traj_return = trajectory_returns[idx]
                advantages[idx] = (rewards[idx] - traj_return / response_mask[idx].sum().clamp(min=1)) * response_mask[idx]
                returns[idx] = rewards[idx] * response_mask[idx]
                continue
            
            # Get returns for this episode group
            group_returns = trajectory_returns[ep_indices]
            
            # Compute group statistics
            group_mean = group_returns.mean()
            group_std = group_returns.std().clamp(min=self.config.get("advantage_norm_eps", 1e-8))
            
            # Normalize returns within group
            normalized_returns = (group_returns - group_mean) / group_std
            
            # Step-level grouping if anchor observations are available
            if anchor_observations is not None and self.config.get("use_step_level_grouping", True):
                # Additional step-level normalization using anchor observations
                anchor_obs = anchor_observations[ep_indices]
                step_groups = defaultdict(list)
                
                for i, (local_idx, obs) in enumerate(zip(ep_indices, anchor_obs)):
                    # Use observation hash as step group key
                    if isinstance(obs, (list, tuple)):
                        obs_key = str(obs)
                    elif isinstance(obs, np.ndarray):
                        obs_key = obs.tobytes()
                    else:
                        obs_key = str(obs)
                    step_groups[obs_key].append((i, local_idx))
                
                # Compute step-level normalized advantages
                for step_key, indices_pairs in step_groups.items():
                    if len(indices_pairs) > 1:
                        local_indices = [p[0] for p in indices_pairs]
                        global_indices = [p[1] for p in indices_pairs]
                        
                        step_returns = group_returns[local_indices]
                        step_mean = step_returns.mean()
                        step_std = step_returns.std().clamp(min=self.config.get("advantage_norm_eps", 1e-8))
                        step_norm_returns = (step_returns - step_mean) / step_std
                        
                        for j, global_idx in enumerate(global_indices):
                            norm_ret = step_norm_returns[j].item()
                            advantages[global_idx] = norm_ret * response_mask[global_idx]
                            returns[global_idx] = trajectory_returns[global_idx] * response_mask[global_idx] / response_mask[global_idx].sum().clamp(min=1)
                    else:
                        # Single trajectory at this step, use episode-level normalization
                        local_idx, global_idx = indices_pairs[0]
                        norm_ret = normalized_returns[local_idx].item()
                        advantages[global_idx] = norm_ret * response_mask[global_idx]
                        returns[global_idx] = trajectory_returns[global_idx] * response_mask[global_idx] / response_mask[global_idx].sum().clamp(min=1)
            else:
                # Episode-level grouping only (GRPO style)
                for i, global_idx in enumerate(ep_indices):
                    norm_ret = normalized_returns[i].item()
                    advantages[global_idx] = norm_ret * response_mask[global_idx]
                    returns[global_idx] = trajectory_returns[global_idx] * response_mask[global_idx] / response_mask[global_idx].sum().clamp(min=1)
        
        # Final whitening of advantages across batch for stability
        if response_mask.sum() > 1:
            adv_mean = masked_mean(advantages, response_mask)
            adv_var = masked_mean((advantages - adv_mean) ** 2, response_mask)
            advantages = (advantages - adv_mean) / (adv_var.sqrt() + self.config.get("advantage_norm_eps", 1e-8))
            advantages = advantages * response_mask
        
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
        Compute PPO-style clipped policy loss.
        """
        # Compute probability ratio
        ratio = torch.exp(log_prob - old_log_prob)
        
        # Clipped surrogate objective
        clip_ratio = self.config.get("clip_ratio", clip_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        
        # PPO loss: min of clipped and unclipped
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        policy_loss = -torch.min(surr1, surr2)
        
        # Apply mask and compute mean loss
        loss = masked_mean(policy_loss, response_mask)
        
        # Compute metrics
        with torch.no_grad():
            clip_frac = masked_mean(
                ((ratio - 1.0).abs() > clip_ratio).float(),
                response_mask
            ).item()
            
            approx_kl = masked_mean(
                0.5 * (log_prob - old_log_prob) ** 2,
                response_mask
            ).item()
            
            mean_ratio = masked_mean(ratio, response_mask).item()
        
        metrics = {
            "clip_frac": clip_frac,
            "approx_kl": approx_kl,
            "mean_ratio": mean_ratio,
            "policy_loss": loss.item(),
        }
        
        # Optional KL penalty
        if self.config.get("use_kl_loss", False):
            kl_coef = self.config.get("kl_loss_coef", 0.01)
            kl_loss = masked_mean((old_log_prob - log_prob), response_mask)
            loss = loss + kl_coef * kl_loss
            metrics["kl_loss"] = kl_loss.item()
        
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
        
        Strategy: Assign most reward to final step with decay for earlier steps.
        For sparse binary rewards (0/1), this helps with credit assignment.
        """
        step_rewards = np.zeros(trajectory_length, dtype=np.float32)
        
        if trajectory_length == 0:
            return step_rewards
        
        if episode_reward > 0:
            # Successful episode: assign decayed rewards
            gamma = self.config.get("reward_decay", 0.95)
            
            # Compute discounted rewards from end to start
            discounted_reward = episode_reward
            for t in range(trajectory_length - 1, -1, -1):
                step_rewards[t] = discounted_reward
                discounted_reward *= gamma
            
            # Normalize to sum to episode_reward
            total = step_rewards.sum()
            if total > 0:
                step_rewards = step_rewards * (episode_reward / total)
        else:
            # Failed episode: small negative reward at final step
            # This encourages shorter failure trajectories
            step_rewards[-1] = -0.1 / trajectory_length
        
        return step_rewards