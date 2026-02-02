import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        self.config = {
            "gamma": 0.95,
            "clip_ratio": 0.2,
            "use_kl_loss": True,
            "kl_loss_coef": 0.01,
            "lambda": 0.95,
            "whiten_advantages": True,
            "normalize_returns": True,
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
        
        rewards = token_level_rewards
        if step_rewards is not None:
            step_rewards_expanded = step_rewards.unsqueeze(-1).expand(-1, seq_len)
            rewards = torch.where(response_mask.bool(), step_rewards_expanded, rewards)
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        episode_indices = torch.from_numpy(episode_index).to(device)
        trajectory_indices = torch.from_numpy(trajectory_index).to(device)
        
        lambda_ = self.config.get("lambda", 0.95)
        
        if anchor_observations is not None and step_rewards is not None:
            anchor_tensor = torch.from_numpy(anchor_observations).to(device)
            
            unique_episodes = episode_indices.unique()
            for ep_idx in unique_episodes:
                ep_mask = (episode_indices == ep_idx)
                
                unique_anchors = anchor_tensor[ep_mask].unique()
                for anchor in unique_anchors:
                    anchor_mask = (anchor_tensor == anchor) & ep_mask
                    
                    if anchor_mask.any():
                        anchor_rewards = rewards[anchor_mask]
                        anchor_mask_bool = anchor_mask[ep_mask][anchor_mask[ep_mask]]
                        
                        if anchor_rewards.numel() > 0:
                            anchor_returns = self._compute_discounted_returns(
                                anchor_rewards, gamma
                            )
                            anchor_advantages = self._compute_gae(
                                anchor_rewards, anchor_returns, gamma, lambda_
                            )
                            
                            if self.config.get("whiten_advantages", True):
                                anchor_advantages = self._whiten_advantages(
                                    anchor_advantages, anchor_mask_bool
                                )
                            
                            returns[anchor_mask] = anchor_returns
                            advantages[anchor_mask] = anchor_advantages
        else:
            unique_episodes = episode_indices.unique()
            for ep_idx in unique_episodes:
                ep_mask = (episode_indices == ep_idx)
                
                unique_trajectories = trajectory_indices[ep_mask].unique()
                for traj_idx in unique_trajectories:
                    traj_mask = (trajectory_indices == traj_idx) & ep_mask
                    
                    if traj_mask.any():
                        traj_rewards = rewards[traj_mask]
                        traj_mask_bool = traj_mask[ep_mask][traj_mask[ep_mask]]
                        
                        if traj_rewards.numel() > 0:
                            traj_returns = self._compute_discounted_returns(
                                traj_rewards, gamma
                            )
                            traj_advantages = self._compute_gae(
                                traj_rewards, traj_returns, gamma, lambda_
                            )
                            
                            if self.config.get("whiten_advantages", True):
                                traj_advantages = self._whiten_advantages(
                                    traj_advantages, traj_mask_bool
                                )
                            
                            returns[traj_mask] = traj_returns
                            advantages[traj_mask] = traj_advantages
        
        if self.config.get("normalize_returns", True):
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        advantages = advantages * response_mask
        returns = returns * response_mask
        
        return advantages, returns
    
    def _compute_discounted_returns(
        self, rewards: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(rewards.shape[0])):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        return returns
    
    def _compute_gae(
        self, rewards: torch.Tensor, returns: torch.Tensor, gamma: float, lambda_: float
    ) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        next_advantage = 0
        
        for t in reversed(range(rewards.shape[0])):
            delta = rewards[t] + gamma * returns[t+1] - returns[t] if t < len(rewards)-1 else rewards[t] - returns[t]
            advantages[t] = delta + gamma * lambda_ * next_advantage
            next_advantage = advantages[t]
        
        return advantages
    
    def _whiten_advantages(self, advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() > 0:
            masked_advantages = advantages[mask]
            if masked_advantages.std() > 1e-8:
                advantages[mask] = (masked_advantages - masked_advantages.mean()) / (masked_advantages.std() + 1e-8)
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
        device = old_log_prob.device
        
        ratio = torch.exp(log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        
        policy_loss1 = ratio * advantages
        policy_loss2 = clipped_ratio * advantages
        policy_loss = -torch.min(policy_loss1, policy_loss2)
        
        policy_loss = (policy_loss * response_mask).sum() / (response_mask.sum() + 1e-8)
        
        clip_fraction = ((ratio < 1 - clip_ratio) | (ratio > 1 + clip_ratio)).float()
        clip_fraction = (clip_fraction * response_mask).sum() / (response_mask.sum() + 1e-8)
        
        metrics = {"clip_frac": clip_fraction.item()}
        
        if self.config.get("use_kl_loss", False):
            kl_div = old_log_prob - log_prob
            kl_loss = (kl_div * response_mask).sum() / (response_mask.sum() + 1e-8)
            kl_penalty = self.config.get("kl_loss_coef", 0.01) * kl_loss
            policy_loss = policy_loss + kl_penalty
            metrics["kl_div"] = kl_loss.item()
        
        return policy_loss, metrics
    
    def assign_step_rewards(
        self,
        episode_reward: float,
        trajectory_length: int,
        step_observations: list,
        step_actions: list,
        **kwargs
    ) -> np.ndarray:
        step_rewards = np.zeros(trajectory_length)
        
        if episode_reward > 0:
            if trajectory_length <= 5:
                step_rewards[:] = episode_reward / trajectory_length
            else:
                step_rewards[-1] = episode_reward * 0.7
                
                remaining_reward = episode_reward * 0.3
                num_prior_steps = min(3, trajectory_length - 1)
                
                if num_prior_steps > 0:
                    step_rewards[-num_prior_steps-1:-1] = remaining_reward / num_prior_steps
                
                progress_bonus = np.linspace(0.1, 0.3, trajectory_length)
                step_rewards += progress_bonus * (episode_reward / trajectory_length)
        
        else:
            step_rewards[-1] = -0.1
            
            for i in range(trajectory_length - 1):
                obs = step_observations[i].lower()
                action = step_actions[i].lower()
                
                if "nothing" in obs or "cannot" in obs or "failed" in obs:
                    step_rewards[i] -= 0.05
                
                if action.count(" ") > 10:
                    step_rewards[i] -= 0.02
        
        step_rewards = np.clip(step_rewards, -0.2, 1.0)
        
        return step_rewards