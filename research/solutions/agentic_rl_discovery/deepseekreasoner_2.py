import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List
import math


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm. Called once before training.
        """
        self.config = {
            "gamma": 0.95,            # Discount factor
            "clip_ratio": 0.2,        # PPO clip range
            "use_kl_loss": False,     # Add KL penalty
            "kl_loss_coef": 0.01,     # KL penalty coefficient
            "gae_lambda": 0.95,       # GAE lambda
            "normalize_advantages": True,
            "value_bootstrap": True,
            "step_grouping": True,
            "shaping_reward_coef": 0.1,
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
        Compute advantages and returns using GAE with step-level grouping.
        """
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device
        
        # Combine sparse token rewards with step rewards if available
        if step_rewards is not None:
            rewards = token_level_rewards + step_rewards
        else:
            rewards = token_level_rewards
        
        # Compute returns and advantages with step-level grouping
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Get unique episode groups
        unique_episodes = np.unique(episode_index)
        
        for ep_idx in unique_episodes:
            # Get indices for this episode
            ep_mask = (episode_index == ep_idx)
            ep_trajectories = trajectory_index[ep_mask]
            
            # Get unique trajectories in this episode
            unique_trajectories = np.unique(ep_trajectories)
            
            for traj_idx in unique_trajectories:
                # Get mask for this trajectory
                traj_mask = ep_mask & (trajectory_index == traj_idx)
                traj_batch_indices = np.where(traj_mask)[0]
                
                if len(traj_batch_indices) == 0:
                    continue
                
                # Step-level grouping using anchor observations if available
                if anchor_observations is not None and self.config["step_grouping"]:
                    # Group by similar anchor observations (steps)
                    step_groups = self._group_by_anchor(
                        anchor_observations[traj_mask], traj_batch_indices)
                else:
                    # Treat entire trajectory as one group
                    step_groups = [traj_batch_indices]
                
                for step_indices in step_groups:
                    if len(step_indices) == 0:
                        continue
                    
                    # Compute advantages within this step group
                    for batch_idx in step_indices:
                        # Extract rewards and mask for this sequence
                        seq_rewards = rewards[batch_idx]
                        seq_mask = response_mask[batch_idx]
                        
                        # Compute returns using discounting
                        seq_returns = self._compute_returns(
                            seq_rewards, seq_mask, gamma)
                        returns[batch_idx] = seq_returns
                        
                        # Compute advantages using GAE
                        seq_advantages = self._compute_gae(
                            seq_rewards, seq_mask, gamma, 
                            self.config["gae_lambda"])
                        advantages[batch_idx] = seq_advantages
        
        # Normalize advantages within each episode group
        if self.config["normalize_advantages"]:
            advantages = self._normalize_advantages(
                advantages, response_mask, episode_index, trajectory_index)
        
        return advantages, returns
    
    def _group_by_anchor(
        self, 
        anchor_observations: np.ndarray, 
        batch_indices: np.ndarray
    ) -> List[np.ndarray]:
        """Group indices by similar anchor observations."""
        if len(anchor_observations) <= 1:
            return [batch_indices]
        
        # Simple grouping: treat each unique observation as a group
        groups = defaultdict(list)
        for i, obs in enumerate(anchor_observations):
            # Create hashable representation
            if isinstance(obs, str):
                obs_hash = obs
            else:
                obs_hash = str(obs)
            groups[obs_hash].append(batch_indices[i])
        
        return list(groups.values())
    
    def _compute_returns(
        self, 
        rewards: torch.Tensor, 
        mask: torch.Tensor,
        gamma: float
    ) -> torch.Tensor:
        """Compute discounted returns."""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        # Reverse cumulative sum
        for t in reversed(range(len(rewards))):
            if mask[t]:
                running_return = rewards[t] + gamma * running_return
                returns[t] = running_return
        
        return returns
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        gamma: float,
        gae_lambda: float
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        # Simple GAE without value function
        # Since we don't have value function, use returns as value baseline
        returns = self._compute_returns(rewards, mask, gamma)
        value_estimates = returns.clone()
        
        # Reverse computation
        for t in reversed(range(len(rewards))):
            if mask[t]:
                if t == len(rewards) - 1:
                    delta = rewards[t] - value_estimates[t]
                else:
                    next_value = value_estimates[t + 1] if mask[t + 1] else 0
                    delta = rewards[t] + gamma * next_value - value_estimates[t]
                
                last_advantage = delta + gamma * gae_lambda * last_advantage
                advantages[t] = last_advantage
        
        return advantages
    
    def _normalize_advantages(
        self,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        episode_index: np.ndarray,
        trajectory_index: np.ndarray
    ) -> torch.Tensor:
        """Normalize advantages within each episode-trajectory group."""
        normalized_advantages = advantages.clone()
        unique_episodes = np.unique(episode_index)
        
        for ep_idx in unique_episodes:
            ep_mask = (episode_index == ep_idx)
            ep_trajectories = trajectory_index[ep_mask]
            unique_trajectories = np.unique(ep_trajectories)
            
            for traj_idx in unique_trajectories:
                traj_mask = ep_mask & (trajectory_index == traj_idx)
                traj_batch_indices = np.where(traj_mask)[0]
                
                if len(traj_batch_indices) == 0:
                    continue
                
                # Collect all advantages in this trajectory
                traj_advantages = []
                for batch_idx in traj_batch_indices:
                    seq_adv = advantages[batch_idx]
                    seq_mask = response_mask[batch_idx]
                    masked_adv = seq_adv[seq_mask.bool()]
                    if len(masked_adv) > 0:
                        traj_advantages.append(masked_adv)
                
                if traj_advantages:
                    all_adv = torch.cat(traj_advantages)
                    if len(all_adv) > 1:
                        mean = all_adv.mean()
                        std = all_adv.std() + 1e-8
                        
                        # Normalize advantages in this trajectory
                        for batch_idx in traj_batch_indices:
                            mask_bool = response_mask[batch_idx].bool()
                            if mask_bool.any():
                                normalized_advantages[batch_idx][mask_bool] = (
                                    (advantages[batch_idx][mask_bool] - mean) / std
                                )
        
        return normalized_advantages

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
        Compute PPO-style policy loss with KL penalty.
        """
        device = log_prob.device
        mask = response_mask.bool()
        
        # Calculate probability ratio
        ratio = torch.exp(log_prob - old_log_prob)
        
        # PPO clipped loss
        adv_clamped = advantages.clone()
        adv_clamped = torch.clamp(adv_clamped, -10, 10)  # Clip extreme advantages
        
        surr1 = ratio * adv_clamped
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_clamped
        policy_loss = -torch.min(surr1, surr2)
        
        # Apply mask
        masked_loss = policy_loss * mask
        loss = masked_loss.sum() / (mask.sum() + 1e-8)
        
        # Add KL penalty if enabled
        if self.config.get("use_kl_loss", False):
            kl_loss = self._compute_kl_penalty(
                old_log_prob, log_prob, mask,
                self.config.get("kl_loss_coef", 0.01)
            )
            loss = loss + kl_loss
            kl_div = ((old_log_prob - log_prob) * mask).sum() / (mask.sum() + 1e-8)
        else:
            kl_div = torch.tensor(0.0, device=device)
        
        # Calculate metrics
        clipped = ((ratio > (1.0 + clip_ratio)) | 
                  (ratio < (1.0 - clip_ratio))) & mask
        clip_frac = clipped.sum().float() / (mask.sum() + 1e-8)
        
        metrics = {
            "clip_frac": clip_frac.item(),
            "kl_div": kl_div.item(),
            "advantages_mean": adv_clamped[mask].mean().item(),
            "advantages_std": adv_clamped[mask].std().item(),
            "ratio_mean": ratio[mask].mean().item(),
            "policy_loss": loss.item(),
        }
        
        return loss, metrics
    
    def _compute_kl_penalty(
        self,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        mask: torch.Tensor,
        coef: float
    ) -> torch.Tensor:
        """Compute KL divergence penalty."""
        kl = old_log_prob - log_prob
        masked_kl = kl * mask
        mean_kl = masked_kl.sum() / (mask.sum() + 1e-8)
        return coef * mean_kl

    def assign_step_rewards(
        self,
        episode_reward: float,
        trajectory_length: int,
        step_observations: list,
        step_actions: list,
        **kwargs
    ) -> np.ndarray:
        """
        Distribute episode reward with shaped rewards based on progress.
        """
        step_rewards = np.zeros(trajectory_length)
        
        if trajectory_length == 0:
            return step_rewards
        
        # Base sparse reward: assign to final step if successful
        if episode_reward > 0:
            step_rewards[-1] = 1.0
        
        # Add shaped rewards for progress
        if self.config.get("shaping_reward_coef", 0.0) > 0:
            shaping_rewards = self._compute_shaping_rewards(
                step_observations, step_actions, episode_reward
            )
            step_rewards += shaping_rewards
        
        # Normalize to preserve total reward
        total_shaped = step_rewards.sum()
        if total_shaped > 0:
            step_rewards = step_rewards * (episode_reward / total_shaped)
        
        return step_rewards
    
    def _compute_shaping_rewards(
        self,
        observations: List[str],
        actions: List[str],
        episode_reward: float
    ) -> np.ndarray:
        """Compute shaped rewards based on progress."""
        n_steps = len(observations)
        rewards = np.zeros(n_steps)
        
        if n_steps == 0:
            return rewards
        
        coef = self.config.get("shaping_reward_coef", 0.1)
        
        # Simple shaping: reward for taking actions that make progress
        for i in range(n_steps):
            obs = observations[i].lower()
            action = actions[i].lower()
            
            # Reward for executing meaningful actions
            if any(keyword in action for keyword in [
                "take", "put", "open", "close", "go", "look"
            ]):
                rewards[i] += 0.01 * coef
            
            # Reward for discovering objects
            if any(keyword in obs for keyword in [
                "apple", "fridge", "sink", "table", "counter"
            ]):
                rewards[i] += 0.02 * coef
            
            # Penalty for repetitive actions
            if i > 0 and action == actions[i-1].lower():
                rewards[i] -= 0.005 * coef
        
        # Scale by final success
        if episode_reward > 0:
            rewards *= 2.0
        
        return rewards