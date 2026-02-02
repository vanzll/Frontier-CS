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
            "gae_lambda": 0.95,
            "step_grouping": True,
            "group_normalize": True,
            "clip_value_loss": True,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
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
        device = token_level_rewards.device
        batch_size, seq_len = token_level_rewards.shape
        
        rewards = step_rewards if step_rewards is not None else token_level_rewards
        rewards = rewards * response_mask
        
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        episode_index = torch.from_numpy(episode_index).to(device)
        trajectory_index = torch.from_numpy(trajectory_index).to(device)
        
        unique_episodes = torch.unique(episode_index)
        
        for ep_idx in unique_episodes:
            ep_mask = (episode_index == ep_idx)
            ep_indices = torch.where(ep_mask)[0]
            
            if self.config.get("step_grouping", False) and anchor_observations is not None:
                advantages_ep = self._compute_gae_step_grouped(
                    rewards[ep_mask], response_mask[ep_mask],
                    anchor_observations[ep_idx.cpu().numpy()] if ep_idx < len(anchor_observations) else None,
                    gamma, self.config.get("gae_lambda", 0.95)
                )
                advantages[ep_mask] = advantages_ep
                
                ep_trajectories = torch.unique(trajectory_index[ep_mask])
                for traj_idx in ep_trajectories:
                    traj_mask = ep_mask & (trajectory_index == traj_idx)
                    returns_traj = self._compute_returns(
                        rewards[traj_mask], response_mask[traj_mask], gamma
                    )
                    returns[traj_mask] = returns_traj
            else:
                advantages_ep = self._compute_gae_group_normalized(
                    rewards[ep_mask], response_mask[ep_mask], gamma,
                    self.config.get("gae_lambda", 0.95),
                    self.config.get("group_normalize", True)
                )
                advantages[ep_mask] = advantages_ep
                
                ep_trajectories = torch.unique(trajectory_index[ep_mask])
                for traj_idx in ep_trajectories:
                    traj_mask = ep_mask & (trajectory_index == traj_idx)
                    returns_traj = self._compute_returns(
                        rewards[traj_mask], response_mask[traj_mask], gamma
                    )
                    returns[traj_mask] = returns_traj
        
        if self.config.get("group_normalize", True) and not self.config.get("step_grouping", False):
            for ep_idx in unique_episodes:
                ep_mask = (episode_index == ep_idx)
                if ep_mask.sum() > 1:
                    ep_advantages = advantages[ep_mask]
                    ep_valid = response_mask[ep_mask].bool()
                    valid_advantages = ep_advantages[ep_valid]
                    if valid_advantages.numel() > 0:
                        mean = valid_advantages.mean()
                        std = valid_advantages.std() + 1e-8
                        ep_advantages[ep_valid] = (valid_advantages - mean) / std
                        advantages[ep_mask] = ep_advantages
        
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
        device = old_log_prob.device
        response_mask = response_mask.float()
        
        ratio = torch.exp(log_prob - old_log_prob)
        ratio_masked = ratio * response_mask
        
        adv_clipped = advantages.clone()
        if self.config.get("group_normalize", True):
            adv_clipped = torch.clamp(adv_clipped, -10.0, 10.0)
        
        surr1 = ratio_masked * adv_clipped
        surr2 = torch.clamp(ratio_masked, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_clipped
        policy_loss = -torch.min(surr1, surr2)
        
        valid_mask = response_mask > 0
        if valid_mask.any():
            policy_loss = policy_loss[valid_mask].mean()
        else:
            policy_loss = torch.tensor(0.0, device=device)
        
        clip_fraction = ((ratio_masked > (1.0 + clip_ratio)) | 
                        (ratio_masked < (1.0 - clip_ratio))).float()
        clip_fraction = (clip_fraction * response_mask).sum() / (response_mask.sum() + 1e-8)
        
        metrics = {
            "clip_frac": clip_fraction.item(),
            "ratio_mean": ratio_masked[valid_mask].mean().item() if valid_mask.any() else 0.0,
            "policy_loss": policy_loss.item(),
        }
        
        if self.config.get("use_kl_loss", False):
            kl = old_log_prob - log_prob
            kl_loss = (kl * response_mask).mean()
            total_loss = policy_loss + self.config["kl_loss_coef"] * kl_loss
            metrics["kl_loss"] = kl_loss.item()
            metrics["total_loss"] = total_loss.item()
        else:
            total_loss = policy_loss
        
        if "values" in kwargs and "returns" in kwargs:
            values = kwargs["values"]
            returns = kwargs["returns"]
            value_loss = self._compute_value_loss(values, returns, response_mask)
            total_loss = total_loss + self.config.get("value_loss_coef", 0.5) * value_loss
            metrics["value_loss"] = value_loss.item()
        
        if "entropy" in kwargs:
            entropy = kwargs["entropy"]
            entropy_bonus = (entropy * response_mask).mean()
            total_loss = total_loss - self.config.get("entropy_coef", 0.01) * entropy_bonus
            metrics["entropy"] = entropy_bonus.item()
        
        return total_loss, metrics

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
            success_bonus = 1.0 / trajectory_length
            step_rewards[:] = success_bonus
            
            last_action_idx = -1
            for i in range(trajectory_length - 1, -1, -1):
                if step_actions[i].strip():
                    last_action_idx = i
                    break
            
            if last_action_idx >= 0:
                step_rewards[last_action_idx] += 0.5
                
            for i in range(trajectory_length):
                action = step_actions[i].lower()
                obs = step_observations[i].lower()
                
                if "put" in action and "in/on" in action:
                    step_rewards[i] += 0.3
                if "open" in action and "close" not in obs:
                    step_rewards[i] += 0.1
                if "take" in action and "from" in action:
                    step_rewards[i] += 0.2
        else:
            failure_penalty = -0.5 / trajectory_length
            step_rewards[:] = failure_penalty
            
            for i in range(trajectory_length):
                action = step_actions[i].lower()
                if "put" in action and "in/on" in action:
                    step_rewards[i] -= 0.2
                if i > trajectory_length * 0.8:
                    step_rewards[i] -= 0.1
        
        step_rewards = np.clip(step_rewards, -1.0, 1.0)
        return step_rewards

    def _compute_returns(
        self,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        gamma: float
    ) -> torch.Tensor:
        device = rewards.device
        returns = torch.zeros_like(rewards)
        running_return = torch.tensor(0.0, device=device)
        
        for t in reversed(range(rewards.size(0))):
            if mask[t] > 0:
                running_return = rewards[t] + gamma * running_return
                returns[t] = running_return
            else:
                running_return = torch.tensor(0.0, device=device)
        
        return returns

    def _compute_gae_group_normalized(
        self,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        normalize: bool = True
    ) -> torch.Tensor:
        device = rewards.device
        advantages = torch.zeros_like(rewards)
        last_gae_lam = torch.tensor(0.0, device=device)
        
        for t in reversed(range(rewards.size(0))):
            if t == rewards.size(0) - 1:
                next_value = torch.tensor(0.0, device=device)
            else:
                next_value = advantages[t + 1] if mask[t + 1] > 0 else torch.tensor(0.0, device=device)
            
            if mask[t] > 0:
                delta = rewards[t] + gamma * next_value - advantages[t]
                last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
                advantages[t] = last_gae_lam
            else:
                last_gae_lam = torch.tensor(0.0, device=device)
        
        if normalize and advantages[mask > 0].numel() > 0:
            valid_adv = advantages[mask > 0]
            mean = valid_adv.mean()
            std = valid_adv.std() + 1e-8
            advantages[mask > 0] = (valid_adv - mean) / std
        
        return advantages

    def _compute_gae_step_grouped(
        self,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        anchor_observations: Optional[np.ndarray],
        gamma: float,
        gae_lambda: float
    ) -> torch.Tensor:
        device = rewards.device
        seq_len = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        
        if anchor_observations is None or len(anchor_observations) == 0:
            return self._compute_gae_group_normalized(rewards, mask, gamma, gae_lambda, False)
        
        unique_anchors = {}
        for t in range(seq_len):
            if mask[t] > 0 and t < len(anchor_observations):
                anchor = anchor_observations[t]
                anchor_key = str(anchor)
                if anchor_key not in unique_anchors:
                    unique_anchors[anchor_key] = []
                unique_anchors[anchor_key].append(t)
        
        for anchor_key, timesteps in unique_anchors.items():
            if len(timesteps) > 1:
                anchor_rewards = rewards[timesteps]
                anchor_advantages = self._compute_gae_group_normalized(
                    anchor_rewards, 
                    torch.ones_like(anchor_rewards),
                    gamma, gae_lambda, True
                )
                for idx, t in enumerate(timesteps):
                    advantages[t] = anchor_advantages[idx]
        
        missing_mask = (mask > 0) & (advantages == 0)
        if missing_mask.any():
            missing_adv = self._compute_gae_group_normalized(
                rewards[missing_mask],
                mask[missing_mask],
                gamma, gae_lambda, True
            )
            advantages[missing_mask] = missing_adv
        
        return advantages

    def _compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        value_pred_clipped = values.clone()
        value_losses = (values - returns) ** 2
        value_losses_clipped = (value_pred_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        value_loss = (value_loss * mask).sum() / (mask.sum() + 1e-8)
        return value_loss