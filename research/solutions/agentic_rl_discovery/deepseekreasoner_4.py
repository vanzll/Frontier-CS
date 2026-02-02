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
            "normalize_advantages": True,
            "reward_shaping": True,
            "step_grouping": True,
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
        
        if step_rewards is None:
            step_rewards = token_level_rewards
        
        step_rewards_flat = step_rewards.view(-1)
        mask_flat = response_mask.view(-1)
        
        rewards = torch.zeros_like(step_rewards_flat)
        episode_starts = []
        
        unique_episodes = np.unique(episode_index)
        for ep_idx in unique_episodes:
            ep_mask = (episode_index.flatten() == ep_idx) & (mask_flat.cpu().numpy() == 1)
            if not ep_mask.any():
                continue
            
            ep_rewards = step_rewards_flat[ep_mask].cpu().numpy()
            ep_positions = np.where(ep_mask)[0]
            
            if self.config["step_grouping"] and anchor_observations is not None:
                anchor_flat = anchor_observations.flatten()
                unique_anchors, anchor_indices = np.unique(
                    anchor_flat[ep_mask], return_inverse=True
                )
                
                anchor_rewards = []
                for a_idx in range(len(unique_anchors)):
                    anchor_mask = anchor_indices == a_idx
                    if anchor_mask.any():
                        anchor_rewards.append(ep_rewards[anchor_mask].sum())
                    else:
                        anchor_rewards.append(0.0)
                
                anchor_returns = self._compute_discounted_returns(
                    np.array(anchor_rewards), gamma
                )
                
                for a_idx, ret in enumerate(anchor_returns):
                    step_mask = anchor_indices == a_idx
                    if step_mask.any():
                        avg_return = ret / max(1.0, step_mask.sum())
                        rewards[ep_positions[step_mask]] = avg_return
            else:
                ep_returns = self._compute_discounted_returns(ep_rewards, gamma)
                rewards[ep_positions] = torch.from_numpy(ep_returns).to(device)
        
        returns = rewards.view(batch_size, seq_len)
        
        if self.config["normalize_advantages"]:
            advantages = self._normalize_within_groups(
                returns, episode_index, response_mask
            )
        else:
            advantages = returns
        
        advantages = advantages * response_mask
        returns = returns * response_mask
        
        if self.config.get("gae_lambda", 0.95) > 0:
            advantages = self._compute_gae(
                step_rewards, returns, response_mask, 
                gamma, self.config["gae_lambda"]
            )
        
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
        ratio = torch.exp(log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        
        policy_loss = -torch.min(surrogate1, surrogate2)
        
        mask = response_mask.bool()
        policy_loss = policy_loss[mask].mean()
        
        clip_fraction = ((ratio < 1 - clip_ratio) | (ratio > 1 + clip_ratio)).float()
        clip_fraction = clip_fraction[mask].mean()
        
        loss = policy_loss
        
        if self.config["use_kl_loss"]:
            kl = old_log_prob - log_prob
            kl_penalty = (kl ** 2).mean() * self.config["kl_loss_coef"]
            loss = loss + kl_penalty
        
        metrics = {
            "clip_frac": clip_fraction.item(),
            "policy_loss": policy_loss.item(),
            "approx_kl": (0.5 * (log_prob - old_log_prob) ** 2)[mask].mean().item(),
        }
        
        if self.config["use_kl_loss"]:
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
        step_rewards = np.zeros(trajectory_length)
        
        if episode_reward <= 0:
            return step_rewards
        
        if not self.config["reward_shaping"]:
            step_rewards[-1] = episode_reward
            return step_rewards
        
        success_bonus = episode_reward
        step_weight = success_bonus / max(1, trajectory_length)
        
        base_reward = step_weight * 0.3
        step_rewards[:] = base_reward
        
        action_progress = self._compute_action_progress(step_actions)
        observation_progress = self._compute_observation_progress(step_observations)
        
        progress_scores = 0.5 * action_progress + 0.5 * observation_progress
        
        total_progress = progress_scores.sum()
        if total_progress > 0:
            progress_rewards = (progress_scores / total_progress) * success_bonus * 0.7
            step_rewards += progress_rewards
        
        if trajectory_length > 0:
            step_rewards[-1] += success_bonus * 0.3
        
        step_rewards = np.clip(step_rewards, 0, 1)
        
        return step_rewards

    def _compute_discounted_returns(
        self, rewards: np.ndarray, gamma: float
    ) -> np.ndarray:
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0.0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        return returns

    def _normalize_within_groups(
        self, 
        values: torch.Tensor, 
        episode_index: np.ndarray, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len = values.shape
        values_flat = values.view(-1)
        mask_flat = mask.view(-1).cpu().numpy()
        episode_flat = episode_index.flatten()
        
        normalized = torch.zeros_like(values_flat)
        
        unique_episodes = np.unique(episode_flat[mask_flat == 1])
        for ep_idx in unique_episodes:
            ep_mask = (episode_flat == ep_idx) & (mask_flat == 1)
            if ep_mask.sum() > 1:
                ep_values = values_flat[ep_mask]
                ep_mean = ep_values.mean()
                ep_std = ep_values.std() + 1e-8
                normalized[ep_mask] = (ep_values - ep_mean) / ep_std
            elif ep_mask.sum() == 1:
                normalized[ep_mask] = 0.0
        
        return normalized.view(batch_size, seq_len)

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor,
        gamma: float,
        gae_lambda: float
    ) -> torch.Tensor:
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        
        for b in range(batch_size):
            episode_rewards = rewards[b][mask[b].bool()].cpu().numpy()
            episode_returns = returns[b][mask[b].bool()].cpu().numpy()
            
            if len(episode_rewards) == 0:
                continue
            
            gae = 0
            for t in reversed(range(len(episode_rewards))):
                if t == len(episode_rewards) - 1:
                    delta = episode_rewards[t] - episode_returns[t]
                else:
                    delta = episode_rewards[t] + gamma * episode_returns[t + 1] - episode_returns[t]
                gae = delta + gamma * gae_lambda * gae
                
                advantage_idx = torch.where(mask[b])[0][t]
                advantages[b, advantage_idx] = gae
        
        return advantages

    def _compute_action_progress(self, actions: list) -> np.ndarray:
        scores = np.zeros(len(actions))
        
        action_keywords = {
            'go': 0.1,
            'take': 0.2,
            'put': 0.3,
            'open': 0.1,
            'close': 0.1,
            'use': 0.2
        }
        
        for i, action in enumerate(actions):
            action_lower = action.lower()
            for keyword, score in action_keywords.items():
                if keyword in action_lower:
                    scores[i] += score
        
        cumulative = np.cumsum(scores)
        if cumulative[-1] > 0:
            scores = cumulative / cumulative[-1]
        
        return scores

    def _compute_observation_progress(self, observations: list) -> np.ndarray:
        scores = np.zeros(len(observations))
        
        progress_indicators = {
            'clean': 0.3,
            'dirty': -0.1,
            'in': 0.2,
            'on': 0.1,
            'closed': -0.1,
            'open': 0.1
        }
        
        for i, obs in enumerate(observations):
            obs_lower = obs.lower()
            for indicator, score in progress_indicators.items():
                if indicator in obs_lower:
                    scores[i] += score
        
        scores = np.maximum(scores, 0)
        cumulative = np.cumsum(scores)
        if cumulative[-1] > 0:
            scores = cumulative / cumulative[-1]
        
        return scores