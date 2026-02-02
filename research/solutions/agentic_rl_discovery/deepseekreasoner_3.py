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
            "use_gae": True,
            "gae_lambda": 0.95,
            "reward_norm": True,
            "baseline_type": "group_mean",
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
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device

        if step_rewards is not None and step_rewards.ndim == 1:
            step_rewards = step_rewards.unsqueeze(-1)

        rewards_to_use = step_rewards if step_rewards is not None else token_level_rewards

        returns = torch.zeros_like(rewards_to_use)
        advantages = torch.zeros_like(rewards_to_use)

        unique_episodes = np.unique(episode_index)
        episode_advantages = []

        for ep_idx in unique_episodes:
            ep_mask = torch.tensor(episode_index == ep_idx, device=device)
            ep_trajectories = np.unique(trajectory_index[episode_index == ep_idx])

            if anchor_observations is not None and self.config.get("step_grouping", True):
                step_groups = self._group_by_anchor_observations(
                    anchor_observations[episode_index == ep_idx]
                )
                for group_idx in step_groups:
                    group_mask = ep_mask.clone()
                    group_indices = step_groups[group_idx]
                    full_group_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                    ep_indices = np.where(episode_index == ep_idx)[0]
                    for local_idx in group_indices:
                        if local_idx < len(ep_indices):
                            full_group_mask[ep_indices[local_idx]] = True
                    group_mask = group_mask & full_group_mask

                    if group_mask.any():
                        group_rewards = rewards_to_use[group_mask]
                        group_returns, group_advantages = self._compute_episode_advantages(
                            group_rewards, gamma
                        )
                        returns[group_mask] = group_returns
                        advantages[group_mask] = group_advantages
                        episode_advantages.append(group_advantages.mean().item())
            else:
                ep_rewards = rewards_to_use[ep_mask]
                ep_returns, ep_advantages = self._compute_episode_advantages(ep_rewards, gamma)
                returns[ep_mask] = ep_returns
                advantages[ep_mask] = ep_advantages
                episode_advantages.append(ep_advantages.mean().item())

        if self.config.get("reward_norm", True):
            if episode_advantages:
                adv_mean = np.mean(episode_advantages)
                adv_std = np.std(episode_advantages) + 1e-8
                advantages = (advantages - adv_mean) / adv_std

        if step_rewards is not None and step_rewards.shape != token_level_rewards.shape:
            advantages_expanded = torch.zeros_like(token_level_rewards)
            returns_expanded = torch.zeros_like(token_level_rewards)
            for i in range(batch_size):
                traj_idx = trajectory_index[i]
                if traj_idx < len(advantages[i]):
                    advantages_expanded[i] = advantages[i, traj_idx]
                    returns_expanded[i] = returns[i, traj_idx]
            advantages = advantages_expanded
            returns = returns_expanded

        advantages = advantages * response_mask
        returns = returns * response_mask

        return advantages, returns

    def _compute_episode_advantages(
        self, rewards: torch.Tensor, gamma: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.get("use_gae", True):
            return self._compute_gae(rewards, gamma)
        else:
            return self._compute_monte_carlo(rewards, gamma)

    def _compute_monte_carlo(
        self, rewards: torch.Tensor, gamma: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = rewards.shape[0]
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        running_return = 0
        for t in reversed(range(T)):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return

        baseline = returns.mean()
        advantages = returns - baseline

        return returns, advantages

    def _compute_gae(
        self, rewards: torch.Tensor, gamma: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = rewards.shape[0]
        gae_lambda = self.config.get("gae_lambda", 0.95)

        values = torch.zeros(T + 1, device=rewards.device)
        advantages = torch.zeros(T, device=rewards.device)
        returns = torch.zeros(T, device=rewards.device)

        last_gae_lam = 0
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * last_gae_lam

        returns = advantages + values[:-1]

        baseline = values[:-1].mean()
        advantages = returns - baseline

        return returns, advantages

    def _group_by_anchor_observations(self, anchor_obs: np.ndarray) -> Dict[int, List[int]]:
        groups = defaultdict(list)
        for idx, obs in enumerate(anchor_obs):
            obs_hash = hash(str(obs)) % 100000
            groups[obs_hash].append(idx)
        return dict(groups)

    def compute_policy_loss(
        self,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        clip_ratio: float = 0.2,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        device = log_prob.device
        clip_ratio = self.config.get("clip_ratio", clip_ratio)

        ratio = torch.exp(log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

        policy_loss1 = ratio * advantages
        policy_loss2 = clipped_ratio * advantages
        policy_loss = -torch.min(policy_loss1, policy_loss2)

        policy_loss = (policy_loss * response_mask).sum() / (response_mask.sum() + 1e-8)

        clip_fraction = ((ratio - 1).abs() > clip_ratio).float()
        clip_fraction = (clip_fraction * response_mask).sum() / (response_mask.sum() + 1e-8)

        metrics = {"clip_fraction": clip_fraction.item()}

        if self.config.get("use_kl_loss", False):
            kl_loss = self._compute_kl_loss(old_log_prob, log_prob, response_mask)
            policy_loss = policy_loss + self.config.get("kl_loss_coef", 0.01) * kl_loss
            metrics["kl_loss"] = kl_loss.item()

        return policy_loss, metrics

    def _compute_kl_loss(
        self, old_log_prob: torch.Tensor, log_prob: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        kl_div = old_log_prob - log_prob
        kl_loss = (kl_div * mask).sum() / (mask.sum() + 1e-8)
        return kl_loss

    def assign_step_rewards(
        self,
        episode_reward: float,
        trajectory_length: int,
        step_observations: list,
        step_actions: list,
        **kwargs,
    ) -> np.ndarray:
        step_rewards = np.zeros(trajectory_length)

        if episode_reward > 0:
            step_rewards[-1] = episode_reward

            success_bonus = 0.1
            progress_bonus = 0.05

            for i in range(trajectory_length - 1):
                if self._is_progress_step(step_observations[i], step_actions[i]):
                    step_rewards[i] += progress_bonus

            step_rewards[-1] += success_bonus

        return step_rewards

    def _is_progress_step(self, observation: str, action: str) -> bool:
        progress_indicators = [
            "find", "take", "put", "open", "close", "go", "look",
            "clean", "heat", "cool", "slice", "turn"
        ]
        action_lower = action.lower()
        return any(indicator in action_lower for indicator in progress_indicators)