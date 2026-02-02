import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional

try:
    from verl.utils.torch_functional import masked_mean, masked_whiten
except ImportError:

    def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean of a tensor over the masked elements.
        """
        return (tensor * mask).sum() / mask.sum().clamp(min=1e-9)

    def masked_whiten(tensor: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
        """
        Whiten a tensor with a mask.
        """
        tensor_masked = tensor * mask
        num_elements = torch.sum(mask)
        if num_elements > 1:
            mean = torch.sum(tensor_masked) / num_elements if shift_mean else 0.0
            var = torch.sum(torch.pow(tensor_masked - mean, 2) * mask) / num_elements
            std = torch.sqrt(var + 1e-8)
            whitened = (tensor - mean) / std
            return whitened * mask
        return tensor


class Solution:
    """
    Implements a GiGPO/GRPO-style reinforcement learning algorithm for LLM agents.

    The algorithm is based on the following principles:
    1.  **Advantage Computation**: Uses a sophisticated group-based baseline inspired by
        GiGPO and GRPO to reduce variance. It groups steps by their underlying
        state representations (`anchor_observations`) to compute a leave-one-out
        mean return as a baseline. This provides a step-specific advantage estimate.
        It falls back to trajectory-level grouping if state representations are unavailable.
        Advantages are then whitened for stability.

    2.  **Policy Loss**: Employs the PPO-clip objective function, a standard in modern
        RL, to ensure stable policy updates by preventing excessively large changes.
        An optional KL-divergence penalty term is added to the loss to further
        regularize the policy updates and prevent divergence from the behavior policy.

    3.  **Reward Assignment**: Uses a sparse, last-step reward assignment strategy.
        For successful episodes, the full reward is attributed to the final action,
        which is a robust approach for tasks with delayed rewards like ALFWorld.
    """

    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm and set hyperparameters.
        """
        self.config = {
            "gamma": 0.99,
            "clip_ratio": 0.2,
            "use_kl_loss": True,
            "kl_loss_coef": 0.02,
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
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes advantages and returns using a GiGPO-style group-based baseline.
        """
        batch_size, seq_len = response_mask.shape

        step_level_rewards = torch.sum(token_level_rewards, dim=-1)

        ep_traj_ids = np.stack([episode_index, trajectory_index], axis=1)
        is_new_traj = np.concatenate(
            [[True], np.any(ep_traj_ids[1:] != ep_traj_ids[:-1], axis=1)]
        )
        traj_starts = np.where(is_new_traj)[0]
        traj_lengths = np.diff(np.append(traj_starts, batch_size))

        discounted_returns = torch.zeros_like(step_level_rewards)
        for start, length in zip(traj_starts, traj_lengths):
            rewards_traj = step_level_rewards[start : start + length]
            G = 0.0
            for t in reversed(range(length)):
                G = rewards_traj[t] + gamma * G
                discounted_returns[start + t] = G

        baselines = torch.zeros_like(step_level_rewards)
        if anchor_observations is not None and len(anchor_observations) == batch_size:
            anchor_obs_hashable = [obs.tobytes() for obs in anchor_observations]
            grouped_returns = defaultdict(lambda: {"sum": 0.0, "count": 0})
            for i, obs_hash in enumerate(anchor_obs_hashable):
                grouped_returns[obs_hash]["sum"] += discounted_returns[i].item()
                grouped_returns[obs_hash]["count"] += 1

            for i, obs_hash in enumerate(anchor_obs_hashable):
                group = grouped_returns[obs_hash]
                if group["count"] > 1:
                    baselines[i] = (
                        group["sum"] - discounted_returns[i].item()
                    ) / (group["count"] - 1)
        else:
            traj_total_returns = [
                discounted_returns[start].item() for start in traj_starts
            ]
            ep_groups = defaultdict(list)
            for i, start in enumerate(traj_starts):
                ep_id = episode_index[start]
                ep_groups[ep_id].append(traj_total_returns[i])

            traj_baselines = np.zeros(len(traj_starts))
            for i, start in enumerate(traj_starts):
                ep_id = episode_index[start]
                group_returns = ep_groups[ep_id]
                my_return = traj_total_returns[i]
                if len(group_returns) > 1:
                    traj_baselines[i] = (sum(group_returns) - my_return) / (
                        len(group_returns) - 1
                    )

            for i, (start, length) in enumerate(zip(traj_starts, traj_lengths)):
                baselines[start : start + length] = traj_baselines[i]

        step_advantages = discounted_returns - baselines

        step_mask = response_mask.sum(dim=-1) > 0
        step_advantages = masked_whiten(step_advantages, step_mask)

        advantages = (
            step_advantages.unsqueeze(1).expand(-1, seq_len) * response_mask
        )
        returns = (
            discounted_returns.unsqueeze(1).expand(-1, seq_len) * response_mask
        )

        return advantages, returns

    def compute_policy_loss(
        self,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        clip_ratio: float = 0.2,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes the PPO-clip policy loss with an optional KL penalty.
        """
        log_ratio = log_prob - old_log_prob
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages

        policy_objective = torch.min(surr1, surr2)
        policy_loss = -masked_mean(policy_objective, response_mask)

        metrics = {}
        if self.config.get("use_kl_loss", False):
            kl_div = ratio - 1 - log_ratio
            kl_loss = masked_mean(kl_div, response_mask)
            policy_loss += self.config.get("kl_loss_coef", 0.01) * kl_loss
            metrics["kl_loss"] = kl_loss.item()

        with torch.no_grad():
            clipped = (ratio > 1.0 + clip_ratio) | (ratio < 1.0 - clip_ratio)
            masked_clipped = clipped & response_mask.bool()
            clip_frac = torch.sum(masked_clipped.float()) / torch.sum(
                response_mask.float()
            )
            approx_kl = masked_mean((ratio - 1) - log_ratio, response_mask)

            metrics["clip_frac"] = clip_frac.item()
            metrics["approx_kl"] = approx_kl.item()

        return policy_loss, metrics

    def assign_step_rewards(
        self,
        episode_reward: float,
        trajectory_length: int,
        step_observations: list,
        step_actions: list,
        **kwargs,
    ) -> np.ndarray:
        """
        Assigns rewards to the steps of a trajectory using a sparse, last-step scheme.
        """
        step_rewards = np.zeros(trajectory_length, dtype=np.float32)
        if episode_reward > 0:
            step_rewards[-1] = episode_reward
        return step_rewards