import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm. Called once before training.
        """
        self.config: Dict[str, Any] = {
            "gamma": 0.95,          # Discount factor for token-level returns
            "clip_ratio": 0.2,      # PPO clip range
            "use_kl_loss": False,   # Add KL penalty
            "kl_loss_coef": 0.01,   # KL penalty coefficient
            # Additional internal hyperparameters
            "step_reward_discount": 0.97,  # Discount for distributing episode reward over steps
        }
        return self

    def compute_advantage(
        self,
        token_level_rewards: torch.Tensor,   # (batch, seq_len)
        response_mask: torch.Tensor,         # (batch, seq_len)
        episode_index: np.ndarray,           # (batch,) - episode group IDs
        trajectory_index: np.ndarray,        # (batch,) - trajectory IDs within group (unused)
        step_rewards: Optional[torch.Tensor] = None,
        anchor_observations: Optional[np.ndarray] = None,  # unused
        gamma: float = 0.95,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns for policy update.
        """
        device = token_level_rewards.device
        dtype = token_level_rewards.dtype

        # Resolve gamma: argument > config > default
        if gamma is None:
            gamma = float(self.config.get("gamma", 0.95))
        else:
            gamma = float(gamma)

        rewards = token_level_rewards.clone()

        # Ensure mask is float for arithmetic and bool for indexing
        mask_float = response_mask.to(dtype)
        mask_bool = response_mask.bool()

        # Incorporate optional step-level rewards if provided
        if step_rewards is not None:
            if isinstance(step_rewards, torch.Tensor):
                sr = step_rewards.to(device=device, dtype=dtype)
            else:
                sr = torch.tensor(step_rewards, device=device, dtype=dtype)

            if sr.shape == rewards.shape:
                # Already token-aligned
                rewards = rewards + sr
            else:
                step_sum = None
                if sr.dim() == 2 and sr.shape[0] == rewards.shape[0]:
                    # (batch, num_steps)
                    step_sum = sr.sum(dim=-1)  # (batch,)
                elif sr.dim() == 1 and sr.shape[0] == rewards.shape[0]:
                    # (batch,)
                    step_sum = sr
                # Distribute per-trajectory step rewards uniformly over valid response tokens
                if step_sum is not None:
                    token_counts = mask_float.sum(dim=-1).clamp_min(1.0)  # (batch,)
                    per_token_bonus = (step_sum / token_counts).unsqueeze(-1)  # (batch, 1)
                    rewards = rewards + per_token_bonus * mask_float

        # Only count rewards on valid response tokens
        rewards = rewards * mask_float

        batch_size, seq_len = rewards.shape

        # Compute discounted returns over tokens: R_t = r_t + gamma * R_{t+1}
        returns = torch.zeros_like(rewards)
        running = torch.zeros(batch_size, device=device, dtype=dtype)
        for t in range(seq_len - 1, -1, -1):
            running = rewards[:, t] + gamma * running
            returns[:, t] = running

        # Zero-out returns on invalid tokens for cleanliness
        returns = returns * mask_float

        # Sequence-level scalar return for each trajectory: return from first valid token
        # This is used for group-wise baselines (RLOO-style).
        if mask_bool.any():
            cumsum = mask_bool.to(torch.int64).cumsum(dim=-1)
            first_pos_mask = (cumsum == 1) & mask_bool  # True only at first valid token
            seq_return = (returns * first_pos_mask.to(dtype)).sum(dim=-1)  # (batch,)
        else:
            seq_return = torch.zeros(batch_size, device=device, dtype=dtype)

        # Group-wise leave-one-out baselines using episode_index
        episode_index_np = np.asarray(episode_index)
        unique_episodes = np.unique(episode_index_np)

        baseline_seq = torch.zeros_like(seq_return)
        for ep_id in unique_episodes:
            idx_np = np.nonzero(episode_index_np == ep_id)[0]
            if idx_np.size == 0:
                continue
            idx = torch.as_tensor(idx_np, device=device, dtype=torch.long)
            group_scores = seq_return[idx]  # (group_size,)
            group_size = group_scores.shape[0]

            if group_size > 1:
                sum_scores = group_scores.sum()
                # Leave-one-out baseline for each trajectory
                baseline_vals = (sum_scores - group_scores) / (group_size - 1)
            else:
                baseline_vals = torch.zeros_like(group_scores)

            baseline_seq[idx] = baseline_vals

        # Broadcast sequence baseline to token level
        baseline_tokens = baseline_seq.unsqueeze(-1).expand_as(returns)

        # Advantage per token
        advantages = returns - baseline_tokens

        # Only keep advantages/returns on valid tokens
        advantages = advantages * mask_float
        returns = returns * mask_float

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
        Compute PPO-style clipped policy gradient loss with optional KL penalty.
        """
        device = log_prob.device
        dtype = log_prob.dtype

        # Resolve clip_ratio: argument > config > default
        if clip_ratio is None:
            clip_ratio = float(self.config.get("clip_ratio", 0.2))
        else:
            clip_ratio = float(clip_ratio)

        mask_bool = response_mask.bool()
        mask_float = response_mask.to(dtype)

        # Detach advantages to avoid backprop through advantage computation
        adv = advantages.detach()

        # Normalize advantages over valid tokens
        adv_flat = adv[mask_bool]
        if adv_flat.numel() > 0:
            adv_mean = adv_flat.mean()
            adv_std = adv_flat.std(unbiased=False).clamp_min(1e-8)
            adv_norm = torch.zeros_like(adv)
            adv_norm[mask_bool] = (adv_flat - adv_mean) / adv_std
        else:
            adv_mean = torch.tensor(0.0, device=device, dtype=dtype)
            adv_std = torch.tensor(1.0, device=device, dtype=dtype)
            adv_norm = torch.zeros_like(adv)

        # Probability ratio
        log_ratio = log_prob - old_log_prob
        ratio = torch.exp(log_ratio)

        # Surrogate objective
        surr1 = ratio * adv_norm
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_norm
        objective = torch.min(surr1, surr2)

        # Masked mean over valid tokens
        masked_obj = objective * mask_float
        denom = mask_float.sum().clamp_min(1.0)
        loss = -masked_obj.sum() / denom

        # Approximate KL divergence (token-level)
        approx_kl = 0.5 * (log_ratio ** 2)
        kl_masked = approx_kl * mask_float
        kl_mean = kl_masked.sum() / denom

        # Optional KL penalty
        if self.config.get("use_kl_loss", False):
            kl_coef = float(self.config.get("kl_loss_coef", 0.01))
            loss = loss + kl_coef * kl_mean

        # Clip fraction: fraction of tokens where ratio was clipped
        clipped = (torch.abs(ratio - 1.0) > clip_ratio).to(dtype) * mask_float
        clip_frac = clipped.sum() / denom

        # Metrics for logging
        metrics: Dict[str, Any] = {}
        metrics["policy_loss"] = float(loss.detach().cpu().item())
        metrics["approx_kl"] = float(kl_mean.detach().cpu().item())
        metrics["clip_frac"] = float(clip_frac.detach().cpu().item())
        if adv_flat.numel() > 0:
            metrics["adv_mean"] = float(adv_flat.mean().detach().cpu().item())
            metrics["adv_std"] = float(adv_flat.std(unbiased=False).detach().cpu().item())
        else:
            metrics["adv_mean"] = 0.0
            metrics["adv_std"] = 0.0

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
        Distribute episode reward to individual steps with a time-biased scheme
        and simple heuristic penalties for obviously unproductive actions.
        """
        # Handle edge cases
        if trajectory_length <= 0:
            return np.zeros(0, dtype=np.float32)

        # No reward if episode failed (ALFWorld uses 0/1)
        if episode_reward <= 0.0:
            return np.zeros(trajectory_length, dtype=np.float32)

        # Base temporal weighting: later steps receive more credit
        step_gamma = float(self.config.get("step_reward_discount", 0.97))
        indices = np.arange(trajectory_length, dtype=np.float32)
        # Weight_t = step_gamma^(L-1-t): increases toward the end of the episode
        base_weights = step_gamma ** (trajectory_length - 1 - indices)

        # Heuristic adjustments based on actions and observations
        weights = base_weights.copy()

        noop_prefixes = (
            "look",
            "inventory",
            "examine",
            "wait",
            "help",
            "hint",
            "restart",
            "undo",
        )

        for t in range(trajectory_length):
            # Extract current action and observation text
            action = ""
            if t < len(step_actions) and step_actions[t] is not None:
                action = str(step_actions[t]).strip().lower()

            obs = ""
            if t < len(step_observations) and step_observations[t] is not None:
                obs = str(step_observations[t]).strip().lower()

            # Down-weight obvious no-op or information-only actions
            if any(action.startswith(p) for p in noop_prefixes):
                weights[t] *= 0.3

            # Down-weight repeated identical actions
            if t > 0 and t < len(step_actions):
                prev_action = ""
                if step_actions[t - 1] is not None:
                    prev_action = str(step_actions[t - 1]).strip().lower()
                if action != "" and action == prev_action:
                    weights[t] *= 0.5

            # Down-weight clearly failed/invalid steps indicated in observation text
            if (
                "you can't" in obs
                or "you cannot" in obs
                or "can't go that way" in obs
                or "invalid" in obs
                or "is not possible" in obs
                or "not allowed" in obs
                or "don't understand" in obs
                or "do not understand" in obs
                or "i didn't understand" in obs
            ):
                weights[t] *= 0.2

        # Normalize weights so that sum of step rewards equals episode_reward
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            # Fallback to uniform distribution if all heuristic weights collapsed
            step_rewards = np.full(trajectory_length, episode_reward / float(trajectory_length), dtype=np.float32)
        else:
            step_rewards = (episode_reward * weights / total_weight).astype(np.float32)

        return step_rewards