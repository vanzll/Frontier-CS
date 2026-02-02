import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm. Called once before training.
        """
        self.config = {
            "gamma": 0.95,
            "clip_ratio": 0.2,
            "use_kl_loss": False,
            "kl_loss_coef": 0.01,
        }
        return self

    def _compute_group_baseline(
        self,
        rewards: torch.Tensor,
        group_indices: Optional[np.ndarray],
    ) -> torch.Tensor:
        """
        Compute leave-one-out baseline per group given by group_indices.
        rewards: (batch,)
        group_indices: np.ndarray of shape (batch,)
        """
        device = rewards.device
        baseline = torch.zeros_like(rewards)

        if group_indices is None:
            return baseline

        if group_indices.shape[0] != rewards.shape[0]:
            return baseline

        unique_groups = np.unique(group_indices)
        for g in unique_groups:
            idx_np = np.nonzero(group_indices == g)[0]
            if idx_np.size == 0:
                continue
            idx_t = torch.as_tensor(idx_np, device=device, dtype=torch.long)
            group_rewards = rewards[idx_t]
            n = idx_t.numel()
            if n <= 1:
                baseline[idx_t] = 0.0
            else:
                group_sum = group_rewards.sum()
                baseline[idx_t] = (group_sum - group_rewards) / float(n - 1)
        return baseline

    def _compute_anchor_baseline(
        self,
        rewards: torch.Tensor,
        anchor_observations: Optional[np.ndarray],
    ) -> Optional[torch.Tensor]:
        """
        Compute leave-one-out baseline grouped by anchor_observations.
        rewards: (batch,)
        anchor_observations: np.ndarray or None
        """
        if anchor_observations is None:
            return None

        batch_size = rewards.shape[0]
        device = rewards.device

        try:
            anchor_arr = np.asarray(anchor_observations)
        except Exception:
            return None

        if anchor_arr.shape[0] != batch_size:
            return None

        # Build hashable keys for each anchor
        if anchor_arr.ndim == 1:
            keys = anchor_arr.tolist()
        else:
            flat = anchor_arr.reshape(anchor_arr.shape[0], -1)
            keys = [tuple(row.tolist()) for row in flat]

        anchor_to_indices: Dict[Any, List[int]] = defaultdict(list)
        for i, k in enumerate(keys):
            anchor_to_indices[k].append(i)

        baseline = torch.zeros_like(rewards)
        for idx_list in anchor_to_indices.values():
            if len(idx_list) <= 1:
                continue
            idx_t = torch.tensor(idx_list, device=device, dtype=torch.long)
            group_rewards = rewards[idx_t]
            n = idx_t.numel()
            group_sum = group_rewards.sum()
            baseline[idx_t] = (group_sum - group_rewards) / float(n - 1)

        return baseline

    def compute_advantage(
        self,
        token_level_rewards: torch.Tensor,   # (batch, seq_len)
        response_mask: torch.Tensor,        # (batch, seq_len)
        episode_index: np.ndarray,          # (batch,)
        trajectory_index: np.ndarray,       # (batch,)
        step_rewards: Optional[torch.Tensor] = None,
        anchor_observations: Optional[np.ndarray] = None,
        gamma: float = 0.95,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns for policy update.
        """
        device = token_level_rewards.device
        dtype = token_level_rewards.dtype
        batch_size, seq_len = token_level_rewards.shape

        # Ensure mask is float for arithmetic
        if response_mask.dtype != torch.float32 and response_mask.dtype != torch.float64:
            mask = response_mask.to(dtype=dtype)
        else:
            mask = response_mask

        # Start from token-level rewards
        reward_tokens = token_level_rewards.to(dtype)

        # Incorporate step-level rewards if provided
        if step_rewards is not None:
            if not isinstance(step_rewards, torch.Tensor):
                step_rewards_t = torch.tensor(step_rewards, device=device, dtype=dtype)
            else:
                step_rewards_t = step_rewards.to(device=device, dtype=dtype)

            if step_rewards_t.ndim == 1:
                # Per-sequence reward: assign to last valid response token
                extra = torch.zeros_like(reward_tokens)
                lengths = mask.sum(dim=1).long()
                last_idx = torch.clamp(lengths - 1, min=0)
                rows = torch.arange(batch_size, device=device)
                valid = lengths > 0
                rows_v = rows[valid]
                idx_v = last_idx[valid]
                extra[rows_v, idx_v] = step_rewards_t[rows_v]
                reward_tokens = reward_tokens + extra
            elif step_rewards_t.shape == reward_tokens.shape:
                # Per-token step rewards (already aligned)
                reward_tokens = reward_tokens + step_rewards_t
            elif step_rewards_t.ndim == 2 and step_rewards_t.size(0) == batch_size:
                # Per-step per-sequence rewards, aggregate to last token
                per_seq_step = step_rewards_t.sum(dim=1)
                extra = torch.zeros_like(reward_tokens)
                lengths = mask.sum(dim=1).long()
                last_idx = torch.clamp(lengths - 1, min=0)
                rows = torch.arange(batch_size, device=device)
                valid = lengths > 0
                rows_v = rows[valid]
                idx_v = last_idx[valid]
                extra[rows_v, idx_v] = per_seq_step[rows_v]
                reward_tokens = reward_tokens + extra
            else:
                # Unsupported shape; ignore additional step rewards
                pass

        # Mask out context tokens
        reward_tokens = reward_tokens * mask

        # Discount factor
        gamma_eff = gamma if gamma is not None else self.config.get("gamma", 0.95)

        # Compute discounted returns per token
        returns = torch.zeros_like(reward_tokens)
        G = torch.zeros(batch_size, device=device, dtype=dtype)
        for t in range(seq_len - 1, -1, -1):
            r_t = reward_tokens[:, t]
            G = r_t + gamma_eff * G
            returns[:, t] = G
        returns = returns * mask
        returns = returns.detach()

        # Scalar per-sequence reward (undiscounted total over response tokens)
        seq_reward = reward_tokens.sum(dim=1).detach()

        # Episode-level leave-one-out baseline
        epi_indices = np.asarray(episode_index) if episode_index is not None else None
        baseline_epi = self._compute_group_baseline(seq_reward, epi_indices)

        # Anchor-level leave-one-out baseline (optional)
        baseline_anchor = self._compute_anchor_baseline(seq_reward, anchor_observations)
        if baseline_anchor is not None:
            baseline = 0.5 * baseline_epi + 0.5 * baseline_anchor
        else:
            baseline = baseline_epi

        # Sequence-level advantages (no gradient)
        adv_seq = (seq_reward - baseline).detach()

        # Normalize advantages across batch (whitening)
        mean = adv_seq.mean()
        std = adv_seq.std()
        adv_seq_norm = (adv_seq - mean) / (std + 1e-8)

        # Broadcast to token level and apply mask
        advantages = adv_seq_norm.unsqueeze(1).expand(-1, seq_len)
        advantages = advantages * mask

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
        Compute PPO-style policy gradient loss with optional KL penalty.
        """
        device = log_prob.device
        dtype = log_prob.dtype

        # Mask
        if response_mask.dtype != torch.float32 and response_mask.dtype != torch.float64:
            mask = response_mask.to(dtype=dtype)
        else:
            mask = response_mask

        # Ensure shapes are consistent
        assert old_log_prob.shape == log_prob.shape == advantages.shape == mask.shape

        # Importance sampling ratio
        log_ratio = log_prob - old_log_prob
        ratio = torch.exp(log_ratio)

        # PPO clipped objective
        unclipped_obj = ratio * advantages
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        clipped_obj = clipped_ratio * advantages

        per_token_loss = -torch.min(unclipped_obj, clipped_obj)
        per_token_loss = per_token_loss * mask

        denom = mask.sum().clamp(min=1.0)
        pg_loss = per_token_loss.sum() / denom

        # Approximate KL (from behavior policy to current)
        with torch.no_grad():
            approx_kl = ((old_log_prob - log_prob) * mask).sum() / denom
            clip_frac = ((torch.abs(ratio - 1.0) > clip_ratio).float() * mask).sum() / denom
            entropy = (-log_prob * mask).sum() / denom
            ratio_mean = (ratio * mask).sum() / denom
            adv_mean = (advantages * mask).sum() / denom

        # Optional KL penalty
        use_kl_loss = kwargs.get("use_kl_loss", self.config.get("use_kl_loss", False))
        kl_coef = kwargs.get("kl_loss_coef", self.config.get("kl_loss_coef", 0.01))

        loss = pg_loss
        if use_kl_loss and kl_coef is not None and kl_coef > 0.0:
            loss = loss + kl_coef * approx_kl

        metrics: Dict[str, Any] = {
            "policy_loss": loss.detach(),
            "pg_loss": pg_loss.detach(),
            "approx_kl": approx_kl.detach(),
            "clip_frac": clip_frac.detach(),
            "entropy": entropy.detach(),
            "ratio_mean": ratio_mean.detach(),
            "adv_mean": adv_mean.detach(),
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
        Distribute episode reward to individual steps with simple shaping.
        """
        # Ensure valid trajectory length
        if trajectory_length <= 0:
            return np.zeros(0, dtype=np.float32)

        # Truncate lists if they are longer than trajectory_length
        if len(step_actions) > trajectory_length:
            step_actions = step_actions[:trajectory_length]
        if len(step_observations) > trajectory_length:
            step_observations = step_observations[:trajectory_length]

        rewards = np.zeros(trajectory_length, dtype=np.float32)

        # If no terminal reward, assign small negative step cost
        if episode_reward <= 0.0:
            # Encourage shorter, more purposeful trajectories
            step_penalty = -0.01
            rewards[:] = step_penalty
            return rewards

        # Successful episode: shape reward toward important and later steps
        importance = np.ones(trajectory_length, dtype=np.float32)

        # Heuristic based on action text
        positive_keywords = [
            "open",
            "close",
            "go to",
            "go",
            "walk",
            "move",
            "examine",
            "look",
            "take",
            "pick up",
            "grab",
            "put",
            "place",
            "insert",
            "drop",
            "clean",
            "wash",
            "rinse",
            "cook",
            "heat",
            "cool",
            "slice",
            "chop",
            "turn on",
            "switch on",
            "turn off",
            "switch off",
            "plug in",
            "unplug",
        ]

        for t in range(trajectory_length):
            act = ""
            if t < len(step_actions) and isinstance(step_actions[t], str):
                act = step_actions[t].lower()
            inc = 0.0
            for kw in positive_keywords:
                if kw in act:
                    inc += 1.0
            importance[t] += inc

        # Emphasize later steps (typically closer to task completion)
        time_weights = np.linspace(0.5, 1.0, trajectory_length, dtype=np.float32)

        weights = importance * time_weights
        total = float(weights.sum())
        if total <= 0.0:
            weights = np.ones(trajectory_length, dtype=np.float32) / float(trajectory_length)
        else:
            weights = weights / total

        rewards = weights.astype(np.float32) * float(episode_reward)

        return rewards