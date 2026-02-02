import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List

from verl.utils.torch_functional import masked_mean, masked_whiten, entropy_from_logits


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm. Called once before training.
        """
        self.config: Dict[str, Any] = {
            "gamma": 0.95,           # Discount factor (mainly used in step-level shaping)
            "clip_ratio": 0.2,       # PPO clip range
            "use_kl_loss": False,    # Add KL penalty between current and behavior policy
            "kl_loss_coef": 0.01,    # KL penalty coefficient
            # Extra knobs (not required by spec but used internally)
            "use_rloo_baseline": True,   # Use leave-one-out baseline within episode groups
            "whiten_advantage": True,    # Whiten advantages within episode groups
        }
        return self

    def compute_advantage(
        self,
        token_level_rewards: torch.Tensor,   # (batch, seq_len)
        response_mask: torch.Tensor,         # (batch, seq_len)
        episode_index: np.ndarray,           # (batch,) - episode group IDs
        trajectory_index: np.ndarray,        # (batch,) - trajectory IDs within group
        step_rewards: Optional[torch.Tensor] = None,
        anchor_observations: Optional[np.ndarray] = None,
        gamma: float = 0.95,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GRPO-style sequence-level advantages with group-wise RLOO baseline.
        """
        # Ensure basic shapes
        assert token_level_rewards.dim() == 2, "token_level_rewards must be (batch, seq_len)"
        assert response_mask.shape == token_level_rewards.shape, "response_mask must match reward shape"

        device = token_level_rewards.device
        dtype = token_level_rewards.dtype
        batch_size, seq_len = token_level_rewards.shape

        # Convert mask to float for arithmetic
        mask = response_mask.to(dtype)

        # Combine token-level rewards with optional step-level rewards if shapes allow
        rewards = token_level_rewards

        if step_rewards is not None:
            # Bring step_rewards to same device/dtype
            if not isinstance(step_rewards, torch.Tensor):
                step_rewards = torch.tensor(step_rewards, dtype=dtype, device=device)
            else:
                step_rewards = step_rewards.to(device=device, dtype=dtype)

            # Try to add directly; if shape mismatches, fall back to simple aggregations
            added = False
            try:
                rewards = rewards + step_rewards
                added = True
            except Exception:
                added = False

            if not added:
                # Handle some common mismatched shapes gracefully
                try:
                    if step_rewards.dim() == 2 and step_rewards.size(0) == batch_size:
                        if step_rewards.size(1) == seq_len:
                            rewards = rewards + step_rewards
                            added = True
                        else:
                            # Aggregate per-trajectory and broadcast
                            sr = step_rewards.mean(dim=1, keepdim=True)
                            rewards = rewards + sr
                            added = True
                    elif step_rewards.dim() == 1 and step_rewards.size(0) == batch_size:
                        rewards = rewards + step_rewards.view(batch_size, 1)
                        added = True
                except Exception:
                    added = False
            # If still not added, silently ignore step_rewards

        # Mask out prompt/pad tokens
        rewards = rewards * mask  # (B, T)

        # Sequence-level scalar reward per trajectory
        # (sum of all token-level rewards, usually just EOS + any shaped terms)
        seq_rewards = rewards.sum(dim=1)  # (B,)

        # Group-wise advantage computation (GRPO-style with RLOO baseline)
        ep_idx_np = np.asarray(episode_index)
        assert ep_idx_np.shape[0] == batch_size, "episode_index must have length batch_size"

        adv_seq = torch.zeros_like(seq_rewards)

        use_rloo = self.config.get("use_rloo_baseline", True)
        whiten_adv = self.config.get("whiten_advantage", True)

        unique_eps = np.unique(ep_idx_np)
        for ep in unique_eps:
            mask_np = ep_idx_np == ep
            idx_np = np.nonzero(mask_np)[0]
            if idx_np.size == 0:
                continue
            idx_t = torch.from_numpy(idx_np).to(device=device, dtype=torch.long)

            r_g = seq_rewards[idx_t]  # (n_group,)

            n = r_g.size(0)
            if n <= 1:
                # With only one trajectory, baseline can't reduce variance;
                # fall back to zero-centered (no update) for stability
                adv_g = torch.zeros_like(r_g)
            else:
                if use_rloo:
                    # Leave-one-out baseline: b_i = (sum_j r_j - r_i) / (n-1)
                    r_sum = r_g.sum()
                    baseline = (r_sum - r_g) / (n - 1)
                    adv_g = r_g - baseline
                else:
                    # Simple group mean baseline
                    baseline = r_g.mean()
                    adv_g = r_g - baseline

            if whiten_adv:
                # Whiten within group to stabilize scale
                mean = adv_g.mean()
                std = adv_g.std(unbiased=False)
                if torch.isfinite(std) and std > 1e-6:
                    adv_g = (adv_g - mean) / (std + 1e-8)

            adv_seq[idx_t] = adv_g

        # Broadcast scalar sequence advantages/returns across tokens in the response
        advantages = adv_seq.unsqueeze(1) * mask  # (B, T)
        returns = seq_rewards.unsqueeze(1) * mask  # (B, T)

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
        PPO-style clipped policy gradient loss on token level.
        """
        assert old_log_prob.shape == log_prob.shape == advantages.shape == response_mask.shape, \
            "Shapes of log probs, advantages, and mask must all match"

        device = log_prob.device
        dtype = log_prob.dtype

        mask = response_mask.to(dtype)
        valid_count = mask.sum().clamp_min(1.0)

        # Detach advantages to avoid backprop through advantage computation
        adv = advantages.to(dtype).detach()

        # Optional global whitening of advantages over valid tokens
        # (helps stabilize optimization when reward scales drift)
        if self.config.get("whiten_advantage", True):
            adv_mean = (adv * mask).sum() / valid_count
            adv_var = ((adv - adv_mean) ** 2 * mask).sum() / valid_count
            adv_std = torch.sqrt(adv_var + 1e-8)
            adv = (adv - adv_mean) / (adv_std + 1e-8)

        log_ratio = log_prob - old_log_prob  # (B, T)
        ratio = torch.exp(log_ratio)         # (B, T)

        # PPO clipped objective (per token)
        unclipped_obj = ratio * adv
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        clipped_obj = clipped_ratio * adv

        # We use negative because we minimize loss while maximizing objective
        token_loss = -torch.min(unclipped_obj, clipped_obj) * mask
        loss = token_loss.sum() / valid_count

        metrics: Dict[str, Any] = {}

        with torch.no_grad():
            approx_kl = 0.5 * ((log_ratio ** 2) * mask).sum() / valid_count
            clip_mask = (torch.abs(ratio - 1.0) > clip_ratio).to(dtype) * mask
            clip_frac = clip_mask.sum() / valid_count
            adv_mean = (adv * mask).sum() / valid_count
            adv_abs_mean = (adv.abs() * mask).sum() / valid_count

            metrics["approx_kl"] = approx_kl.item()
            metrics["clip_frac"] = clip_frac.item()
            metrics["adv_mean"] = adv_mean.item()
            metrics["adv_abs_mean"] = adv_abs_mean.item()

        # Optional additional KL penalty with behavior policy
        if self.config.get("use_kl_loss", False):
            # KL(old || new) ≈ E_old[log p_old - log p_new]
            kl = ((old_log_prob - log_prob) * mask).sum() / valid_count
            kl_coef = float(self.config.get("kl_loss_coef", 0.01))
            loss = loss + kl_coef * kl
            with torch.no_grad():
                metrics["kl"] = kl.item()
                metrics["kl_coef"] = kl_coef

        metrics["loss"] = loss.detach().item()

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
        Distribute episode reward across environment steps.

        For successful episodes (reward > 0), we:
          - Start from uniform allocation across steps
          - Upweight semantically "useful" actions (e.g., take/put/open/close)
          - Upweight the final step (which typically achieves the goal)
          - Normalize so total equals episode_reward

        For failed episodes (reward <= 0), we return all zeros to avoid
        misleading credit assignment.
        """
        if trajectory_length <= 0:
            return np.zeros(0, dtype=np.float32)

        # No shaping for failed episodes: keep credit sparse
        if episode_reward <= 0.0:
            return np.zeros(trajectory_length, dtype=np.float32)

        # Initial uniform weights
        weights = np.ones(trajectory_length, dtype=np.float32)

        # Heuristic action-based weighting
        # Encourage state-changing actions over pure information gathering
        effectful_keywords = [
            "take", "put", "open", "close", "switch", "toggle",
            "insert", "place", "drop", "clean", "heat", "cool",
        ]
        cheap_keywords = [
            "look", "examine", "inventory", "wait", "check",
        ]

        for t in range(trajectory_length):
            action = ""
            if t < len(step_actions) and step_actions[t] is not None:
                action = str(step_actions[t]).lower()

            if any(kw in action for kw in effectful_keywords):
                weights[t] *= 1.5
            elif any(kw in action for kw in cheap_keywords):
                weights[t] *= 0.75

        # Upweight the final step; it usually corresponds to task completion
        weights[-1] *= 2.0

        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            # Fallback to uniform split if something went wrong
            return np.full(trajectory_length, episode_reward / float(trajectory_length), dtype=np.float32)

        step_rewards = (weights / total_weight) * float(episode_reward)
        return step_rewards.astype(np.float32)