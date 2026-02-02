import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List

from verl.utils.torch_functional import masked_mean, masked_whiten, entropy_from_logits


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm. Called once before training.

        Set hyperparameters in self.config:
        """
        self.config = {
            "gamma": 0.95,           # Discount factor for token-level returns
            "clip_ratio": 0.2,       # PPO clip range
            "use_kl_loss": False,    # Add KL penalty against a reference policy
            "kl_loss_coef": 0.01,    # KL penalty coefficient
            "normalize_advantages": True,
            "adv_whiten_eps": 1e-8,
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
        Compute advantages and returns for policy update.
        """
        del trajectory_index, anchor_observations  # not used in this implementation

        device = token_level_rewards.device
        dtype = token_level_rewards.dtype
        if gamma is None:
            gamma = self.config.get("gamma", 0.95)

        # Ensure mask is float and on the same device
        mask = response_mask
        if mask.dtype != dtype:
            mask = mask.to(dtype)
        else:
            mask = mask.clone()
        mask = mask.to(device)

        batch_size, seq_len = token_level_rewards.shape

        # 1. Compute discounted token-level returns (backward pass)
        returns = torch.zeros_like(token_level_rewards, device=device, dtype=dtype)
        running_return = torch.zeros(batch_size, device=device, dtype=dtype)

        for t in range(seq_len - 1, -1, -1):
            r_t = token_level_rewards[:, t]
            running_return = r_t + gamma * running_return
            returns[:, t] = running_return

        # Mask out invalid tokens
        returns = returns * mask

        # 2. Sequence-level scalar reward per trajectory (for group baselines)
        #    Typically rewards are sparse and only at EOS, so sum is fine.
        scalar_rewards = (token_level_rewards * mask).sum(dim=1)

        # Optionally incorporate step-level rewards, if provided
        if step_rewards is not None:
            sr = step_rewards
            if not isinstance(sr, torch.Tensor):
                sr = torch.as_tensor(sr, device=device, dtype=dtype)
            else:
                sr = sr.to(device=device, dtype=dtype)

            # Heuristic handling of possible shapes
            if sr.dim() == 1 and sr.shape[0] == batch_size:
                scalar_rewards = scalar_rewards + sr
            elif sr.dim() == 2 and sr.shape[0] == batch_size:
                if sr.shape[1] == seq_len:
                    scalar_rewards = scalar_rewards + (sr * mask).sum(dim=1)
                else:
                    scalar_rewards = scalar_rewards + sr.sum(dim=1)
            elif sr.dim() == token_level_rewards.dim() and sr.shape == token_level_rewards.shape:
                scalar_rewards = scalar_rewards + (sr * mask).sum(dim=1)
            # Otherwise ignore mismatched shapes

        # 3. Group baseline using episode_index (GRPO-style)
        group_baseline = torch.zeros(batch_size, device=device, dtype=dtype)
        groups: Dict[int, List[int]] = defaultdict(list)
        for i, eid in enumerate(episode_index):
            groups[int(eid)].append(i)

        for _, idxs in groups.items():
            idx_tensor = torch.tensor(idxs, device=device, dtype=torch.long)
            group_mean = scalar_rewards[idx_tensor].mean()
            group_baseline[idx_tensor] = group_mean

        adv_scalar = scalar_rewards - group_baseline  # (batch,)

        # 4. (Optional) normalize advantages across the batch for stability
        if self.config.get("normalize_advantages", True):
            mean = adv_scalar.mean()
            std = adv_scalar.std(unbiased=False)
            adv_scalar = (adv_scalar - mean) / (std + self.config.get("adv_whiten_eps", 1e-8))

        # 5. Broadcast sequence-level advantages to all valid tokens
        advantages = adv_scalar.unsqueeze(1) * mask  # (batch, seq_len)

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
        Compute PPO-style clipped policy gradient loss.
        """
        device = log_prob.device
        dtype = log_prob.dtype

        if clip_ratio is None:
            clip_ratio = self.config.get("clip_ratio", 0.2)

        mask = response_mask
        if mask.dtype != dtype:
            mask = mask.to(dtype)
        mask = mask.to(device)

        # Detach advantages to avoid backprop through reward computation
        adv = advantages.to(device).detach()
        adv = adv * mask

        # Compute log-prob ratio
        log_ratio = (log_prob - old_log_prob).to(device)
        ratio = torch.exp(log_ratio)

        # Clipped surrogate objective
        ratio_clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * ratio_clipped
        pg_loss = torch.max(pg_loss1, pg_loss2)

        # Mask and average over valid tokens
        pg_loss = pg_loss * mask
        denom = mask.sum().clamp(min=1.0)
        loss = pg_loss.sum() / denom

        metrics: Dict[str, Any] = {}

        # Approximate KL divergence between old and new policy
        approx_kl = 0.5 * ((log_ratio ** 2) * mask).sum() / denom
        metrics["approx_kl"] = approx_kl.item()

        # Clip fraction: fraction of tokens where ratio was clipped
        clipped = ((ratio - 1.0).abs() > clip_ratio).to(dtype) * mask
        clip_frac = clipped.sum() / denom
        metrics["clip_frac"] = clip_frac.item()

        # Optional entropy term if logits are provided
        logits = kwargs.get("logits", None)
        if logits is not None:
            ent = entropy_from_logits(logits)
            if ent.shape != mask.shape:
                # Broadcast/reshape if necessary; fall back to mean if shapes mismatch
                ent_mean = ent.mean()
            else:
                ent_mean = (ent * mask).sum() / denom
            metrics["entropy"] = ent_mean.item()

        # Optional KL penalty against a reference policy
        if self.config.get("use_kl_loss", False):
            ref_log_prob = kwargs.get("ref_log_prob", None)
            if ref_log_prob is not None:
                ref_log_prob = ref_log_prob.to(device)
                # KL(current || ref) = E[log_pi - log_ref]
                kl = ((log_prob - ref_log_prob) * mask).sum() / denom
                kl_coef = self.config.get("kl_loss_coef", 0.01)
                loss = loss + kl_coef * kl
                metrics["kl_penalty"] = kl.item()
                metrics["kl_coef"] = kl_coef

        metrics["policy_loss"] = loss.item()

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
        """
        del step_observations, step_actions  # not used in this simple scheme

        if trajectory_length <= 0:
            return np.zeros(0, dtype=np.float32)

        # If episode failed (reward 0), assign zero to all steps
        if episode_reward <= 0.0:
            return np.zeros(trajectory_length, dtype=np.float32)

        # Successful episode: distribute reward across steps, emphasizing later steps
        # Linearly increasing weights from early to late steps, normalized to sum to 1.
        weights = np.arange(1, trajectory_length + 1, dtype=np.float32)
        weights_sum = weights.sum()
        if weights_sum <= 0:
            return np.zeros(trajectory_length, dtype=np.float32)

        weights = weights / weights_sum
        step_rewards = episode_reward * weights.astype(np.float32)

        return step_rewards