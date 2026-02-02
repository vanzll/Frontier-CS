import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List


def _to_device(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if t.device != ref.device:
        return t.to(ref.device)
    return t


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    if dim is None:
        denom = mask.sum().clamp_min(1.0)
        return (x * mask).sum() / denom
    else:
        denom = mask.sum(dim=dim, keepdim=True).clamp_min(1.0)
        out = (x * mask).sum(dim=dim, keepdim=True) / denom
        if not keepdim:
            out = out.squeeze(dim)
        return out


def _masked_std(x: torch.Tensor, mask: torch.Tensor, dim=None, keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
    mean = _masked_mean(x, mask, dim=dim, keepdim=True)
    var = _masked_mean((x - mean) ** 2, mask, dim=dim, keepdim=True)
    std = (var + eps) ** 0.5
    if dim is None or keepdim:
        return std
    return std.squeeze(dim)


def _masked_whiten(x: torch.Tensor, mask: torch.Tensor, dim=None, keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
    mean = _masked_mean(x, mask, dim=dim, keepdim=True)
    std = _masked_std(x, mask, dim=dim, keepdim=True, eps=eps)
    y = (x - mean) / (std + eps)
    if dim is None or keepdim:
        return y
    return y.squeeze(dim)


def _discount_cumsum_token(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    # rewards: (B, T), already masked for invalid tokens if needed
    B, T = rewards.shape
    returns = torch.zeros_like(rewards)
    running = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    for t in range(T - 1, -1, -1):
        running = rewards[:, t] + gamma * running
        returns[:, t] = running
    return returns


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        self.config = {
            "gamma": 0.95,
            "clip_ratio": 0.2,
            "use_kl_loss": False,
            "kl_loss_coef": 0.01,
        }
        return self

    def compute_advantage(
        self,
        token_level_rewards: torch.Tensor,   # (batch, seq_len)
        response_mask: torch.Tensor,          # (batch, seq_len)
        episode_index: np.ndarray,            # (batch,) - episode group IDs
        trajectory_index: np.ndarray,         # (batch,) - trajectory IDs within group
        step_rewards: Optional[torch.Tensor] = None,
        anchor_observations: Optional[np.ndarray] = None,
        gamma: float = 0.95,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Setup
        device = token_level_rewards.device
        dtype = token_level_rewards.dtype
        mask = response_mask.to(device=device, dtype=dtype)
        B, T = token_level_rewards.shape
        cfg_gamma = float(self.config.get("gamma", gamma))

        # Aggregate per-token reward
        rewards = token_level_rewards.to(dtype=dtype)
        # If additional step rewards are provided and shape matches, include them
        if step_rewards is not None:
            if isinstance(step_rewards, torch.Tensor):
                if step_rewards.shape == rewards.shape:
                    rewards = rewards + step_rewards.to(device=device, dtype=dtype)
            # If step_rewards is provided in a different shape, ignore at this stage
        rewards = rewards * mask

        # Compute token-level discounted returns
        returns = _discount_cumsum_token(rewards, cfg_gamma) * mask

        # Sequence-level return per trajectory for group normalization
        # Use return at first valid token for each sequence
        with torch.no_grad():
            valid_mask_bool = (mask > 0.0)
            idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            masked_idxs = idxs.masked_fill(~valid_mask_bool, T)
            first_idx = masked_idxs.min(dim=1).values  # T if no valid token
            any_valid = valid_mask_bool.any(dim=1)
            first_idx_clamped = torch.clamp(first_idx, max=T - 1)
            gather_idx = first_idx_clamped.view(B, 1)
            seq_return = returns.gather(1, gather_idx).squeeze(1) * any_valid.to(dtype=dtype)

        # Group-wise whitening (GRPO-style) within episode groups
        episode_index_t = torch.as_tensor(episode_index, device=device)
        unique_groups = torch.unique(episode_index_t)
        seq_adv = torch.zeros(B, device=device, dtype=dtype)
        eps = 1e-8
        for g in unique_groups.tolist():
            idx = torch.nonzero(episode_index_t == g, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            vals = seq_return.index_select(0, idx)
            mean = vals.mean()
            std = vals.std(unbiased=False)
            if std.item() < eps:
                # Fallback to centered advantage without scaling if std is too small
                grp_adv = vals - mean
            else:
                grp_adv = (vals - mean) / (std + eps)
            seq_adv.index_copy_(0, idx, grp_adv)

        # Broadcast sequence-level advantage to tokens
        advantages = (seq_adv.view(B, 1).expand(B, T)) * mask

        # Optional: add a small per-batch normalization for stability (across masked tokens)
        advantages = _masked_whiten(advantages, mask, dim=(0, 1))

        # Ensure returns are zeroed outside mask
        returns = returns * mask

        return advantages.to(dtype=dtype), returns.to(dtype=dtype)

    def compute_policy_loss(
        self,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        clip_ratio: float = 0.2,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Prepare
        mask = response_mask.to(dtype=log_prob.dtype, device=log_prob.device)
        clip = float(self.config.get("clip_ratio", clip_ratio))
        use_kl = bool(self.config.get("use_kl_loss", False))
        kl_coef = float(self.config.get("kl_loss_coef", 0.01))

        # Flatten
        logp = log_prob
        old_logp = old_log_prob
        adv = advantages

        # Apply mask
        # Normalize advantages across masked tokens for stability
        adv = _masked_whiten(adv, mask, dim=(0, 1))
        # Clip extreme advantages to reduce variance spikes
        adv = adv.clamp(-5.0, 5.0)

        log_ratio = (logp - old_logp)
        ratio = torch.exp(log_ratio)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
        pg_loss = -_masked_mean(torch.min(surr1, surr2), mask, dim=(0, 1))

        # Approximate KL for logging and optional penalty
        approx_kl = 0.5 * _masked_mean(log_ratio.pow(2), mask, dim=(0, 1))
        clipped = ((ratio - 1.0).abs() > clip).to(mask.dtype)
        clip_frac = _masked_mean(clipped, mask, dim=(0, 1))

        loss = pg_loss
        kl_loss = torch.tensor(0.0, device=logp.device, dtype=logp.dtype)
        if use_kl:
            kl_loss = kl_coef * approx_kl
            loss = loss + kl_loss

        metrics = {
            "loss": loss.detach(),
            "pg_loss": pg_loss.detach(),
            "kl": approx_kl.detach(),
            "kl_loss": kl_loss.detach(),
            "clip_frac": clip_frac.detach(),
            "adv_mean": _masked_mean(adv, mask, dim=(0, 1)).detach(),
            "adv_std": _masked_std(adv, mask, dim=(0, 1)).detach(),
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
        T = int(trajectory_length) if trajectory_length is not None else len(step_actions) if step_actions is not None else 0
        if T <= 0:
            return np.zeros((0,), dtype=np.float32)

        # Ensure lists have at least T elements
        if step_actions is None:
            step_actions = [""] * T
        if step_observations is None:
            step_observations = [""] * T
        if len(step_actions) < T:
            step_actions = step_actions + [""] * (T - len(step_actions))
        if len(step_observations) < T:
            step_observations = step_observations + [""] * (T - len(step_observations))

        step_rewards = np.zeros(T, dtype=np.float32)

        # Heuristic shaping
        success = float(episode_reward) >= 0.999
        if success:
            # Allocate a small portion of the terminal success reward across steps to bias efficiency
            total_shaping = 0.30  # keep small to not overshadow terminal reward
            eff_frac = 0.60
            act_frac = 0.30
            obs_frac = 0.10
            gamma_e = 0.97

            # Efficiency component: earlier steps get slightly higher reward
            weights = np.power(gamma_e, np.arange(T, dtype=np.float64))
            if weights.sum() > 0:
                weights = weights / weights.sum()
            step_rewards += (total_shaping * eff_frac) * weights.astype(np.float32)

            # Key action component
            key_action_tokens = [
                "put", "place", "insert", "drop", "store", "open", "close",
                "clean", "wash", "rinse", "cook", "heat", "cool", "turn on",
                "turn off", "toggle", "take", "grab", "pick", "pick up", "go to"
            ]
            action_hits = []
            for i, a in enumerate(step_actions[:T]):
                al = (a or "").lower()
                if any(tok in al for tok in key_action_tokens):
                    action_hits.append(i)
            if len(action_hits) > 0:
                per = (total_shaping * act_frac) / len(action_hits)
                for i in action_hits:
                    step_rewards[i] += per
            else:
                # reallocate to efficiency if no key action found
                step_rewards += (total_shaping * act_frac) * weights.astype(np.float32)

            # Observation progress component
            positive_obs_phrases = [
                "you put", "you place", "is now in", "is now on", "you open", "you close",
                "you cleaned", "you washed", "success", "done", "completed", "now clean",
                "inside the", "on the", "into the", "placed", "stored"
            ]
            obs_hits = []
            for i, o in enumerate(step_observations[:T]):
                ol = (o or "").lower()
                if any(ph in ol for ph in positive_obs_phrases):
                    obs_hits.append(i)
            if len(obs_hits) > 0:
                per = (total_shaping * obs_frac) / len(obs_hits)
                for i in obs_hits:
                    step_rewards[i] += per
            else:
                step_rewards += (total_shaping * obs_frac) * weights.astype(np.float32)

        else:
            # Failure: small per-step penalty to encourage shorter, purposeful trajectories
            total_penalty = 0.05
            if T > 0:
                step_rewards -= (total_penalty / T)

            # Penalize repeated identical consecutive actions more
            rep_penalty_total = 0.02
            repeats = []
            for i in range(1, T):
                a_prev = (step_actions[i - 1] or "").strip().lower()
                a_cur = (step_actions[i] or "").strip().lower()
                if a_cur == a_prev and a_cur != "":
                    repeats.append(i)
            if len(repeats) > 0:
                per = rep_penalty_total / len(repeats)
                for i in repeats:
                    step_rewards[i] -= per

        # Clip per-step rewards to keep magnitudes controlled
        np.clip(step_rewards, -0.2, 0.2, out=step_rewards)
        return step_rewards.astype(np.float32)