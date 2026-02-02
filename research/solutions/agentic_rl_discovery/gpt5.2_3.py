import math
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import torch
from collections import defaultdict

try:
    from verl.utils.torch_functional import masked_mean as verl_masked_mean
    from verl.utils.torch_functional import masked_whiten as verl_masked_whiten
except Exception:
    verl_masked_mean = None
    verl_masked_whiten = None


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, keepdim: bool = False, eps: float = 1e-8):
    mask_f = mask.to(dtype=x.dtype)
    if dim is None:
        denom = mask_f.sum().clamp_min(eps)
        return (x * mask_f).sum() / denom
    denom = mask_f.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return (x * mask_f).sum(dim=dim, keepdim=keepdim) / denom


def masked_whiten(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8, shift_mean: bool = True):
    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    mean = (x * mask_f).sum() / denom
    if shift_mean:
        x0 = x - mean
    else:
        x0 = x
    var = (x0 * x0 * mask_f).sum() / denom
    std = torch.sqrt(var + eps)
    return x0 / std.clamp_min(1e-6)


def _fnv1a64(data: bytes) -> int:
    h = 1469598103934665603
    fnv_prime = 1099511628211
    for b in data:
        h ^= b
        h = (h * fnv_prime) & 0xFFFFFFFFFFFFFFFF
    return h


def _hash_anchor_observations(anchor_observations: np.ndarray) -> np.ndarray:
    if anchor_observations is None:
        return None
    a = np.asarray(anchor_observations)
    if a.ndim == 1 and a.dtype.kind in ("i", "u"):
        return a.astype(np.int64, copy=False)

    out = np.empty((a.shape[0],), dtype=np.int64)
    if a.ndim == 1:
        for i in range(a.shape[0]):
            v = a[i]
            if isinstance(v, bytes):
                out[i] = np.int64(_fnv1a64(v))
            elif isinstance(v, str):
                out[i] = np.int64(_fnv1a64(v.encode("utf-8", errors="ignore")))
            elif isinstance(v, np.ndarray):
                out[i] = np.int64(_fnv1a64(v.tobytes()))
            else:
                out[i] = np.int64(_fnv1a64(str(v).encode("utf-8", errors="ignore")))
        return out

    # For multi-d arrays: hash each row
    flat = a.reshape(a.shape[0], -1)
    if flat.dtype.kind in ("i", "u", "f", "b"):
        for i in range(flat.shape[0]):
            out[i] = np.int64(_fnv1a64(np.ascontiguousarray(flat[i]).tobytes()))
        return out

    for i in range(flat.shape[0]):
        out[i] = np.int64(_fnv1a64(str(flat[i]).encode("utf-8", errors="ignore")))
    return out


def _group_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    use_loo: bool = True,
    normalize: bool = True,
    eps: float = 1e-6,
    degenerate_std_to_one: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # rewards: (B,)
    # group_ids: (B,) int64
    device = rewards.device
    group_ids = group_ids.to(device=device, dtype=torch.long)
    uniq, inv = torch.unique(group_ids, sorted=True, return_inverse=True)
    g = uniq.numel()

    ones = torch.ones_like(rewards, dtype=rewards.dtype)
    sums = torch.zeros(g, device=device, dtype=rewards.dtype).scatter_add_(0, inv, rewards)
    sums2 = torch.zeros(g, device=device, dtype=rewards.dtype).scatter_add_(0, inv, rewards * rewards)
    counts = torch.zeros(g, device=device, dtype=rewards.dtype).scatter_add_(0, inv, ones)

    mean = sums / counts.clamp_min(1.0)
    var = (sums2 / counts.clamp_min(1.0)) - mean * mean
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var + eps)

    if degenerate_std_to_one:
        std = torch.where(std < 1e-3, torch.ones_like(std), std)

    if use_loo:
        denom = (counts - 1.0).clamp_min(1.0)
        loo_mean = (sums - rewards) / denom
        loo_mean = torch.where(counts > 1.5, loo_mean, torch.zeros_like(loo_mean))
        adv = rewards - loo_mean
    else:
        adv = rewards - mean

    if normalize:
        adv = adv / std

    counts_per_sample = counts[inv]
    return adv, counts_per_sample


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        self.config = {
            "gamma": 0.95,
            "clip_ratio": 0.2,
            "use_kl_loss": False,
            "kl_loss_coef": 0.01,
            "use_anchor_grouping": True,
            "anchor_blend_max_group": 8,
            "adv_clip": 5.0,
            "use_loo_baseline": True,
            "global_whiten_adv": True,
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
        device = token_level_rewards.device
        mask = response_mask.to(device=device, dtype=token_level_rewards.dtype)

        if step_rewards is not None:
            sr = step_rewards.to(device=device, dtype=token_level_rewards.dtype)
            if sr.ndim == 2:
                rewards = (sr * mask).sum(dim=1)
            else:
                rewards = sr.reshape(-1)
        else:
            rewards = (token_level_rewards.to(dtype=token_level_rewards.dtype) * mask).sum(dim=1)

        if rewards.ndim != 1:
            rewards = rewards.reshape(-1)

        ep = torch.from_numpy(np.asarray(episode_index, dtype=np.int64)).to(device=device, dtype=torch.long)

        adv_ep, _ = _group_advantages(
            rewards,
            ep,
            use_loo=bool(self.config.get("use_loo_baseline", True)),
            normalize=True,
        )

        adv = adv_ep

        use_anchor = bool(self.config.get("use_anchor_grouping", True)) and (anchor_observations is not None)
        if use_anchor:
            ah = _hash_anchor_observations(anchor_observations)
            if ah is not None:
                ah_t = torch.from_numpy(ah.astype(np.int64, copy=False)).to(device=device, dtype=torch.long)
                gid = ep * 1315423911 + ah_t
                adv_anchor, anchor_counts = _group_advantages(
                    rewards,
                    gid,
                    use_loo=bool(self.config.get("use_loo_baseline", True)),
                    normalize=True,
                )
                max_g = float(self.config.get("anchor_blend_max_group", 8))
                w = (anchor_counts - 1.0) / max(1.0, (max_g - 1.0))
                w = torch.clamp(w, 0.0, 1.0)
                adv = (1.0 - w) * adv_ep + w * adv_anchor

        adv = torch.clamp(adv, -float(self.config.get("adv_clip", 5.0)), float(self.config.get("adv_clip", 5.0)))

        adv_tok = adv.unsqueeze(1) * mask
        ret_tok = rewards.unsqueeze(1) * mask

        if bool(self.config.get("global_whiten_adv", True)):
            if verl_masked_whiten is not None:
                adv_tok = verl_masked_whiten(adv_tok, mask, shift_mean=True)
            else:
                adv_tok = masked_whiten(adv_tok, mask, shift_mean=True)
            adv_tok = torch.clamp(adv_tok, -float(self.config.get("adv_clip", 5.0)), float(self.config.get("adv_clip", 5.0)))

        return adv_tok, ret_tok

    def compute_policy_loss(
        self,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        clip_ratio: float = 0.2,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        mask = response_mask.to(dtype=log_prob.dtype)

        adv = advantages.to(dtype=log_prob.dtype)
        adv = adv * mask

        log_ratio = (log_prob - old_log_prob).to(dtype=log_prob.dtype)
        ratio = torch.exp(log_ratio)

        clip_ratio = float(clip_ratio)
        ratio_clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

        pg1 = ratio * adv
        pg2 = ratio_clipped * adv
        pg = torch.minimum(pg1, pg2)

        if verl_masked_mean is not None:
            loss = -verl_masked_mean(pg, mask)
            clip_frac = verl_masked_mean(((ratio > (1.0 + clip_ratio)) | (ratio < (1.0 - clip_ratio))).to(pg.dtype), mask)
            approx_kl = verl_masked_mean((old_log_prob - log_prob), mask)
        else:
            loss = -masked_mean(pg, mask)
            clip_frac = masked_mean(((ratio > (1.0 + clip_ratio)) | (ratio < (1.0 - clip_ratio))).to(pg.dtype), mask)
            approx_kl = masked_mean((old_log_prob - log_prob), mask)

        if bool(self.config.get("use_kl_loss", False)):
            kl_coef = float(self.config.get("kl_loss_coef", 0.01))
            loss = loss + kl_coef * approx_kl

        metrics = {
            "clip_frac": float(clip_frac.detach().cpu()),
            "approx_kl": float(approx_kl.detach().cpu()),
            "adv_mean": float(masked_mean(adv, mask).detach().cpu()) if verl_masked_mean is None else float(verl_masked_mean(adv, mask).detach().cpu()),
            "ratio_mean": float((masked_mean(ratio, mask).detach().cpu()) if verl_masked_mean is None else float(verl_masked_mean(ratio, mask).detach().cpu())),
        }
        return loss, metrics

    def assign_step_rewards(
        self,
        episode_reward: float,
        trajectory_length: int,
        step_observations: list,
        step_actions: list,
        **kwargs,
    ) -> np.ndarray:
        T = int(trajectory_length)
        if T <= 0:
            return np.zeros((0,), dtype=np.float32)

        R = float(episode_reward)
        if R <= 0.0:
            return np.zeros((T,), dtype=np.float32)

        gamma = float(kwargs.get("gamma", self.config.get("gamma", 0.95)))
        if gamma < 0.0:
            gamma = 0.0
        if gamma > 0.9999:
            gamma = 0.9999

        if abs(gamma - 1.0) < 1e-8:
            denom = float(T)
        else:
            denom = (1.0 - (gamma ** T)) / (1.0 - gamma)
            denom = max(denom, 1e-8)

        # Distribute reward so sum_t r_t = episode_reward; also gives bigger per-step reward for shorter successful trajectories.
        # Geometric credit from end: r_t ∝ gamma^(T-1-t).
        idx = np.arange(T, dtype=np.float32)
        powers = (gamma ** (T - 1 - idx)).astype(np.float32)
        step_rewards = (R * powers / denom).astype(np.float32)
        return step_rewards