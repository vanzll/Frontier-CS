import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List

try:
    from verl.utils.torch_functional import masked_mean as verl_masked_mean
    from verl.utils.torch_functional import masked_whiten as verl_masked_whiten
except Exception:
    verl_masked_mean = None
    verl_masked_whiten = None


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False, eps: float = 1e-8) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim, keepdim=keepdim)
    m = mask.to(x.dtype)
    if dim is None:
        denom = m.sum().clamp(min=eps)
        return (x * m).sum() / denom
    else:
        denom = m.sum(dim=dim, keepdim=keepdim).clamp(min=eps)
        return (x * m).sum(dim=dim, keepdim=keepdim) / denom


def _masked_whiten(x: torch.Tensor, mask: torch.Tensor, dim=None, eps: float = 1e-8) -> torch.Tensor:
    if mask is None:
        mean = x.mean(dim=dim, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=dim, keepdim=True)
        return (x - mean) / torch.sqrt(var + eps)
    m = mask.to(x.dtype)
    if dim is None:
        mean = (x * m).sum() / m.sum().clamp(min=eps)
        var = ((x - mean) ** 2 * m).sum() / m.sum().clamp(min=eps)
        return (x - mean) / torch.sqrt(var + eps)
    else:
        mean = _masked_mean(x, mask, dim=dim, keepdim=True, eps=eps)
        var = _masked_mean((x - mean) ** 2, mask, dim=dim, keepdim=True, eps=eps)
        return (x - mean) / torch.sqrt(var + eps)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False, eps: float = 1e-8) -> torch.Tensor:
    if verl_masked_mean is not None:
        return verl_masked_mean(x, mask, dim=dim, keepdim=keepdim)
    return _masked_mean(x, mask, dim=dim, keepdim=keepdim, eps=eps)


def masked_whiten(x: torch.Tensor, mask: torch.Tensor, dim=None, eps: float = 1e-8) -> torch.Tensor:
    if verl_masked_whiten is not None:
        return verl_masked_whiten(x, mask, dim=dim)
    return _masked_whiten(x, mask, dim=dim, eps=eps)


def _build_group_indices(group_ids: np.ndarray) -> Dict[Any, List[int]]:
    mapping: Dict[Any, List[int]] = defaultdict(list)
    if isinstance(group_ids, torch.Tensor):
        group_ids_list = group_ids.detach().cpu().tolist()
    else:
        group_ids_list = group_ids.tolist()
    for i, g in enumerate(group_ids_list):
        mapping[g].append(i)
    return mapping


def _loo_baseline_per_group(values: torch.Tensor, group_map: Dict[Any, List[int]]) -> torch.Tensor:
    device = values.device
    baseline = torch.zeros_like(values)
    for _, idxs in group_map.items():
        idx = torch.tensor(idxs, dtype=torch.long, device=device)
        n = idx.numel()
        if n > 1:
            v = values.index_select(0, idx)
            s = v.sum(dim=0, keepdim=True)
            b = (s - v) / (n - 1)
            baseline.index_copy_(0, idx, b)
        else:
            baseline.index_fill_(0, idx, 0.0)
    return baseline


def _whiten_by_group(values: torch.Tensor, group_map: Dict[Any, List[int]], eps: float = 1e-8) -> torch.Tensor:
    device = values.device
    out = torch.zeros_like(values)
    for _, idxs in group_map.items():
        idx = torch.tensor(idxs, dtype=torch.long, device=device)
        v = values.index_select(0, idx)
        if v.numel() <= 1:
            out.index_copy_(0, idx, torch.zeros_like(v))
        else:
            m = v.mean(dim=0, keepdim=True)
            std = v.std(dim=0, keepdim=True, unbiased=False)
            w = (v - m) / (std + eps)
            out.index_copy_(0, idx, w)
    return out


def _discount_cumsum_tokenwise(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    batch, seq_len = rewards.size()
    returns = torch.zeros_like(rewards)
    running = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)
    for t in range(seq_len - 1, -1, -1):
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
            "normalize_adv": True,
            "adv_clip_value": 10.0,
            "step_adv_coef": 0.7,
            "final_whiten_by_group": True,
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
        cfg = self.config if hasattr(self, "config") else {}
        gamma = float(cfg.get("gamma", gamma))
        eps = 1e-8

        device = token_level_rewards.device
        rewards = token_level_rewards.to(dtype=torch.float32)
        mask = response_mask.to(dtype=torch.float32)
        B, T = rewards.shape

        # Sequence-level scalar reward per sample
        seq_rewards = (rewards * mask).sum(dim=1)

        # Grouping by episode_index (trajectories sampled from same initial prompt)
        group_map = _build_group_indices(episode_index)

        # Leave-one-out baseline within group (GRPO-style)
        seq_baseline = _loo_baseline_per_group(seq_rewards, group_map)
        seq_adv = seq_rewards - seq_baseline

        # Normalize sequence-level advantages within group
        seq_adv = _whiten_by_group(seq_adv, group_map)

        # Step-level shaping (GiGPO-inspired) using per-step rewards across same episode group
        step_adv_scalar = torch.zeros(B, device=device, dtype=torch.float32)
        if step_rewards is not None and isinstance(step_rewards, torch.Tensor) and step_rewards.ndim == 2:
            step_r = step_rewards.to(device=device, dtype=torch.float32)  # (B, S)
            B2, S = step_r.shape
            if B2 != B:
                # If mismatch, ignore step shaping to avoid shape errors
                step_r = None
            else:
                # Leave-one-out baseline per (group, step_index)
                step_baseline = torch.zeros_like(step_r)
                for g, idxs in group_map.items():
                    idx = torch.tensor(idxs, dtype=torch.long, device=device)
                    n = idx.numel()
                    step_vals = step_r.index_select(0, idx)  # (n, S)
                    if n > 1:
                        s = step_vals.sum(dim=0, keepdim=True)  # (1, S)
                        b = (s - step_vals) / (n - 1)
                        step_baseline.index_copy_(0, idx, b)
                    else:
                        step_baseline.index_fill_(0, idx, 0.0)
                step_adv = step_r - step_baseline  # (B, S)
                # Discount across steps to emphasize later decisions
                with torch.no_grad():
                    discount = torch.pow(torch.full((S,), gamma, device=device, dtype=torch.float32),
                                         torch.arange(S, device=device, dtype=torch.float32))
                step_adv_scalar = (step_adv * discount.unsqueeze(0)).sum(dim=1)
                # Normalize within group
                step_adv_scalar = _whiten_by_group(step_adv_scalar, group_map)

        # Combine sequence and step-level advantages
        coef = float(cfg.get("step_adv_coef", 0.7))
        total_adv_scalar = seq_adv + coef * step_adv_scalar

        # Optional final group-wise whitening for stability
        if cfg.get("final_whiten_by_group", True):
            total_adv_scalar = _whiten_by_group(total_adv_scalar, group_map)

        # Clip extreme advantages to stabilize training
        adv_clip_value = cfg.get("adv_clip_value", None)
        if adv_clip_value is not None and adv_clip_value > 0:
            total_adv_scalar = torch.clamp(total_adv_scalar, -adv_clip_value, adv_clip_value)

        # Broadcast scalar advantages to token-level and mask
        advantages = total_adv_scalar.unsqueeze(1) * mask

        # Compute token-level returns (discounted return-to-go over token rewards)
        returns = _discount_cumsum_tokenwise(rewards * mask, gamma=gamma) * mask

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
        cfg = self.config if hasattr(self, "config") else {}
        clip_ratio = float(cfg.get("clip_ratio", clip_ratio))
        use_kl = bool(cfg.get("use_kl_loss", False))
        kl_coef = float(cfg.get("kl_loss_coef", 0.01))
        normalize_adv = bool(cfg.get("normalize_adv", True))

        mask = response_mask.to(dtype=torch.float32)
        adv = advantages
        if normalize_adv:
            adv = masked_whiten(adv, mask)

        # PPO clipped objective at token level
        ratio = torch.exp((log_prob - old_log_prob).clamp(min=-20, max=20))
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
        pg_loss = -masked_mean(torch.minimum(surr1, surr2), mask)

        # Approx KL penalty (optional)
        approx_kl = masked_mean(old_log_prob - log_prob, mask)
        loss = pg_loss + (kl_coef * approx_kl if use_kl else 0.0)

        # Diagnostics
        clip_frac = masked_mean((torch.abs(ratio - 1.0) > clip_ratio).float(), mask)
        metrics = {
            "loss": loss.detach().item(),
            "policy_loss": pg_loss.detach().item(),
            "approx_kl": approx_kl.detach().item(),
            "clip_frac": clip_frac.detach().item(),
        }
        if use_kl:
            metrics["kl_coef"] = kl_coef

        return loss, metrics

    def assign_step_rewards(
        self,
        episode_reward: float,
        trajectory_length: int,
        step_observations: list,
        step_actions: list,
        **kwargs
    ) -> np.ndarray:
        L = int(trajectory_length) if trajectory_length is not None else len(step_actions) if step_actions is not None else len(step_observations)
        L = max(1, L)
        episode_reward = float(episode_reward)

        # If failure, return zeros to keep unbiased estimator
        if episode_reward <= 0.0:
            return np.zeros(L, dtype=np.float32)

        # Base weights encourage later steps slightly
        t = np.arange(L, dtype=np.float32)
        time_boost = 1.02 ** t  # increasing weight for later steps
        weights = np.full(L, 0.05, dtype=np.float32) * time_boost

        # Action-based shaping
        actions = step_actions if step_actions is not None and len(step_actions) == L else [""] * L
        observations = step_observations if step_observations is not None and len(step_observations) == L else [""] * L

        action_keywords = [
            ("turn on", 0.5), ("switch on", 0.5), ("turn off", 0.5), ("switch off", 0.5),
            ("take", 1.0), ("grab", 1.0), ("pick", 1.0), ("pickup", 1.0),
            ("open", 0.6), ("close", 0.3),
            ("put", 1.2), ("place", 1.2), ("insert", 1.2),
            ("drop", 0.2),
            ("go", 0.2), ("walk", 0.2), ("move", 0.2),
            ("examine", 0.05), ("look", 0.05),
            ("clean", 1.0), ("wash", 1.0), ("rinse", 1.0),
            ("heat", 1.0), ("cook", 1.0), ("microwave", 1.0), ("warm", 1.0),
            ("cool", 1.0), ("freeze", 1.0), ("refrigerate", 1.0),
            ("slice", 1.0), ("cut", 1.0), ("chop", 1.0),
            ("pour", 0.8), ("fill", 0.8),
            ("fridge", 0.3), ("sink", 0.2), ("microwave", 0.4), ("stove", 0.4),
        ]

        obs_keywords = [
            ("you put", 1.5), ("you place", 1.5), ("inside", 0.7), ("in the", 0.5),
            ("opened", 0.7), ("closed", 0.3),
            ("clean", 1.0), ("heated", 1.0), ("cooled", 1.0),
            ("success", 2.0), ("completed", 2.0), ("succeeded", 2.0),
        ]

        for i in range(L):
            aw = 0.0
            ow = 0.0
            a = actions[i].lower() if isinstance(actions[i], str) else ""
            o = observations[i].lower() if isinstance(observations[i], str) else ""
            for kw, w in action_keywords:
                if kw in a:
                    aw += w
            for kw, w in obs_keywords:
                if kw in o:
                    ow += w
            weights[i] += (aw + 0.5 * ow) * time_boost[i]

        # Ensure final steps have higher share (credit assignment)
        weights[-1] += 2.0 * time_boost[-1]
        if L >= 2:
            weights[-2] += 0.8 * time_boost[-2]

        wsum = float(weights.sum())
        if wsum <= 1e-8:
            step_rewards = np.zeros(L, dtype=np.float32)
            step_rewards[-1] = np.float32(episode_reward)
            return step_rewards

        step_rewards = (weights / wsum) * episode_reward
        return step_rewards.astype(np.float32)