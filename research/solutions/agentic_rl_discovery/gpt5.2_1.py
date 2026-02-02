import re
from typing import Tuple, Dict, Any, Optional, List
from collections import defaultdict

import numpy as np
import torch

try:
    from verl.utils.torch_functional import masked_mean as _masked_mean
except Exception:
    _masked_mean = None

try:
    from verl.utils.torch_functional import masked_whiten as _masked_whiten
except Exception:
    _masked_whiten = None


def _safe_masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if _masked_mean is not None:
        return _masked_mean(x, mask)
    m = mask.to(dtype=x.dtype)
    denom = m.sum().clamp_min(eps)
    return (x * m).sum() / denom


def _safe_masked_whiten(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if _masked_whiten is not None:
        return _masked_whiten(x, mask)
    m = mask.to(dtype=x.dtype)
    denom = m.sum().clamp_min(eps)
    mean = (x * m).sum() / denom
    var = ((x - mean) ** 2 * m).sum() / denom
    return (x - mean) / torch.sqrt(var + eps)


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        self.config = {
            "gamma": 0.95,
            "clip_ratio": 0.2,
            "use_kl_loss": False,
            "kl_loss_coef": 0.01,
            "adv_eps": 1e-6,
            "adv_clip": 5.0,
            "use_global_adv_whiten": False,
            "rloo_baseline": True,
            "reward_living_penalty": -0.003,
            "reward_invalid_penalty": -0.02,
            "reward_repeat_penalty": -0.01,
            "reward_progress_bonus": 0.01,
            "reward_length_penalty": -0.001,
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
        device = token_level_rewards.device
        dtype = token_level_rewards.dtype
        mask = response_mask.to(dtype=dtype)

        B, S = token_level_rewards.shape
        eps = float(kwargs.get("adv_eps", self.config.get("adv_eps", 1e-6)))
        adv_clip = float(kwargs.get("adv_clip", self.config.get("adv_clip", 5.0)))
        use_global_adv_whiten = bool(kwargs.get("use_global_adv_whiten", self.config.get("use_global_adv_whiten", False)))
        rloo = bool(kwargs.get("rloo_baseline", self.config.get("rloo_baseline", True)))

        use_gamma = float(kwargs.get("gamma", gamma if gamma is not None else self.config.get("gamma", 0.95)))

        returns_scalar = None

        if step_rewards is not None and torch.is_tensor(step_rewards):
            if step_rewards.shape == token_level_rewards.shape:
                returns_scalar = (step_rewards.to(device=device, dtype=dtype) * mask).sum(dim=1)
            elif step_rewards.dim() == 2 and step_rewards.shape[0] == B:
                sr = step_rewards.to(device=device, dtype=dtype)
                T = sr.shape[1]
                if T > 0:
                    discounts = (use_gamma ** torch.arange(T, device=device, dtype=dtype)).unsqueeze(0)
                    returns_scalar = (sr * discounts).sum(dim=1)
                else:
                    returns_scalar = torch.zeros(B, device=device, dtype=dtype)
            else:
                returns_scalar = (token_level_rewards * mask).sum(dim=1)
        else:
            returns_scalar = (token_level_rewards * mask).sum(dim=1)

        adv_scalar = torch.zeros_like(returns_scalar)

        ep = np.asarray(episode_index)
        uniq_eps = np.unique(ep)
        for g in uniq_eps:
            idx_np = np.nonzero(ep == g)[0]
            if idx_np.size == 0:
                continue
            idx = torch.as_tensor(idx_np, device=device, dtype=torch.long)
            r = returns_scalar.index_select(0, idx)

            n = int(r.numel())
            mean = r.mean()
            std = r.std(unbiased=False)

            if rloo and n > 1:
                s = r.sum()
                baseline = (s - r) / float(n - 1)
                adv = r - baseline
            else:
                adv = r - mean

            adv = adv / (std + eps)

            if adv_clip is not None and adv_clip > 0:
                adv = torch.clamp(adv, -adv_clip, adv_clip)

            adv_scalar.index_copy_(0, idx, adv)

        if use_global_adv_whiten:
            adv_tokens = adv_scalar[:, None].expand(B, S) * mask
            adv_tokens = _safe_masked_whiten(adv_tokens, mask)
            adv_tokens = torch.clamp(adv_tokens, -adv_clip, adv_clip) if (adv_clip is not None and adv_clip > 0) else adv_tokens
            ret_tokens = returns_scalar[:, None].expand(B, S) * mask
            return adv_tokens, ret_tokens

        advantages = adv_scalar[:, None].expand(B, S) * mask
        returns = returns_scalar[:, None].expand(B, S) * mask
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
        dtype = log_prob.dtype
        mask = response_mask.to(dtype=dtype)

        adv = advantages.detach()
        adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)

        adv_norm = bool(kwargs.get("adv_norm", True))
        if adv_norm:
            adv = _safe_masked_whiten(adv, mask)

        diff = (log_prob - old_log_prob).clamp(-20.0, 20.0)
        ratio = torch.exp(diff)

        clip = float(kwargs.get("clip_ratio", clip_ratio if clip_ratio is not None else self.config.get("clip_ratio", 0.2)))
        ratio_clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)

        pg_loss1 = -ratio * adv
        pg_loss2 = -ratio_clipped * adv
        pg_loss = torch.maximum(pg_loss1, pg_loss2)

        loss = _safe_masked_mean(pg_loss, mask)

        clip_frac = _safe_masked_mean(((ratio - 1.0).abs() > clip).to(dtype=dtype), mask).detach()
        approx_kl = _safe_masked_mean((old_log_prob - log_prob), mask).detach()
        ratio_mean = _safe_masked_mean(ratio, mask).detach()
        adv_mean = _safe_masked_mean(adv, mask).detach()

        use_kl_loss = bool(kwargs.get("use_kl_loss", self.config.get("use_kl_loss", False)))
        kl_loss = None
        if use_kl_loss:
            ref_log_prob = kwargs.get("ref_log_prob", None)
            if ref_log_prob is not None and torch.is_tensor(ref_log_prob):
                kl = (log_prob - ref_log_prob).clamp(-20.0, 20.0)
                kl_loss = _safe_masked_mean(kl, mask)
                kl_coef = float(kwargs.get("kl_loss_coef", self.config.get("kl_loss_coef", 0.01)))
                loss = loss + kl_coef * kl_loss

        metrics: Dict[str, Any] = {
            "clip_frac": clip_frac,
            "approx_kl": approx_kl,
            "ratio_mean": ratio_mean,
            "adv_mean": adv_mean,
        }
        if kl_loss is not None:
            metrics["kl_loss"] = kl_loss.detach()
        return loss, metrics

    def assign_step_rewards(
        self,
        episode_reward: float,
        trajectory_length: int,
        step_observations: list,
        step_actions: list,
        **kwargs
    ) -> np.ndarray:
        L = int(trajectory_length)
        if L <= 0:
            return np.zeros((0,), dtype=np.float32)

        living = float(kwargs.get("reward_living_penalty", self.config.get("reward_living_penalty", -0.003)))
        invalid_pen = float(kwargs.get("reward_invalid_penalty", self.config.get("reward_invalid_penalty", -0.02)))
        repeat_pen = float(kwargs.get("reward_repeat_penalty", self.config.get("reward_repeat_penalty", -0.01)))
        prog_bonus = float(kwargs.get("reward_progress_bonus", self.config.get("reward_progress_bonus", 0.01)))
        length_pen = float(kwargs.get("reward_length_penalty", self.config.get("reward_length_penalty", -0.001)))

        rewards = np.full((L,), living, dtype=np.float32)

        neg_patterns = (
            r"don't understand",
            r"do not understand",
            r"can't do that",
            r"cannot do that",
            r"can't",
            r"cannot",
            r"not possible",
            r"doesn't seem possible",
            r"nothing happens",
            r"not sure",
            r"invalid",
            r"you can't",
            r"you cannot",
            r"i don't",
            r"you aren't able",
        )
        pos_patterns = (
            r"you (pick|picked) up",
            r"you (take|took)",
            r"you (put|placed)",
            r"you (open|opened)",
            r"you (close|closed)",
            r"you (turn|turned) on",
            r"you (turn|turned) off",
            r"you (clean|cleaned)",
            r"you (heat|heated)",
            r"you (cool|cooled)",
            r"you (slice|sliced)",
            r"you (wash|washed)",
        )
        neg_re = re.compile("|".join(neg_patterns), flags=re.IGNORECASE)
        pos_re = re.compile("|".join(pos_patterns), flags=re.IGNORECASE)

        prev_action = None
        for t in range(L):
            act = ""
            if step_actions is not None and t < len(step_actions) and step_actions[t] is not None:
                act = str(step_actions[t]).strip()

            obs = ""
            if step_observations is not None and t < len(step_observations) and step_observations[t] is not None:
                obs = str(step_observations[t])

            if act:
                if prev_action is not None and act == prev_action:
                    rewards[t] += repeat_pen
                prev_action = act

                if len(act) > 60:
                    rewards[t] += length_pen

            if obs:
                if neg_re.search(obs) is not None:
                    rewards[t] += invalid_pen
                if pos_re.search(obs) is not None:
                    rewards[t] += prog_bonus

        if float(episode_reward) > 0.0:
            rewards[-1] += float(episode_reward)

        return rewards.astype(np.float32)