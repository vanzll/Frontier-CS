import re
from typing import Tuple, Dict, Any, Optional, List
from collections import defaultdict

import numpy as np
import torch

try:
    from verl.utils.torch_functional import masked_mean as _masked_mean  # type: ignore
    from verl.utils.torch_functional import masked_whiten as _masked_whiten  # type: ignore
except Exception:
    _masked_mean = None
    _masked_whiten = None


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, eps: float = 1e-8) -> torch.Tensor:
    mask_f = mask.to(dtype=x.dtype)
    if dim is None:
        denom = mask_f.sum().clamp_min(eps)
        return (x * mask_f).sum() / denom
    denom = mask_f.sum(dim=dim).clamp_min(eps)
    return (x * mask_f).sum(dim=dim) / denom


def masked_whiten(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    mean = (x * mask_f).sum() / denom
    var = ((x - mean) * mask_f).pow(2).sum() / denom
    y = (x - mean) / torch.sqrt(var + eps)
    return y * mask_f


def _anchor_key(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        return ("bytes", hash(x))
    if isinstance(x, (list, tuple)):
        return ("list", tuple(_anchor_key(v) for v in x))
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            try:
                return ("obj", tuple(_anchor_key(v) for v in x.tolist()))
            except Exception:
                return ("obj_repr", repr(x))
        try:
            return ("nd", x.shape, str(x.dtype), hash(x.tobytes()))
        except Exception:
            return ("nd_repr", repr(x))
    return ("repr", repr(x))


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        self.config = {
            "gamma": 0.95,
            "clip_ratio": 0.2,
            "use_kl_loss": True,
            "kl_loss_coef": 0.005,
            "adv_whiten_tokens": True,
            "use_anchor_grouping": True,
            "anchor_fallback_to_episode": True,
            "group_norm_eps": 1e-6,
        }
        return self

    def _total_reward_from_step_rewards(
        self,
        step_rewards: torch.Tensor,
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        device = token_level_rewards.device
        dtype = token_level_rewards.dtype

        if step_rewards is None:
            return (token_level_rewards * response_mask.to(dtype=dtype)).sum(dim=1)

        if not torch.is_tensor(step_rewards):
            step_rewards = torch.as_tensor(step_rewards, device=device, dtype=dtype)
        else:
            step_rewards = step_rewards.to(device=device, dtype=dtype)

        if step_rewards.dim() == 2:
            if step_rewards.shape == token_level_rewards.shape:
                return (step_rewards * response_mask.to(dtype=dtype)).sum(dim=1)
            B, S = step_rewards.shape
            if S == 0:
                return torch.zeros((B,), device=device, dtype=dtype)
            powers = (gamma ** torch.arange(S, device=device, dtype=dtype)).view(1, -1)
            return (step_rewards * powers).sum(dim=1)

        if step_rewards.dim() == 1:
            return step_rewards

        step_rewards_flat = step_rewards.view(step_rewards.shape[0], -1)
        S = step_rewards_flat.shape[1]
        if S == 0:
            return torch.zeros((step_rewards_flat.shape[0],), device=device, dtype=dtype)
        powers = (gamma ** torch.arange(S, device=device, dtype=dtype)).view(1, -1)
        return (step_rewards_flat * powers).sum(dim=1)

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
        dtype = token_level_rewards.dtype

        B, T = token_level_rewards.shape
        mask = response_mask.to(device=device, dtype=dtype)

        total_reward = self._total_reward_from_step_rewards(step_rewards, token_level_rewards, response_mask, gamma=gamma)
        total_reward = total_reward.to(device=device, dtype=dtype)

        ep = torch.as_tensor(episode_index, device=device, dtype=torch.long).view(-1)
        if ep.numel() != B:
            ep = ep[:B] if ep.numel() > B else torch.nn.functional.pad(ep, (0, B - ep.numel()), value=0)

        use_anchor = bool(anchor_observations is not None) and bool(self.config.get("use_anchor_grouping", True))
        keys: List[Any]
        if use_anchor:
            try:
                anchors = anchor_observations
                if isinstance(anchors, np.ndarray) and anchors.shape[0] != B:
                    anchors = anchors[:B] if anchors.shape[0] > B else np.pad(anchors, (0, B - anchors.shape[0]), mode="edge")
                if isinstance(anchors, np.ndarray) and anchors.dtype == object:
                    anchors_list = anchors.tolist()
                elif isinstance(anchors, np.ndarray) and anchors.ndim == 1:
                    anchors_list = anchors.tolist()
                else:
                    anchors_list = [anchors[i] for i in range(B)]
            except Exception:
                anchors_list = [None] * B
            keys = [(int(ep[i].item()), _anchor_key(anchors_list[i])) for i in range(B)]
        else:
            keys = [(int(ep[i].item()), None) for i in range(B)]

        groups = defaultdict(list)
        for i, k in enumerate(keys):
            groups[k].append(i)

        if use_anchor and bool(self.config.get("anchor_fallback_to_episode", True)):
            for k, idxs in list(groups.items()):
                if len(idxs) <= 1:
                    del groups[k]
            if len(groups) == 0:
                groups = defaultdict(list)
                for i in range(B):
                    groups[(int(ep[i].item()), None)].append(i)

        adv = torch.zeros((B,), device=device, dtype=dtype)
        eps = float(self.config.get("group_norm_eps", 1e-6))

        for _, idxs in groups.items():
            if len(idxs) == 0:
                continue
            idx_t = torch.tensor(idxs, device=device, dtype=torch.long)
            r = total_reward.index_select(0, idx_t)

            n = r.numel()
            if n == 1:
                adv_i = torch.zeros_like(r)
            else:
                sum_r = r.sum()
                baseline = (sum_r - r) / (n - 1)
                adv_i = r - baseline

            if n > 1:
                std = r.std(unbiased=False).clamp_min(eps)
                adv_i = adv_i / std

            adv.index_copy_(0, idx_t, adv_i)

        adv_tok = (adv.view(B, 1) * mask).to(dtype=dtype)
        ret_tok = (total_reward.view(B, 1) * mask).to(dtype=dtype)
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
        mask = response_mask.to(dtype=log_prob.dtype, device=log_prob.device)
        adv = advantages.to(dtype=log_prob.dtype, device=log_prob.device)

        if bool(self.config.get("adv_whiten_tokens", True)):
            if _masked_whiten is not None:
                adv = _masked_whiten(adv, mask)
            else:
                adv = masked_whiten(adv, mask)

        log_ratio = (log_prob - old_log_prob)
        ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))

        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
        obj = torch.minimum(unclipped, clipped)

        if _masked_mean is not None:
            pg_loss = -_masked_mean(obj, mask)
            clip_frac = _masked_mean(((ratio - 1.0).abs() > clip_ratio).to(dtype=log_prob.dtype), mask)
            approx_kl = _masked_mean((old_log_prob - log_prob), mask)
        else:
            pg_loss = -masked_mean(obj, mask)
            clip_frac = masked_mean(((ratio - 1.0).abs() > clip_ratio).to(dtype=log_prob.dtype), mask)
            approx_kl = masked_mean((old_log_prob - log_prob), mask)

        loss = pg_loss

        if bool(self.config.get("use_kl_loss", False)):
            ref_log_prob = kwargs.get("ref_log_prob", None)
            if ref_log_prob is None:
                ref_log_prob = kwargs.get("reference_log_prob", None)
            if ref_log_prob is not None:
                ref_log_prob = ref_log_prob.to(device=log_prob.device, dtype=log_prob.dtype)
                if _masked_mean is not None:
                    kl = _masked_mean((log_prob - ref_log_prob), mask)
                else:
                    kl = masked_mean((log_prob - ref_log_prob), mask)
                loss = loss + float(self.config.get("kl_loss_coef", 0.01)) * kl
            else:
                kl = approx_kl.detach()
        else:
            kl = approx_kl.detach()

        metrics: Dict[str, Any] = {
            "clip_frac": float(clip_frac.detach().cpu().item()) if torch.is_tensor(clip_frac) else float(clip_frac),
            "approx_kl": float(approx_kl.detach().cpu().item()) if torch.is_tensor(approx_kl) else float(approx_kl),
            "kl_term": float(kl.detach().cpu().item()) if torch.is_tensor(kl) else float(kl),
            "pg_loss": float(pg_loss.detach().cpu().item()) if torch.is_tensor(pg_loss) else float(pg_loss),
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
        T = int(trajectory_length) if trajectory_length is not None else 0
        if T <= 0:
            return np.zeros((0,), dtype=np.float32)

        obs = step_observations if step_observations is not None else []
        acts = step_actions if step_actions is not None else []
        L = min(T, len(obs), len(acts)) if (len(obs) > 0 and len(acts) > 0) else T

        rewards = np.zeros((T,), dtype=np.float32)

        living_penalty = float(kwargs.get("living_penalty", -0.002))
        look_penalty = float(kwargs.get("look_penalty", -0.001))
        repeat_penalty = float(kwargs.get("repeat_penalty", -0.002))
        invalid_penalty = float(kwargs.get("invalid_penalty", -0.01))
        progress_bonus = float(kwargs.get("progress_bonus", 0.004))

        if living_penalty != 0.0:
            rewards[:] += living_penalty

        pos_pat = re.compile(
            r"(you\s+(take|pick up|put|place|open|close|turn on|turn off|unlock|lock|clean|wash|cook|heat|cool|slice|cut))",
            re.IGNORECASE,
        )
        neg_pat = re.compile(
            r"(can't|cannot|don't know|not possible|doesn't seem|won't work|you are not holding|you aren't holding|you don't have|nothing happens|invalid|unknown)",
            re.IGNORECASE,
        )

        def _norm_action(a: str) -> str:
            a = (a or "").strip().lower()
            a = re.sub(r"\s+", " ", a)
            return a

        prev_a = None
        for t in range(L):
            o = obs[t] if t < len(obs) else ""
            a = acts[t] if t < len(acts) else ""

            a_norm = _norm_action(a)
            o_txt = (o or "")

            if a_norm in ("look", "inventory"):
                rewards[t] += look_penalty

            if prev_a is not None and a_norm == prev_a and a_norm != "":
                rewards[t] += repeat_penalty
            prev_a = a_norm

            if pos_pat.search(o_txt) is not None:
                rewards[t] += progress_bonus
            if neg_pat.search(o_txt) is not None:
                rewards[t] += invalid_penalty

        er = float(episode_reward)
        if er > 0.0:
            spread = float(kwargs.get("success_spread", 0.3))
            spread = max(0.0, min(1.0, spread))
            g_spread = float(kwargs.get("spread_gamma", 0.95))
            g_spread = max(0.0, min(0.9999, g_spread))

            weights = np.power(g_spread, np.arange(T - 1, -1, -1, dtype=np.float32))
            wsum = float(weights.sum()) if float(weights.sum()) > 0 else 1.0
            weights = weights / wsum
            rewards += (er * spread) * weights
            rewards[T - 1] += er * (1.0 - spread)

        return rewards.astype(np.float32)