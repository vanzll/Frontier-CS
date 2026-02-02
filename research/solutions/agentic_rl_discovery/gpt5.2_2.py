import re
from typing import Tuple, Dict, Any, Optional, List
from collections import defaultdict

import numpy as np
import torch

try:
    from verl.utils.torch_functional import masked_mean as _masked_mean_impl
    from verl.utils.torch_functional import masked_whiten as _masked_whiten_impl
except Exception:
    _masked_mean_impl = None
    _masked_whiten_impl = None


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    if dim is None:
        num = (x * mask).sum()
        den = mask.sum().clamp_min(eps)
        return num / den
    num = (x * mask).sum(dim=dim, keepdim=keepdim)
    den = mask.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return num / den


def masked_whiten(x: torch.Tensor, mask: torch.Tensor, dim=None, eps: float = 1e-8) -> torch.Tensor:
    m = masked_mean(x, mask, dim=dim, keepdim=True, eps=eps)
    v = masked_mean((x - m) ** 2, mask, dim=dim, keepdim=True, eps=eps)
    return (x - m) / torch.sqrt(v + eps)


if _masked_mean_impl is not None:
    masked_mean = _masked_mean_impl  # type: ignore
if _masked_whiten_impl is not None:
    masked_whiten = _masked_whiten_impl  # type: ignore


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        self.config = {
            "gamma": 0.95,
            "clip_ratio": 0.2,
            "use_kl_loss": True,
            "kl_loss_coef": 0.003,
            "reward_gamma": 0.97,
            "shape_coef": 0.05,
            "invalid_penalty": 0.01,
            "valid_bonus": 0.006,
            "adv_eps": 1e-6,
            "ratio_clip_log": 10.0,
        }
        return self

    @staticmethod
    def _anchor_key(x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, (bytes, str, int, float, bool, tuple)):
            return x
        if isinstance(x, np.generic):
            try:
                return x.item()
            except Exception:
                return str(x)
        if isinstance(x, np.ndarray):
            arr = np.ascontiguousarray(x)
            return (arr.dtype.str, arr.shape, arr.tobytes())
        try:
            return (type(x).__name__, str(x))
        except Exception:
            return (type(x).__name__, repr(x))

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
        dtype = torch.float32

        rmask = response_mask.to(device=device, dtype=dtype)
        rewards_tok = token_level_rewards.to(device=device, dtype=dtype)

        reward_from_tokens = (rewards_tok * rmask).sum(dim=1)

        r = reward_from_tokens
        if step_rewards is not None:
            sr = step_rewards
            if not torch.is_tensor(sr):
                sr = torch.as_tensor(sr)
            sr = sr.to(device=device, dtype=dtype)
            if sr.ndim == 1 and sr.shape[0] == r.shape[0]:
                r = r + sr
            elif sr.ndim == 2 and sr.shape == rewards_tok.shape:
                r = r + (sr * rmask).sum(dim=1)
            elif sr.ndim == 2 and sr.shape[0] == r.shape[0] and sr.shape[1] == 1:
                r = r + sr.squeeze(1)

        bsz = r.shape[0]
        if bsz == 0:
            adv = rewards_tok.new_zeros(rewards_tok.shape, dtype=dtype)
            ret = rewards_tok.new_zeros(rewards_tok.shape, dtype=dtype)
            return adv, ret

        ep = torch.as_tensor(episode_index, device=device, dtype=torch.long)
        uniq_eps = torch.unique(ep)

        adv_raw = torch.zeros_like(r)

        # Episode-level leave-one-out baseline
        for e in uniq_eps.tolist():
            idx = (ep == e).nonzero(as_tuple=False).squeeze(-1)
            n = int(idx.numel())
            if n <= 0:
                continue
            ri = r.index_select(0, idx)
            if n == 1:
                adv_e = ri - ri
            else:
                sum_r = ri.sum()
                baseline_loo = (sum_r - ri) / (n - 1)
                adv_e = ri - baseline_loo
            adv_raw.index_copy_(0, idx, adv_e)

        # Optional anchor-level baseline inside episode
        if anchor_observations is not None:
            try:
                anchors = anchor_observations
                if isinstance(anchors, np.ndarray) and anchors.shape[0] == bsz:
                    anchor_keys = [self._anchor_key(anchors[i]) for i in range(bsz)]
                else:
                    anchor_keys = [self._anchor_key(x) for x in list(anchors)]
                    if len(anchor_keys) != bsz:
                        anchor_keys = anchor_keys[:bsz] + [None] * max(0, bsz - len(anchor_keys))
            except Exception:
                anchor_keys = [None] * bsz

            groups = defaultdict(list)
            ep_cpu = ep.detach().cpu().tolist()
            for i in range(bsz):
                groups[(ep_cpu[i], anchor_keys[i])].append(i)

            for (e, _a), idx_list in groups.items():
                if len(idx_list) < 2:
                    continue
                idx_t = torch.as_tensor(idx_list, device=device, dtype=torch.long)
                vals = adv_raw.index_select(0, idx_t)
                m = vals.mean()
                adv_raw.index_copy_(0, idx_t, vals - m)

        # Normalize within episode for stability
        adv = torch.zeros_like(adv_raw)
        eps = float(self.config.get("adv_eps", 1e-6))
        for e in uniq_eps.tolist():
            idx = (ep == e).nonzero(as_tuple=False).squeeze(-1)
            n = int(idx.numel())
            if n <= 0:
                continue
            v = adv_raw.index_select(0, idx)
            m = v.mean()
            s = v.std(unbiased=False).clamp_min(eps)
            adv.index_copy_(0, idx, (v - m) / s)

        advantages = (adv[:, None] * rmask).to(dtype=dtype)
        advantages = masked_whiten(advantages, rmask, dim=None)

        returns = (r[:, None] * rmask).to(dtype=dtype)

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
        mask = response_mask.to(dtype=torch.float32, device=log_prob.device)
        adv = advantages.to(dtype=torch.float32, device=log_prob.device)
        logp = log_prob.to(dtype=torch.float32)
        old_logp = old_log_prob.to(dtype=torch.float32)

        diff = (logp - old_logp).clamp(min=-float(self.config.get("ratio_clip_log", 10.0)),
                                       max=float(self.config.get("ratio_clip_log", 10.0)))
        ratio = torch.exp(diff)

        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
        pg_obj = torch.minimum(unclipped, clipped)

        loss = -masked_mean(pg_obj, mask, dim=None)

        with torch.no_grad():
            clip_frac = masked_mean(((ratio - 1.0).abs() > clip_ratio).to(torch.float32), mask, dim=None)
            approx_kl = masked_mean((old_logp - logp), mask, dim=None)
            mean_ratio = masked_mean(ratio, mask, dim=None)
            adv_mean = masked_mean(adv, mask, dim=None)
            adv_std = torch.sqrt(masked_mean((adv - adv_mean) ** 2, mask, dim=None) + 1e-8)

        metrics: Dict[str, Any] = {
            "clip_frac": clip_frac,
            "approx_kl": approx_kl,
            "mean_ratio": mean_ratio,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
        }

        use_kl = bool(self.config.get("use_kl_loss", False))
        if use_kl:
            ref_log_prob = kwargs.get("ref_log_prob", None)
            if ref_log_prob is not None:
                ref_logp = ref_log_prob.to(dtype=torch.float32, device=log_prob.device)
                kl_sample = (logp - ref_logp)
                kl_to_ref = masked_mean(kl_sample, mask, dim=None)
                kl_coef = float(self.config.get("kl_loss_coef", 0.0))
                loss = loss + kl_coef * kl_to_ref
                metrics["kl_to_ref"] = kl_to_ref
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
        T = int(trajectory_length)
        if T <= 0:
            return np.zeros((0,), dtype=np.float32)

        obs = step_observations if isinstance(step_observations, list) else list(step_observations)
        act = step_actions if isinstance(step_actions, list) else list(step_actions)
        if len(obs) < T:
            obs = obs + [""] * (T - len(obs))
        if len(act) < T:
            act = act + [""] * (T - len(act))
        if len(obs) > T:
            obs = obs[-T:]
        if len(act) > T:
            act = act[-T:]

        rg = float(self.config.get("reward_gamma", 0.97))
        shape_coef = float(self.config.get("shape_coef", 0.05))
        invalid_penalty = float(self.config.get("invalid_penalty", 0.01))
        valid_bonus = float(self.config.get("valid_bonus", 0.006))

        # Base success reward distribution
        base = np.zeros((T,), dtype=np.float32)
        if episode_reward > 0:
            weights = np.array([rg ** (T - 1 - t) for t in range(T)], dtype=np.float32)
            s = float(weights.sum())
            if s > 0:
                base = (weights / s) * float(episode_reward)
            else:
                base[-1] = float(episode_reward)

        invalid_phrases = (
            "don't understand",
            "do not understand",
            "cannot",
            "can't",
            "not possible",
            "doesn't seem possible",
            "does not seem possible",
            "you don't have",
            "you do not have",
            "you aren't holding",
            "you are not holding",
            "nothing happens",
            "won't budge",
            "not sure",
        )
        positive_patterns = (
            "you open",
            "you close",
            "you take",
            "you pick up",
            "you put",
            "you place",
            "you insert",
            "you pour",
            "you fill",
            "you turn on",
            "you turn off",
            "you switch on",
            "you switch off",
            "you clean",
            "you wash",
            "you slice",
            "you cut",
            "you unlock",
            "you lock",
        )

        allowed_verbs = (
            "go", "open", "close", "take", "pick", "put", "place", "insert", "pour", "fill",
            "toggle", "turn", "switch", "clean", "wash", "slice", "cut", "heat", "cool", "use",
            "examine", "look", "inventory"
        )
        verb_re = re.compile(r"^\s*(" + "|".join(re.escape(v) for v in allowed_verbs) + r")\b", re.IGNORECASE)

        shaping = np.zeros((T,), dtype=np.float32)
        for t in range(T):
            o = (obs[t] or "")
            a = (act[t] or "")
            ol = o.lower()
            al = a.lower()

            score = 0.0
            if any(p in ol for p in invalid_phrases):
                score -= invalid_penalty
            if any(p in ol for p in positive_patterns):
                score += valid_bonus

            # Encourage well-formed action commands; discourage verbose reasoning
            if verb_re.search(al) is not None:
                score += 0.002
            if "think" in al or "reason" in al or "\n" in a:
                score -= 0.003
            if len(a) > 120:
                score -= 0.003

            shaping[t] = float(np.clip(score, -0.02, 0.02))

        step_rewards = base + (shape_coef * shaping).astype(np.float32)

        # Keep success trajectories clearly above failures
        if episode_reward <= 0:
            step_rewards = np.clip(step_rewards, -0.02, 0.02).astype(np.float32)
        else:
            # Prevent excessive deviation of total reward from episode_reward
            total = float(step_rewards.sum())
            if total > 0:
                step_rewards = (step_rewards * (float(episode_reward) / total)).astype(np.float32)

        return step_rewards.astype(np.float32)