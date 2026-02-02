import math
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import torch

try:
    from verl.utils.torch_functional import masked_mean, masked_whiten
except Exception:
    def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False, eps: float = 1e-8) -> torch.Tensor:
        mask_f = mask.to(dtype=x.dtype)
        if dim is None:
            denom = mask_f.sum().clamp_min(eps)
            return (x * mask_f).sum() / denom
        denom = mask_f.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
        return (x * mask_f).sum(dim=dim, keepdim=keepdim) / denom

    def masked_whiten(
        x: torch.Tensor,
        mask: torch.Tensor,
        shift_mean: bool = True,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        mask_f = mask.to(dtype=x.dtype)
        mean = masked_mean(x, mask_f, dim=None, keepdim=False, eps=eps) if shift_mean else torch.zeros((), device=x.device, dtype=x.dtype)
        var = masked_mean((x - mean) ** 2, mask_f, dim=None, keepdim=False, eps=eps)
        std = torch.sqrt(var + eps)
        return (x - mean) / std


class Solution:
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self._anchor_prime_vec: Optional[np.ndarray] = None

    def solve(self, spec_path: str = None) -> "Solution":
        self.config = {
            "gamma": 0.95,
            "clip_ratio": 0.2,
            "use_kl_loss": True,
            "kl_loss_coef": 0.01,
            "whiten_advantages": False,
            "adv_eps": 1e-6,
            "adv_clip": 5.0,
            "traj_adv_coef": 1.0,
            "step_adv_coef": 0.35,
            "step_penalty": 0.005,
            "invalid_action_penalty": 0.01,
            "progress_bonus": 0.01,
            "repeat_action_penalty": 0.003,
            "success_reward_total": 1.0,
            "success_reward_discount": 0.97,
        }
        return self

    @staticmethod
    def _scatter_sum(values: torch.Tensor, inv: torch.Tensor, n_groups: int) -> torch.Tensor:
        out = torch.zeros((n_groups,), device=values.device, dtype=values.dtype)
        out.scatter_add_(0, inv, values)
        return out

    def _hash_anchor_obs(self, anchor_observations: np.ndarray, episode_index: Optional[np.ndarray] = None) -> np.ndarray:
        if anchor_observations is None:
            return None
        if not isinstance(anchor_observations, np.ndarray):
            anchor_observations = np.asarray(anchor_observations)

        if anchor_observations.ndim == 1:
            if anchor_observations.dtype.kind in ("i", "u"):
                return anchor_observations.astype(np.int64, copy=False)
            if anchor_observations.dtype.kind == "f":
                return anchor_observations.view(np.int64)
            if anchor_observations.dtype.kind in ("S", "U"):
                it = anchor_observations.tolist()
                return np.fromiter((hash(s) & ((1 << 63) - 1) for s in it), dtype=np.int64, count=len(it))
            it = anchor_observations.tolist()
            return np.fromiter((hash(x) & ((1 << 63) - 1) for x in it), dtype=np.int64, count=len(it))

        # If it's a numeric matrix (e.g., state embeddings), compress to int64 hash deterministically.
        if anchor_observations.dtype.kind in ("i", "u"):
            arr = anchor_observations.astype(np.int64, copy=False)
        elif anchor_observations.dtype.kind == "f":
            arr = anchor_observations.view(np.int64)
        else:
            flat = anchor_observations.reshape(anchor_observations.shape[0], -1).tolist()
            return np.fromiter((hash(tuple(row)) & ((1 << 63) - 1) for row in flat), dtype=np.int64, count=anchor_observations.shape[0])

        arr = np.ascontiguousarray(arr.reshape(arr.shape[0], -1))
        d = arr.shape[1]
        if self._anchor_prime_vec is None or self._anchor_prime_vec.shape[0] < d:
            rng = np.random.default_rng(12345)
            primes = rng.integers(low=1_000_000_007, high=2_000_000_033, size=max(d, 64), dtype=np.int64)
            primes[primes % 2 == 0] += 1
            self._anchor_prime_vec = primes
        primes = self._anchor_prime_vec[:d]
        h = (arr * primes).sum(axis=1)
        h = (h ^ (h >> 33)) * np.int64(0xff51afd7ed558ccd)
        h = (h ^ (h >> 33)) * np.int64(0xc4ceb9fe1a85ec53)
        h = (h ^ (h >> 33)) & np.int64((1 << 63) - 1)
        return h.astype(np.int64, copy=False)

    @staticmethod
    def _combine_keys64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a64 = a.astype(np.int64, copy=False)
        b64 = b.astype(np.int64, copy=False)
        x = (a64 << np.int64(32)) ^ b64
        x = (x ^ (x >> np.int64(33))) * np.int64(0xff51afd7ed558ccd)
        x = (x ^ (x >> np.int64(33))) * np.int64(0xc4ceb9fe1a85ec53)
        x = (x ^ (x >> np.int64(33))) & np.int64((1 << 63) - 1)
        return x

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

        mask = response_mask
        if mask.dtype != torch.float32 and mask.dtype != torch.float16 and mask.dtype != torch.bfloat16:
            mask_f = mask.to(dtype=dtype)
        else:
            mask_f = mask.to(dtype=dtype)

        # Base step reward from token-level reward for this response/action
        step_r = (token_level_rewards * mask_f).sum(dim=1)

        if step_rewards is not None:
            sr = step_rewards.to(device=device, dtype=dtype)
            if sr.dim() == 2:
                if sr.shape == token_level_rewards.shape:
                    step_r = step_r + (sr * mask_f).sum(dim=1)
                elif sr.shape[1] == 1:
                    step_r = step_r + sr.squeeze(1)
                else:
                    step_r = step_r + sr.sum(dim=1)
            elif sr.dim() == 1:
                step_r = step_r + sr
            else:
                step_r = step_r + sr.reshape(-1)

        bsz = step_r.shape[0]
        if bsz == 0:
            return token_level_rewards.new_zeros(token_level_rewards.shape), token_level_rewards.new_zeros(token_level_rewards.shape)

        # Aggregate per-trajectory return across samples (supports both "one sample per trajectory" and "many steps per trajectory").
        ep_np = np.asarray(episode_index).astype(np.int64, copy=False).reshape(-1)
        tr_np = np.asarray(trajectory_index).astype(np.int64, copy=False).reshape(-1)
        if ep_np.shape[0] != bsz or tr_np.shape[0] != bsz:
            ep_np = np.resize(ep_np, bsz).astype(np.int64, copy=False)
            tr_np = np.resize(tr_np, bsz).astype(np.int64, copy=False)

        traj_key_np = self._combine_keys64(ep_np, tr_np)
        traj_key = torch.from_numpy(traj_key_np).to(device=device)
        uniq_traj, inv_traj = torch.unique(traj_key, return_inverse=True)
        n_traj = uniq_traj.numel()

        traj_count = torch.bincount(inv_traj, minlength=n_traj).to(dtype=dtype)
        traj_sum = self._scatter_sum(step_r, inv_traj, n_traj)

        # episode id per trajectory (constant within trajectory); compute via average
        ep_t = torch.from_numpy(ep_np).to(device=device, dtype=dtype)
        ep_sum_per_traj = self._scatter_sum(ep_t, inv_traj, n_traj)
        ep_per_traj = (ep_sum_per_traj / traj_count.clamp_min(1)).round().to(dtype=torch.int64)

        # Compute GRPO/RLOO-style advantage on per-trajectory returns within each episode group
        uniq_ep_traj, inv_ep_traj = torch.unique(ep_per_traj, return_inverse=True)
        n_ep = uniq_ep_traj.numel()

        ep_traj_count = torch.bincount(inv_ep_traj, minlength=n_ep).to(dtype=dtype)
        ep_traj_sum = self._scatter_sum(traj_sum, inv_ep_traj, n_ep)
        ep_traj_sum2 = self._scatter_sum(traj_sum * traj_sum, inv_ep_traj, n_ep)

        ep_mean = ep_traj_sum / ep_traj_count.clamp_min(1)
        ep_var = (ep_traj_sum2 / ep_traj_count.clamp_min(1) - ep_mean * ep_mean).clamp_min(0)
        ep_std = torch.sqrt(ep_var + self.config.get("adv_eps", 1e-6))

        ep_sum_for = ep_traj_sum[inv_ep_traj]
        ep_cnt_for = ep_traj_count[inv_ep_traj]
        ep_mean_for = ep_mean[inv_ep_traj]
        ep_std_for = ep_std[inv_ep_traj]

        # Leave-one-out baseline per trajectory (over trajectories, not steps)
        loo_mean_traj = torch.where(
            ep_cnt_for > 1,
            (ep_sum_for - traj_sum) / (ep_cnt_for - 1).clamp_min(1),
            ep_mean_for,
        )
        traj_adv = (traj_sum - loo_mean_traj) / ep_std_for.clamp_min(self.config.get("adv_eps", 1e-6))
        traj_adv_per_sample = traj_adv[inv_traj]
        traj_return_per_sample = traj_sum[inv_traj]

        # Optional step-level (GiGPO-style) advantage using anchor observations (state grouping)
        step_adv_per_sample = torch.zeros_like(step_r)
        if anchor_observations is not None:
            anchor_ids_np = self._hash_anchor_obs(anchor_observations)
            if anchor_ids_np is not None:
                if anchor_ids_np.shape[0] != bsz:
                    anchor_ids_np = np.resize(anchor_ids_np, bsz).astype(np.int64, copy=False)
                step_key_np = self._combine_keys64(ep_np, anchor_ids_np)
                step_key = torch.from_numpy(step_key_np).to(device=device)
                uniq_step, inv_step = torch.unique(step_key, return_inverse=True)
                n_step = uniq_step.numel()
                step_cnt = torch.bincount(inv_step, minlength=n_step).to(dtype=dtype)
                step_sum = self._scatter_sum(step_r, inv_step, n_step)
                step_sum2 = self._scatter_sum(step_r * step_r, inv_step, n_step)
                step_mean = step_sum / step_cnt.clamp_min(1)
                step_var = (step_sum2 / step_cnt.clamp_min(1) - step_mean * step_mean).clamp_min(0)
                step_std = torch.sqrt(step_var + self.config.get("adv_eps", 1e-6))

                step_sum_for = step_sum[inv_step]
                step_cnt_for = step_cnt[inv_step]
                step_mean_for = step_mean[inv_step]
                step_std_for = step_std[inv_step]
                loo_mean_step = torch.where(
                    step_cnt_for > 1,
                    (step_sum_for - step_r) / (step_cnt_for - 1).clamp_min(1),
                    step_mean_for,
                )
                step_adv_per_sample = (step_r - loo_mean_step) / step_std_for.clamp_min(self.config.get("adv_eps", 1e-6))

        traj_coef = float(self.config.get("traj_adv_coef", 1.0))
        step_coef = float(self.config.get("step_adv_coef", 0.0)) if anchor_observations is not None else 0.0
        adv_scalar = traj_coef * traj_adv_per_sample + step_coef * step_adv_per_sample

        adv_clip = self.config.get("adv_clip", None)
        if adv_clip is not None:
            adv_scalar = adv_scalar.clamp(min=-float(adv_clip), max=float(adv_clip))

        advantages = adv_scalar[:, None] * mask_f
        returns = traj_return_per_sample[:, None] * mask_f
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
        mask = response_mask
        mask_f = mask.to(dtype=log_prob.dtype) if mask.dtype != log_prob.dtype else mask

        adv = advantages
        if bool(self.config.get("whiten_advantages", False)):
            adv = masked_whiten(adv, mask_f, shift_mean=True)

        log_ratio = log_prob - old_log_prob
        ratio = torch.exp(log_ratio)

        clip_ratio = float(clip_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

        pg_unclipped = ratio * adv
        pg_clipped = clipped_ratio * adv
        pg_obj = torch.minimum(pg_unclipped, pg_clipped)

        loss = -masked_mean(pg_obj, mask_f)

        approx_kl = masked_mean(old_log_prob - log_prob, mask_f).detach()
        clip_frac = masked_mean((torch.abs(ratio - 1.0) > clip_ratio).to(dtype=log_prob.dtype), mask_f).detach()

        metrics: Dict[str, Any] = {
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "adv_mean": masked_mean(adv, mask_f).detach(),
            "adv_abs_mean": masked_mean(torch.abs(adv), mask_f).detach(),
            "ratio_mean": masked_mean(ratio, mask_f).detach(),
        }

        if bool(self.config.get("use_kl_loss", False)):
            ref_log_prob = kwargs.get("ref_log_prob", None)
            if ref_log_prob is not None:
                ref_lp = ref_log_prob.to(device=log_prob.device, dtype=log_prob.dtype)
                # Forward KL estimate on sampled actions: E[log pi - log pref]
                kl_ref = masked_mean(log_prob - ref_lp, mask_f)
                kl_coef = float(self.config.get("kl_loss_coef", 0.0))
                loss = loss + kl_coef * kl_ref
                metrics["kl_ref"] = kl_ref.detach()
                metrics["kl_coef"] = torch.tensor(kl_coef, device=log_prob.device, dtype=log_prob.dtype)

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

        obs = step_observations if step_observations is not None else [""] * T
        act = step_actions if step_actions is not None else [""] * T
        if len(obs) != T:
            obs = (list(obs) + [""] * T)[:T]
        if len(act) != T:
            act = (list(act) + [""] * T)[:T]

        step_penalty = float(self.config.get("step_penalty", 0.0))
        invalid_penalty = float(self.config.get("invalid_action_penalty", 0.0))
        progress_bonus = float(self.config.get("progress_bonus", 0.0))
        repeat_penalty = float(self.config.get("repeat_action_penalty", 0.0))

        rewards = np.full((T,), -step_penalty, dtype=np.float32)

        progress_triggers = (
            "you open",
            "you close",
            "you take",
            "you pick up",
            "you grab",
            "you put",
            "you drop",
            "you place",
            "you insert",
            "you clean",
            "you wash",
            "you rinse",
            "you heat",
            "you warm",
            "you cool",
            "you chill",
            "you turn on",
            "you switch on",
            "you turn off",
            "you switch off",
            "you slice",
            "you cut",
            "you cook",
            "you fry",
            "you bake",
            "you microwave",
            "you unlock",
            "you lock",
        )
        invalid_triggers = (
            "you can't",
            "you cannot",
            "you don't",
            "i don't understand",
            "nothing happens",
            "not possible",
            "is closed",
            "you need to",
            "you are not holding",
            "you aren't holding",
            "you can't see any",
            "there is no",
            "you see no",
            "can't do that",
            "cannot do that",
            "you fail",
        )

        prev_action = None
        for i in range(T):
            o = (obs[i] or "").strip().lower()
            a = (act[i] or "").strip().lower()

            if prev_action is not None and a == prev_action and a != "":
                rewards[i] -= repeat_penalty
            prev_action = a

            if a.startswith("look") or a.startswith("inventory"):
                rewards[i] -= 0.001

            if any(t in o for t in invalid_triggers):
                rewards[i] -= invalid_penalty
            if any(t in o for t in progress_triggers):
                rewards[i] += progress_bonus

        # Distribute terminal episode reward across steps with geometric weighting (normalized to total).
        ep_r = float(episode_reward)
        if ep_r != 0.0:
            total_success = float(self.config.get("success_reward_total", 1.0)) * ep_r
            disc = float(self.config.get("success_reward_discount", 0.97))
            w = np.power(disc, np.arange(T - 1, -1, -1, dtype=np.float32))
            w_sum = float(w.sum()) if float(w.sum()) > 0 else 1.0
            rewards += (total_success * (w / w_sum)).astype(np.float32)

        return rewards.astype(np.float32, copy=False)