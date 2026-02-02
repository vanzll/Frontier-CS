import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List

try:
    from verl.utils.torch_functional import masked_mean as verl_masked_mean
except Exception:
    verl_masked_mean = None


def _to_float_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return mask.float()
    if mask.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return mask.float()
    return mask


def _masked_sum(x: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False, eps: float = 1e-8) -> torch.Tensor:
    mask = _to_float_mask(mask)
    if dim is None:
        return (x * mask).sum()  # scalar
    return (x * mask).sum(dim=dim, keepdim=keepdim)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False, eps: float = 1e-8) -> torch.Tensor:
    mask = _to_float_mask(mask)
    if verl_masked_mean is not None:
        return verl_masked_mean(x, mask, dim=dim, keepdim=keepdim)
    denom = mask.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return _masked_sum(x, mask, dim=dim, keepdim=keepdim) / denom


def _discounted_returns(rewards: torch.Tensor, mask: torch.Tensor, gamma: float) -> torch.Tensor:
    # rewards: (B, T), mask: (B, T)
    B, T = rewards.shape
    mask = _to_float_mask(mask)
    returns = torch.zeros_like(rewards)
    running = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        valid = mask[:, t] > 0.0
        running = torch.where(valid, rewards[:, t] + gamma * running, running)
        returns[:, t] = torch.where(valid, running, torch.zeros_like(running))
    return returns


def _unique_numpy_to_torch_ids(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    # Map arbitrary numpy array to contiguous integer ids
    if arr is None:
        return None
    try:
        flat = arr
        if not isinstance(flat, np.ndarray):
            flat = np.asarray(flat)
        if flat.dtype.type is np.str_ or flat.dtype.type is np.object_:
            flat = flat.astype(str)
        unique, inverse = np.unique(flat, return_inverse=True)
        ids = torch.from_numpy(inverse.astype(np.int64, copy=False)).to(device)
        return ids
    except Exception:
        # Fallback: convert via Python str and dict
        flat = [str(x) for x in arr]
        id_map: Dict[str, int] = {}
        ids_py: List[int] = []
        for s in flat:
            if s not in id_map:
                id_map[s] = len(id_map)
            ids_py.append(id_map[s])
        return torch.tensor(ids_py, dtype=torch.long, device=device)


def _group_sums_counts(values: torch.Tensor, group_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # values: (B,), group_ids: (B,)
    unique_ids, inverse = torch.unique_consecutive(group_ids.sort().values, return_inverse=False), None  # placeholder
    # The above line sorts unique ids but also loses mapping; we will implement differently for speed and simplicity
    # Use torch.unique with return_inverse
    uniq, inv = torch.unique(group_ids, return_inverse=True)
    sums = torch.zeros_like(uniq, dtype=values.dtype)
    counts = torch.zeros_like(uniq, dtype=values.dtype)
    ones = torch.ones_like(values, dtype=values.dtype)
    sums.scatter_add_(0, inv, values)
    counts.scatter_add_(0, inv, ones)
    return sums, counts  # aligned with uniq; inv maps back


def _group_mean_per_sample(values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    uniq, inv = torch.unique(group_ids, return_inverse=True)
    sums = torch.zeros_like(uniq, dtype=values.dtype)
    counts = torch.zeros_like(uniq, dtype=values.dtype)
    ones = torch.ones_like(values, dtype=values.dtype)
    sums.scatter_add_(0, inv, values)
    counts.scatter_add_(0, inv, ones)
    means = sums / counts.clamp_min(1.0)
    return means[inv]


def _group_loo_baseline(values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    # Leave-one-out baseline: (sum_group - v_i) / (count_group - 1)
    uniq, inv = torch.unique(group_ids, return_inverse=True)
    sums = torch.zeros_like(uniq, dtype=values.dtype)
    counts = torch.zeros_like(uniq, dtype=values.dtype)
    ones = torch.ones_like(values, dtype=values.dtype)
    sums.scatter_add_(0, inv, values)
    counts.scatter_add_(0, inv, ones)
    sum_i = sums[inv]
    count_i = counts[inv]
    denom = (count_i - 1.0).clamp_min(1.0)
    baseline = (sum_i - values) / denom
    return baseline


def _whiten_by_group(values: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # values: (B,), group_ids: (B,)
    uniq, inv = torch.unique(group_ids, return_inverse=True)
    sums = torch.zeros_like(uniq, dtype=values.dtype)
    sums_sq = torch.zeros_like(uniq, dtype=values.dtype)
    counts = torch.zeros_like(uniq, dtype=values.dtype)
    ones = torch.ones_like(values, dtype=values.dtype)
    sums.scatter_add_(0, inv, values)
    sums_sq.scatter_add_(0, inv, values * values)
    counts.scatter_add_(0, inv, ones)
    means = sums / counts.clamp_min(1.0)
    vars_ = (sums_sq / counts.clamp_min(1.0)) - means * means
    stds = torch.sqrt(vars_.clamp_min(eps))
    whitened = (values - means[inv]) / stds[inv]
    return whitened


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        self.config = {
            "gamma": 0.95,
            "clip_ratio": 0.2,
            "use_kl_loss": True,
            "kl_loss_coef": 0.02,
            "anchor_weight": 0.5,
            "adv_whiten": True,
            "length_normalize_adv": True,
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
        device = token_level_rewards.device
        dtype = token_level_rewards.dtype
        B, T = token_level_rewards.shape

        mask = _to_float_mask(response_mask).to(device=device, dtype=dtype)

        # Sequence lengths from mask
        lengths = mask.sum(dim=-1)  # (B,)

        # Step rewards per sample
        if step_rewards is None:
            step_rewards_t = torch.zeros(B, device=device, dtype=dtype)
        else:
            step_rewards_t = step_rewards.to(device=device, dtype=dtype).view(-1)

        # Combine rewards for "returns" computation: add step reward to last valid token
        r = token_level_rewards.to(device=device, dtype=dtype).clone()
        last_idx = (lengths.long() - 1).clamp_min(0)
        has_tokens = (lengths > 0.0)
        if has_tokens.any():
            idx_b = torch.arange(B, device=device)
            r[idx_b[has_tokens], last_idx[has_tokens]] = r[idx_b[has_tokens], last_idx[has_tokens]] + step_rewards_t[has_tokens]

        # Discounted returns over tokens (masked)
        returns = _discounted_returns(r, mask, gamma=self.config.get("gamma", gamma))

        # Sequence-level reward for GRPO-style advantage: sum of token rewards + step reward
        seq_token_reward = _masked_sum(token_level_rewards, mask, dim=-1)  # (B,)
        seq_reward = seq_token_reward + step_rewards_t  # (B,)

        # Groupings
        episode_ids = torch.as_tensor(episode_index, dtype=torch.long, device=device).view(-1)
        # Anchor grouping ids (optional)
        anchor_ids = _unique_numpy_to_torch_ids(anchor_observations, device=device) if anchor_observations is not None else None

        # Episode LOO baseline
        baseline_episode = _group_loo_baseline(seq_reward, episode_ids)

        # Anchor LOO baseline
        if anchor_ids is not None:
            baseline_anchor = _group_loo_baseline(seq_reward, anchor_ids)
            # If anchor group size == 1, fall back to episode baseline
            uniq_a, inv_a = torch.unique(anchor_ids, return_inverse=True)
            counts_a = torch.zeros_like(uniq_a, dtype=seq_reward.dtype)
            ones = torch.ones_like(seq_reward, dtype=seq_reward.dtype)
            counts_a.scatter_add_(0, inv_a, ones)
            count_anchor = counts_a[inv_a]
            baseline_anchor = torch.where(count_anchor > 1.0, baseline_anchor, baseline_episode)
            alpha = float(self.config.get("anchor_weight", 0.5))
            baseline = (1.0 - alpha) * baseline_episode + alpha * baseline_anchor
        else:
            baseline = baseline_episode

        adv_seq = seq_reward - baseline  # (B,)

        # Optionally normalize by response length to reduce bias for long generations
        if self.config.get("length_normalize_adv", True):
            adv_seq = adv_seq / lengths.clamp_min(1.0).sqrt()

        # Optional whitening within episode groups
        if self.config.get("adv_whiten", True):
            adv_seq = _whiten_by_group(adv_seq, episode_ids).clamp_(-10.0, 10.0)

        # Broadcast to token level
        advantages = adv_seq.unsqueeze(1).expand(B, T) * mask

        # Optionally normalize advantages globally across tokens for numerical stability
        adv_mean = _masked_mean(advantages, mask)
        adv_std = torch.sqrt((_masked_mean((advantages - adv_mean) ** 2, mask)).clamp_min(1e-8))
        advantages = (advantages - adv_mean) / adv_std.clamp_min(1e-6)

        # Returns are already token-level discounted values; keep only masked positions
        returns = returns * mask

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
        mask = _to_float_mask(response_mask)
        # Ensure shapes align
        assert old_log_prob.shape == log_prob.shape == advantages.shape == mask.shape

        # Normalize advantages per-batch for stability (masked)
        adv_mean = _masked_mean(advantages, mask)
        adv_std = torch.sqrt((_masked_mean((advantages - adv_mean) ** 2, mask)).clamp_min(1e-8))
        norm_adv = (advantages - adv_mean) / adv_std.clamp_min(1e-6)
        norm_adv = norm_adv * mask

        # PPO clip objective
        ratio = torch.exp(log_prob - old_log_prob) * mask + (1.0 - mask)  # leave non-masked as ratio=1
        surr1 = ratio * norm_adv
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * norm_adv
        policy_loss_tokens = -torch.min(surr1, surr2) * mask

        loss = policy_loss_tokens.sum() / mask.sum().clamp_min(1.0)

        # Optional KL penalty (estimated under behavior policy)
        if self.config.get("use_kl_loss", False):
            approx_kl = (old_log_prob - log_prob) * mask  # E_old[log pi_old - log pi_new]
            kl_mean = approx_kl.sum() / mask.sum().clamp_min(1.0)
            loss = loss + self.config.get("kl_loss_coef", 0.01) * kl_mean
        else:
            kl_mean = ((old_log_prob - log_prob) * mask).sum() / mask.sum().clamp_min(1.0)

        # Metrics
        with torch.no_grad():
            clipped = (torch.abs(ratio - 1.0) > clip_ratio).float() * mask
            clip_frac = clipped.sum() / mask.sum().clamp_min(1.0)
            metrics = {
                "loss/policy": loss.detach(),
                "stats/clip_frac": clip_frac.detach(),
                "stats/approx_kl": kl_mean.detach(),
                "stats/ratio_mean": (_masked_mean(ratio, mask)).detach(),
                "stats/adv_mean": adv_mean.detach(),
                "stats/adv_std": adv_std.detach(),
                "stats/token_count": mask.sum().detach(),
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
        T = int(trajectory_length)
        if T <= 0:
            return np.zeros((0,), dtype=np.float32)

        # Base distribution: if success, distribute via backward discount; if fail, base zeros
        success = float(episode_reward) > 0.0
        gamma_step = 0.90
        if success:
            weights = np.array([gamma_step ** (T - 1 - t) for t in range(T)], dtype=np.float64)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            base = weights * float(episode_reward)
        else:
            base = np.zeros(T, dtype=np.float64)

        # Dense shaping signals from actions and observations (small magnitudes)
        small_pos = 0.004
        small_neg = -0.006
        tiny_pos = 0.0015
        tiny_neg = -0.001

        action_bonus_verbs = [
            "open", "close", "take", "put", "drop", "insert", "place", "leave",
            "switch on", "switch off", "turn on", "turn off", "cook", "heat",
            "cool", "clean", "wash", "rinse", "dry", "slice", "chop", "mix",
            "fill", "empty", "go to", "walk to", "move to"
        ]
        observation_bad_phrases = [
            "nothing happens", "can't", "cannot", "not possible", "invalid",
            "no such", "not here", "don't see", "already", "you are already",
            "failed", "error"
        ]

        extras = np.zeros(T, dtype=np.float64)
        seen_actions = set()

        for t in range(T):
            act = str(step_actions[t]).lower() if t < len(step_actions) else ""
            obs = str(step_observations[t]).lower() if t < len(step_observations) else ""

            # Reward state-changing or goal-directed verbs
            for verb in action_bonus_verbs:
                if verb in act:
                    extras[t] += tiny_pos

            # Penalize obvious failures in observation
            for bad in observation_bad_phrases:
                if bad in obs:
                    extras[t] += small_neg
                    break

            # Penalize exact repeats to reduce loops
            if t > 0 and t < len(step_actions):
                if str(step_actions[t]).strip().lower() == str(step_actions[t - 1]).strip().lower():
                    extras[t] += small_neg

            # Encourage novelty very slightly
            normalized_action = " ".join(str(step_actions[t]).strip().lower().split())
            if normalized_action not in seen_actions:
                extras[t] += tiny_pos
                seen_actions.add(normalized_action)
            else:
                extras[t] += tiny_neg

        # Combine and normalize to keep total reward equal to episode_reward (potential-based adjustment)
        shaped = base + extras

        # Adjust by subtracting mean offset so sum equals episode_reward
        total = shaped.sum()
        target_sum = float(episode_reward)
        offset = (total - target_sum) / T
        shaped = shaped - offset

        # Clip for stability
        shaped = np.clip(shaped, -0.2, 1.0)

        # Final correction to ensure exact sum (within numerical tolerance)
        diff = shaped.sum() - target_sum
        if abs(diff) > 1e-6:
            shaped[0] -= diff

        return shaped.astype(np.float32)