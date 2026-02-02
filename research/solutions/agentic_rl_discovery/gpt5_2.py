import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List

try:
    from verl.utils.torch_functional import masked_mean as _verl_masked_mean  # type: ignore
except Exception:
    _verl_masked_mean = None


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, eps: float = 1e-8) -> torch.Tensor:
    if mask.dtype != x.dtype:
        mask = mask.to(dtype=x.dtype)
    if dim is None:
        denom = mask.sum().clamp_min(eps)
        return (x * mask).sum() / denom
    denom = mask.sum(dim=dim).clamp_min(eps)
    return (x * mask).sum(dim=dim) / denom


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, eps: float = 1e-8) -> torch.Tensor:
    if _verl_masked_mean is not None:
        try:
            return _verl_masked_mean(x, mask, dim=dim, eps=eps)
        except Exception:
            pass
    return _masked_mean(x, mask, dim=dim, eps=eps)


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        self.config = {
            "gamma": 0.95,
            "clip_ratio": 0.2,
            "use_kl_loss": False,
            "kl_loss_coef": 0.01,
            "seq_advantage_coef": 1.0,
            "step_advantage_coef": 0.6,
            "normalize_adv_within_group": True,
            "whiten_eps": 1e-8,
            "length_normalization": "per_token",  # ["per_token", "none", "sqrt"]
            "use_discount_in_token_adv": False,
            "step_success_gamma": 0.98,
            "max_step_shaping": 0.1,  # cap absolute sum of shaping to avoid overpowering sparse reward
        }
        return self

    @staticmethod
    def _discounted_cumsum_tokenwise(
        rewards: torch.Tensor, mask: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        # rewards, mask: (B, L)
        B, L = rewards.shape
        dtype = rewards.dtype
        device = rewards.device
        out = torch.zeros_like(rewards, dtype=dtype, device=device)
        running = torch.zeros(B, dtype=dtype, device=device)
        # We propagate only over masked region; outside mask stays zero
        for t in range(L - 1, -1, -1):
            running = rewards[:, t] + gamma * running
            out[:, t] = running
            # Optionally zero-out positions outside the response region
            running = running * mask[:, t] + running.detach() * (1.0 - mask[:, t])
        out = out * mask
        return out

    @staticmethod
    def _unique_inverse(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns (unique_values, inverse_index)
        return torch.unique(t, return_inverse=True)

    @staticmethod
    def _loo_baseline(values: torch.Tensor, group_index: torch.Tensor) -> torch.Tensor:
        # values: (B,), group_index: (B,) contiguous 0..G-1
        device = values.device
        G = int(group_index.max().item()) + 1 if values.numel() > 0 else 0
        if G == 0:
            return torch.zeros_like(values)
        counts = torch.bincount(group_index, minlength=G).to(values.dtype)
        sums = torch.zeros(G, dtype=values.dtype, device=device)
        sums = sums.index_add(0, group_index, values)
        counts_i = counts[group_index]
        denom = (counts_i - 1.0).clamp_min(1.0)
        baseline = torch.where(
            counts_i > 1.0, (sums[group_index] - values) / denom, torch.zeros_like(values)
        )
        return baseline

    @staticmethod
    def _group_whiten(x: torch.Tensor, group_index: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # x: (B,), group_index: (B,)
        out = x.clone()
        if x.numel() == 0:
            return out
        unique_groups = torch.unique(group_index)
        for g in unique_groups:
            mask = (group_index == g)
            if mask.sum() <= 1:
                continue
            xg = out[mask]
            mean = xg.mean()
            var = xg.var(unbiased=False)
            std = torch.sqrt(var + eps)
            out[mask] = (xg - mean) / std
        return out

    @staticmethod
    def _to_list_per_sample(
        step_rewards: Optional[torch.Tensor],
        anchor_observations: Optional[np.ndarray],
        batch_size: int
    ) -> Tuple[Optional[List[List[float]]], Optional[List[List[str]]], Optional[List[int]]]:
        if step_rewards is None or anchor_observations is None:
            return None, None, None

        # Convert anchor_observations to python list of lists
        anchors_list: List[List[str]] = []
        if isinstance(anchor_observations, np.ndarray):
            # anchor_observations likely is an object array of lists
            for i in range(len(anchor_observations)):
                ai = anchor_observations[i]
                if isinstance(ai, (list, tuple)):
                    anchors_list.append([str(x) for x in ai])
                else:
                    try:
                        anchors_list.append([str(y) for y in list(ai)])
                    except Exception:
                        anchors_list.append([])
        elif isinstance(anchor_observations, list):
            for ai in anchor_observations:
                if isinstance(ai, (list, tuple)):
                    anchors_list.append([str(x) for x in ai])
                else:
                    anchors_list.append([])
        else:
            anchors_list = [[] for _ in range(batch_size)]

        # Convert step_rewards to list of lists
        rewards_list: List[List[float]] = []
        if isinstance(step_rewards, torch.Tensor):
            sr = step_rewards.detach().cpu().numpy()
            # Determine each length using anchors length if available
            for i in range(batch_size):
                if i < len(anchors_list):
                    T = len(anchors_list[i])
                else:
                    T = sr.shape[1] if sr.ndim == 2 else 0
                if sr.ndim == 1:
                    # If shape (B,), interpret as single step or empty
                    vals = [float(sr[i])] if i < sr.shape[0] else []
                elif sr.ndim == 2:
                    T = min(T, sr.shape[1]) if T > 0 else sr.shape[1]
                    vals = [float(x) for x in sr[i, :T]]
                else:
                    vals = []
                rewards_list.append(vals)
        elif isinstance(step_rewards, (list, tuple, np.ndarray)):
            try:
                for i in range(batch_size):
                    ri = step_rewards[i]
                    if isinstance(ri, (list, tuple, np.ndarray)):
                        rewards_list.append([float(x) for x in ri])
                    else:
                        rewards_list.append([float(ri)])
            except Exception:
                rewards_list = [[] for _ in range(batch_size)]
        else:
            rewards_list = [[] for _ in range(batch_size)]

        # Align sizes
        if len(anchors_list) != batch_size:
            anchors_list = (anchors_list + [[] for _ in range(batch_size - len(anchors_list))])[:batch_size]
        if len(rewards_list) != batch_size:
            rewards_list = (rewards_list + [[] for _ in range(batch_size - len(rewards_list))])[:batch_size]

        step_lens: List[int] = []
        for i in range(batch_size):
            T = min(len(anchors_list[i]), len(rewards_list[i]))
            anchors_list[i] = anchors_list[i][:T]
            rewards_list[i] = rewards_list[i][:T]
            step_lens.append(T)

        return rewards_list, anchors_list, step_lens

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
        B, L = token_level_rewards.shape

        mask = response_mask.to(device=device, dtype=dtype)
        rewards = token_level_rewards.to(device=device, dtype=dtype)

        # Token-level returns (discount across response tokens)
        token_returns = self._discounted_cumsum_tokenwise(rewards, mask, gamma)

        # Sequence-level scalar reward
        seq_rewards = (rewards * mask).sum(dim=1)  # (B,)

        # Group IDs as contiguous integers
        epi_ids = torch.as_tensor(episode_index, device=device, dtype=torch.long)
        _, inv = self._unique_inverse(epi_ids)  # (B,)

        # Leave-one-out baseline within each episode group
        seq_baseline = self._loo_baseline(seq_rewards, inv)  # (B,)
        seq_adv = seq_rewards - seq_baseline  # (B,)

        # Optional step-level advantage via anchor grouping (GiGPO-style)
        step_adv = None
        step_lens = None
        if step_rewards is not None and anchor_observations is not None:
            rewards_list, anchors_list, step_lens = self._to_list_per_sample(step_rewards, anchor_observations, B)
            if rewards_list is not None and anchors_list is not None:
                step_adv_vals = torch.zeros(B, dtype=dtype, device=device)
                step_counts = torch.zeros(B, dtype=torch.long, device=device)
                # Build groups per episode index
                unique_groups = torch.unique(inv).tolist()
                for g in unique_groups:
                    idxs = torch.nonzero(inv == g, as_tuple=False).view(-1).tolist()
                    # Map anchor -> list of (idx, step_reward)
                    anchor_map: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
                    for idx in idxs:
                        ai = anchors_list[idx] if idx < len(anchors_list) else []
                        ri = rewards_list[idx] if idx < len(rewards_list) else []
                        Ti = min(len(ai), len(ri))
                        if Ti <= 0:
                            continue
                        step_counts[idx] = Ti
                        for t in range(Ti):
                            key = ai[t]
                            r = float(ri[t])
                            anchor_map[key].append((idx, r))
                    # Compute LOO per anchor
                    for key, entries in anchor_map.items():
                        n = len(entries)
                        if n == 0:
                            continue
                        total = sum(r for (_, r) in entries)
                        if n == 1:
                            i0, r0 = entries[0]
                            step_adv_vals[i0] += r0  # baseline 0 for singleton
                        else:
                            for i0, r0 in entries:
                                baseline = (total - r0) / (n - 1)
                                step_adv_vals[i0] += (r0 - baseline)
                # Normalize by number of steps to keep scale stable
                denom = step_counts.clamp_min(1).to(dtype=dtype)
                step_adv = step_adv_vals / denom

        # Combine sequence-level and step-level advantages
        adv_scalar = seq_adv.clone()
        seq_coef = float(self.config.get("seq_advantage_coef", 1.0))
        step_coef = float(self.config.get("step_advantage_coef", 0.6))
        adv_scalar = seq_coef * seq_adv
        if step_adv is not None:
            adv_scalar = adv_scalar + step_coef * step_adv

        # Optional group-wise whitening
        if bool(self.config.get("normalize_adv_within_group", True)):
            adv_scalar = self._group_whiten(adv_scalar, inv, eps=float(self.config.get("whiten_eps", 1e-8)))

        # Broadcast advantages to tokens
        resp_len = mask.sum(dim=1).clamp_min(1.0)  # (B,)
        length_norm = self.config.get("length_normalization", "per_token")
        if length_norm == "per_token":
            scale = 1.0 / resp_len
        elif length_norm == "sqrt":
            scale = 1.0 / torch.sqrt(resp_len)
        else:
            scale = torch.ones_like(resp_len)
        adv_per_token = (adv_scalar * scale).unsqueeze(1).expand(B, L) * mask

        return adv_per_token.to(dtype=dtype), token_returns.to(dtype=dtype)

    def compute_policy_loss(
        self,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        clip_ratio: float = 0.2,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        device = log_prob.device
        dtype = log_prob.dtype

        mask = response_mask.to(device=device, dtype=dtype)
        adv = advantages.to(device=device, dtype=dtype)

        # Normalize advantages per batch to stabilize (mask-aware)
        adv_mean = masked_mean(adv, mask)
        adv_var = masked_mean((adv - adv_mean) ** 2, mask)
        adv_std = torch.sqrt(adv_var + 1e-8)
        adv = (adv - adv_mean) / adv_std.clamp_min(1e-8)

        ratio = torch.exp(log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

        # PPO objective per token
        obj1 = ratio * adv
        obj2 = clipped_ratio * adv
        obj = torch.minimum(obj1, obj2)

        loss = -masked_mean(obj, mask)

        # Approximate KL
        approx_kl = masked_mean(old_log_prob - log_prob, mask)
        if bool(self.config.get("use_kl_loss", False)):
            loss = loss + float(self.config.get("kl_loss_coef", 0.01)) * approx_kl

        # Clip fraction
        clip_frac = masked_mean((torch.abs(ratio - 1.0) > clip_ratio).to(dtype), mask)

        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "approx_kl": float(approx_kl.detach().cpu().item()),
            "clip_frac": float(clip_frac.detach().cpu().item()),
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

        # Base intrinsic shaping from action/observation heuristics
        obs_list = [str(o).lower() for o in (step_observations or [])]
        if len(obs_list) < T:
            obs_list += [""] * (T - len(obs_list))
        act_list = [str(a).lower() for a in (step_actions or [])]
        if len(act_list) < T:
            act_list += [""] * (T - len(act_list))

        neg_patterns = [
            "can't", "cannot", "nothing happens", "not possible", "invalid",
            "unknown command", "there is no", "you do not see", "you don't see",
            "you can't go", "stuck", "blocked", "failed"
        ]
        pos_action_patterns = [
            "open", "close", "take", "put", "go to", "examine", "toggle",
            "clean", "wash", "heat", "cool", "turn on", "turn off", "drop",
            "inventory", "look", "use"
        ]
        pos_observation_patterns = [
            "you open", "you close", "you take", "you put", "you pick", "you drop",
            "you turn on", "you turn off", "you clean", "you wash", "you unlock",
            "you lock", "you move", "you go", "you insert", "you place"
        ]

        shaping = np.zeros(T, dtype=np.float32)

        # Heuristic: small penalty for invalid actions, small reward for plausible actions
        for t in range(T):
            o = obs_list[t]
            a = act_list[t]
            # Invalid or no-op indicators
            if any(s in o for s in neg_patterns):
                shaping[t] += -0.02
            else:
                shaping[t] += 0.004

            # Action-based small positive signal
            if any(s in a for s in pos_action_patterns):
                shaping[t] += 0.006

            # Observation-based signal indicating state change
            if any(s in o for s in pos_observation_patterns):
                shaping[t] += 0.008

            # Small penalty for repeating the exact same action consecutively
            if t > 0 and a == act_list[t - 1]:
                shaping[t] += -0.004

        # Cap total magnitude of shaping to avoid overpowering sparse terminal reward
        max_shaping = float(self.config.get("max_step_shaping", 0.1))
        total_abs = float(np.sum(np.abs(shaping)))
        if total_abs > max_shaping and total_abs > 0.0:
            shaping *= (max_shaping / total_abs)

        # Success-based dense distribution with backward discounting
        dense = np.zeros(T, dtype=np.float32)
        if episode_reward and episode_reward > 0.0:
            sgamma = float(self.config.get("step_success_gamma", 0.98))
            weights = np.power(sgamma, np.arange(T - 1, -1, -1, dtype=np.float32))
            weights_sum = float(np.sum(weights)) + 1e-8
            dense = (weights / weights_sum) * float(episode_reward)

        step_rewards = dense + shaping
        return step_rewards.astype(np.float32)