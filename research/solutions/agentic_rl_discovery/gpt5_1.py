import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List

try:
    from verl.utils.torch_functional import masked_mean, masked_whiten, entropy_from_logits
except Exception:
    masked_mean = None
    masked_whiten = None
    entropy_from_logits = None


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        self.config = {
            "gamma": 0.95,
            "clip_ratio": 0.2,
            "use_kl_loss": False,
            "kl_loss_coef": 0.01,
            "normalize_adv_by_group": True,
            "adv_eps": 1e-6,
        }
        return self

    @staticmethod
    def _discounted_cumsum(rewards: torch.Tensor, mask: torch.Tensor, gamma: float) -> torch.Tensor:
        # rewards, mask: (B, T)
        B, T = rewards.shape
        device = rewards.device
        ret = torch.zeros_like(rewards, device=device, dtype=rewards.dtype)
        running = torch.zeros(B, device=device, dtype=rewards.dtype)
        for t in range(T - 1, -1, -1):
            r_t = rewards[:, t] * mask[:, t]
            running = r_t + gamma * running
            ret[:, t] = running
            # Mask out invalid positions to strictly keep return only on response tokens
            ret[:, t] = ret[:, t] * mask[:, t]
        return ret

    @staticmethod
    def _rloo_group_advantages(
        seq_rewards: torch.Tensor,  # (B,)
        episode_index: np.ndarray,
        normalize_group: bool = True,
        eps: float = 1e-6,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        # Compute leave-one-out baselines within each episode group
        B = seq_rewards.shape[0]
        adv = torch.zeros_like(seq_rewards)
        # Build mapping: group_id -> list of indices
        group_to_indices: Dict[int, List[int]] = defaultdict(list)
        for i, gid in enumerate(episode_index.tolist()):
            group_to_indices[gid].append(i)
        for gid, idxs in group_to_indices.items():
            idxs_t = torch.tensor(idxs, device=seq_rewards.device, dtype=torch.long)
            group_vals = seq_rewards.index_select(0, idxs_t)
            n = group_vals.shape[0]
            if n > 1:
                total = group_vals.sum()
                loo_baseline = (total - group_vals) / (n - 1)
                group_adv = group_vals - loo_baseline
            else:
                # no baseline available, use zero baseline
                group_adv = group_vals
            if normalize_group and n > 1:
                mean = group_adv.mean()
                std = group_adv.std(unbiased=False)
                group_adv = (group_adv - mean) / (std + eps)
            adv.index_copy_(0, idxs_t, group_adv)
        return adv

    def compute_advantage(
        self,
        token_level_rewards: torch.Tensor,   # (batch, seq_len)
        response_mask: torch.Tensor,         # (batch, seq_len)
        episode_index: np.ndarray,           # (batch,)
        trajectory_index: np.ndarray,        # (batch,)
        step_rewards: Optional[torch.Tensor] = None,
        anchor_observations: Optional[np.ndarray] = None,
        gamma: float = 0.95,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure dtype and device consistency
        device = token_level_rewards.device
        gamma = float(gamma)

        # 1) Compute per-token returns via discounted cumsum
        returns = self._discounted_cumsum(token_level_rewards, response_mask, gamma)

        # 2) Compute per-trajectory scalar reward (sequence-level), typically sum over tokens on valid positions
        seq_rewards = (token_level_rewards * response_mask).sum(dim=-1)

        # 3) Group-wise RLOO baseline and normalize (GRPO-style)
        normalize_group = self.config.get("normalize_adv_by_group", True)
        adv_scalar = self._rloo_group_advantages(
            seq_rewards=seq_rewards,
            episode_index=episode_index,
            normalize_group=normalize_group,
            eps=self.config.get("adv_eps", 1e-6),
            device=device,
        )  # (B,)

        # 4) Broadcast scalar advantages to all valid tokens (constant across tokens)
        advantages = adv_scalar.unsqueeze(-1).expand_as(response_mask) * response_mask

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
        # Flatten masked tokens for metrics convenience
        eps = 1e-6
        clip_ratio = float(clip_ratio)

        # Optional: re-normalize advantages over masked tokens for numerical stability
        # Keep scale consistent
        mask = response_mask
        if masked_whiten is not None:
            norm_adv = masked_whiten(advantages, mask, eps=1e-6)
        else:
            # Fallback whitening if verl util not available
            denom = mask.sum().clamp(min=1.0)
            mean_adv = (advantages * mask).sum() / denom
            var_adv = (((advantages - mean_adv) * mask) ** 2).sum() / denom
            std_adv = torch.sqrt(var_adv + 1e-6)
            norm_adv = (advantages - mean_adv) / (std_adv + 1e-6)
            norm_adv = norm_adv * mask

        # PPO objective
        ratio = torch.exp(torch.clamp(log_prob - old_log_prob, min=-60.0, max=60.0))
        unclipped = ratio * norm_adv
        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * norm_adv
        per_token_obj = torch.min(unclipped, clipped)

        # Masked mean loss (maximize objective -> minimize negative)
        if masked_mean is not None:
            obj = masked_mean(per_token_obj, mask)
        else:
            denom = mask.sum().clamp(min=1.0)
            obj = (per_token_obj * mask).sum() / denom

        policy_loss = -obj

        # Optional KL penalty
        loss = policy_loss
        metrics: Dict[str, Any] = {}

        approx_kl = (old_log_prob - log_prob) * mask
        if masked_mean is not None:
            approx_kl_mean = masked_mean(approx_kl, mask)
        else:
            denom = mask.sum().clamp(min=1.0)
            approx_kl_mean = approx_kl.sum() / denom
        metrics["approx_kl"] = approx_kl_mean.item()

        if self.config.get("use_kl_loss", False):
            kl_coef = float(self.config.get("kl_loss_coef", 0.01))
            kl_loss = kl_coef * approx_kl_mean
            loss = loss + kl_loss
            metrics["kl_coef"] = kl_coef
            metrics["kl_loss"] = kl_loss.item()

        # Clip fraction metric
        clipped_mask = (torch.abs(ratio - 1.0) > clip_ratio).float() * mask
        if masked_mean is not None:
            clip_frac = masked_mean(clipped_mask, mask)
        else:
            denom = mask.sum().clamp(min=1.0)
            clip_frac = clipped_mask.sum() / denom
        metrics["clip_frac"] = clip_frac.item()
        metrics["policy_loss"] = policy_loss.item()
        metrics["loss"] = loss.item()

        return loss, metrics

    def assign_step_rewards(
        self,
        episode_reward: float,
        trajectory_length: int,
        step_observations: list,
        step_actions: list,
        **kwargs
    ) -> np.ndarray:
        # Heuristic step credit assignment:
        # - Allocate positive weights to steps with useful verbs or state changes inferred from action/observation
        # - Downweight failed or redundant actions
        # - Normalize weights to sum to 1 and multiply by episode_reward
        # - If no positive weights found, give all credit to the final step on success, else zeros

        verbs_primary = [
            "open", "close", "take", "pick up", "pick", "drop", "put", "insert",
            "place", "wash", "clean", "rinse", "heat", "cool", "freeze",
            "dry", "toggle", "turn on", "turn off", "switch on", "switch off",
            "slice", "cut", "chop", "cook", "boil", "microwave",
        ]
        success_markers = ["success", "task complete", "task completed", "you win", "congrat", "done"]
        failure_markers = ["you can't", "cannot", "can't", "nothing happens", "invalid", "fail", "failed", "doesn't work", "not possible", "no such", "unknown action", "sorry"]

        obs_l = [str(o).lower() for o in step_observations[:trajectory_length]]
        act_l = [str(a).lower() for a in step_actions[:trajectory_length]]

        weights = np.zeros(trajectory_length, dtype=np.float32)

        # Count duplicates to reduce repeated actions weight
        action_counts: Dict[str, int] = defaultdict(int)
        for a in act_l:
            action_counts[a] += 1

        for t in range(trajectory_length):
            o = obs_l[t]
            a = act_l[t]

            # Failure or invalid feedback => minimal weight
            if any(marker in o for marker in failure_markers):
                base_w = 0.0
            else:
                # Action usefulness based on verbs
                verb_bonus = 0.0
                for v in verbs_primary:
                    if v in a or v in o:
                        verb_bonus += 1.0
                # Diminishing returns for repeated identical actions
                repeat_penalty = 1.0 / np.sqrt(max(1, action_counts[a]))

                # Strong subgoal signal if placing objects somewhere
                subgoal_bonus = 0.0
                if ("put" in a or "insert" in a or "place" in a) and (" in " in a or " into " in a):
                    subgoal_bonus += 1.0
                if "you put" in o or "you place" in o or "you insert" in o:
                    subgoal_bonus += 1.0

                # General positive step if not failure and some action attempted
                base_w = 0.2 + 0.8 * (verb_bonus > 0)
                base_w += 0.5 * subgoal_bonus
                base_w *= repeat_penalty

            # Extra success markers in observations give more credit
            if any(m in o for m in success_markers):
                base_w += 2.0

            weights[t] = max(0.0, base_w)

        # Prefer final step if episode success and no weight
        if weights.sum() <= 0.0:
            if episode_reward > 0.0 and trajectory_length > 0:
                weights[-1] = 1.0
            else:
                return np.zeros(trajectory_length, dtype=np.float32)

        # Slightly bias later steps as they are often closer to completion
        if trajectory_length > 1:
            time_bias = np.linspace(0.9, 1.1, trajectory_length).astype(np.float32)
            weights *= time_bias

        # Normalize to sum 1 then scale by episode reward
        total = float(weights.sum())
        if total > 0:
            weights = weights / total
        step_rewards = weights * float(episode_reward)
        return step_rewards.astype(np.float32)