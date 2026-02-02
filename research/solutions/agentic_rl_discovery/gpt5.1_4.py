import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List
from verl.utils.torch_functional import masked_mean, masked_whiten, entropy_from_logits


class Solution:
    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm. Called once before training.
        """
        self.config: Dict[str, Any] = {
            "gamma": 0.95,           # Discount factor along tokens
            "clip_ratio": 0.2,       # PPO clip range
            "use_kl_loss": False,    # Add KL penalty
            "kl_loss_coef": 0.01,    # KL penalty coefficient
        }
        return self

    @staticmethod
    def _make_anchor_key(obs: Any) -> Any:
        """
        Create a hashable key for an anchor observation, robust to numpy arrays.
        """
        # Simple scalar / string types
        if isinstance(obs, (int, float, str, bytes, np.number)):
            return obs
        try:
            arr = np.asarray(obs)
            if arr.ndim == 0:
                try:
                    return arr.item()
                except Exception:
                    return str(arr)
            # Use shape, dtype and bytes as key to be robust
            return (arr.shape, str(arr.dtype), arr.tobytes())
        except Exception:
            # Fallback to string representation
            return str(obs)

    def compute_advantage(
        self,
        token_level_rewards: torch.Tensor,   # (batch, seq_len)
        response_mask: torch.Tensor,         # (batch, seq_len)
        episode_index: np.ndarray,           # (batch,)
        trajectory_index: np.ndarray,        # (batch,) - unused but kept for API
        step_rewards: Optional[torch.Tensor] = None,
        anchor_observations: Optional[np.ndarray] = None,
        gamma: float = 0.95,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns for policy update.
        """
        device = token_level_rewards.device
        dtype = token_level_rewards.dtype
        batch_size, seq_len = token_level_rewards.shape

        # Ensure mask is float for arithmetic
        resp_mask_float = response_mask.to(dtype)

        # Sequence-level reward: sum of token rewards within the response
        seq_rewards = (token_level_rewards * resp_mask_float).sum(dim=1)  # (B,)

        # Optionally add external per-step rewards (e.g., from assign_step_rewards)
        if step_rewards is not None:
            if isinstance(step_rewards, torch.Tensor):
                sr = step_rewards.to(device=device, dtype=dtype)
                if sr.dim() == 0:
                    sr = sr.expand(batch_size)
                elif sr.dim() > 1:
                    sr = sr.view(batch_size, -1).sum(dim=1)
            else:
                sr_np = np.asarray(step_rewards, dtype=np.float32)
                if sr_np.size == 1:
                    sr_np = np.repeat(sr_np, batch_size)
                sr = torch.from_numpy(sr_np).to(device=device, dtype=dtype)
                if sr.numel() != batch_size:
                    sr = sr.view(batch_size, -1).sum(dim=1)
            seq_rewards = seq_rewards + sr

        # Identify sequences that actually have any response tokens
        resp_lengths = resp_mask_float.sum(dim=1)  # (B,)
        valid_seq_mask = resp_lengths > 0  # (B,)

        if not bool(valid_seq_mask.any()):
            # No valid sequences: return zeros
            advantages = torch.zeros_like(token_level_rewards)
            returns = torch.zeros_like(token_level_rewards)
            return advantages, returns

        # Convert to numpy for baseline & grouping (no gradients needed for rewards)
        seq_rewards_np = seq_rewards.detach().cpu().numpy().astype(np.float32)
        valid_seq_np = valid_seq_mask.detach().cpu().numpy().astype(bool)

        # Episode indices
        ep_index_np = np.asarray(episode_index)
        if ep_index_np.shape[0] != batch_size:
            ep_index_np = ep_index_np.reshape(batch_size)

        # Initialize baseline with global mean over valid sequences
        baseline_np = np.zeros(batch_size, dtype=np.float32)
        valid_rewards = seq_rewards_np[valid_seq_np]
        global_mean = float(valid_rewards.mean()) if valid_rewards.size > 0 else 0.0
        baseline_np[valid_seq_np] = global_mean

        has_step_group = np.zeros(batch_size, dtype=bool)

        # Step-level grouping via anchor_observations (GiGPO-style)
        if anchor_observations is not None and valid_rewards.size > 0:
            anchor_array = np.asarray(anchor_observations)
            if anchor_array.shape[0] != batch_size:
                # Try to reshape assuming leading dimension is batch
                anchor_array = anchor_array.reshape(batch_size, -1)

            step_groups: Dict[Any, List[int]] = defaultdict(list)
            for i in range(batch_size):
                if not valid_seq_np[i]:
                    continue
                obs_i = anchor_array[i]
                key = self._make_anchor_key(obs_i)
                step_groups[key].append(i)

            for idxs in step_groups.values():
                if len(idxs) <= 1:
                    continue
                group_rewards = seq_rewards_np[idxs]
                group_sum = float(group_rewards.sum())
                n = len(idxs)
                denom = max(n - 1, 1)
                for idx in idxs:
                    baseline_np[idx] = (group_sum - float(seq_rewards_np[idx])) / denom
                    has_step_group[idx] = True

        # Episode-level grouping (GRPO-style) for sequences not covered by step groups
        episode_groups: Dict[Any, List[int]] = defaultdict(list)
        for i in range(batch_size):
            if not valid_seq_np[i]:
                continue
            ep_id = int(ep_index_np[i])
            episode_groups[ep_id].append(i)

        for idxs in episode_groups.values():
            remaining = [i for i in idxs if (valid_seq_np[i] and not has_step_group[i])]
            if len(remaining) <= 1:
                continue
            group_rewards = seq_rewards_np[remaining]
            group_sum = float(group_rewards.sum())
            n = len(remaining)
            denom = max(n - 1, 1)
            for idx in remaining:
                baseline_np[idx] = (group_sum - float(seq_rewards_np[idx])) / denom

        # Sequence-level advantages with whitening across valid sequences
        adv_seq_np = np.zeros(batch_size, dtype=np.float32)
        diff = seq_rewards_np[valid_seq_np] - baseline_np[valid_seq_np]
        if diff.size > 0:
            mean_diff = float(diff.mean())
            std_diff = float(diff.std())
            if std_diff < 1e-8:
                norm_diff = diff - mean_diff
            else:
                norm_diff = (diff - mean_diff) / (std_diff + 1e-8)
            adv_seq_np[valid_seq_np] = norm_diff

        adv_seq = torch.from_numpy(adv_seq_np).to(device=device, dtype=dtype)

        # Token-level discounting using gamma along response tokens
        # Exponent: number of steps until final response token
        token_index = resp_mask_float.cumsum(dim=1) - 1.0  # 0-based index within response
        last_index = resp_lengths.view(-1, 1) - 1.0
        exp = (last_index - token_index).clamp(min=0.0)
        gamma_tensor = torch.tensor(gamma, device=device, dtype=dtype)
        discounts = torch.pow(gamma_tensor, exp) * resp_mask_float  # (B, T)

        # Returns: discounted sequence reward per token
        returns = seq_rewards.unsqueeze(1) * discounts

        # Advantages: discounted, normalized sequence advantage per token
        advantages = adv_seq.unsqueeze(1) * discounts

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
        """
        Compute PPO-style clipped policy gradient loss with optional KL penalty.
        """
        # Fetch config defaults if arguments are None / missing
        clip_ratio = clip_ratio if clip_ratio is not None else self.config.get("clip_ratio", 0.2)
        use_kl_loss = self.config.get("use_kl_loss", False)
        kl_coef = self.config.get("kl_loss_coef", 0.01)

        # Flatten all tensors using the response mask
        mask = response_mask.bool()
        if not bool(mask.any()):
            loss = torch.zeros((), device=log_prob.device, dtype=log_prob.dtype)
            metrics = {
                "policy_loss": 0.0,
                "approx_kl": 0.0,
                "clip_frac": 0.0,
            }
            return loss, metrics

        logp = log_prob[mask]          # (N,)
        old_logp = old_log_prob[mask]  # (N,)
        adv = advantages[mask]         # (N,)

        # Normalize advantages (extra stabilization)
        adv_mean = adv.mean()
        adv_std = adv.std()
        if torch.isnan(adv_std) or adv_std < 1e-8:
            adv_norm = adv - adv_mean
        else:
            adv_norm = (adv - adv_mean) / (adv_std + 1e-8)

        # PPO ratio
        ratio = torch.exp(logp - old_logp)  # (N,)

        # Surrogate objectives
        surr1 = ratio * adv_norm
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_norm
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        # Approximate KL divergence for logging & optional penalty
        approx_kl = 0.5 * torch.mean((logp - old_logp) ** 2)

        loss = policy_loss
        if use_kl_loss and kl_coef > 0.0:
            loss = loss + kl_coef * approx_kl

        # Clipping fraction
        with torch.no_grad():
            clip_frac = torch.mean((torch.abs(ratio - 1.0) > clip_ratio).float()).item()
            metrics: Dict[str, Any] = {
                "policy_loss": float(policy_loss.detach().cpu().item()),
                "loss": float(loss.detach().cpu().item()),
                "approx_kl": float(approx_kl.detach().cpu().item()),
                "clip_frac": clip_frac,
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
        """
        Distribute episode reward to individual steps.

        We use a simple shaping:
        - If episode_reward <= 0: all steps get 0 reward (no learning signal).
        - If episode_reward > 0: 50% of the reward is spread evenly across all steps,
          and 50% is assigned to the final step to encourage task completion and
          shorter successful trajectories.
        """
        T = int(trajectory_length)
        if T <= 0:
            return np.zeros(0, dtype=np.float32)

        if episode_reward <= 0.0:
            # Unsuccessful episode: no positive feedback, let advantages come
            # from contrast with successful trajectories.
            return np.zeros(T, dtype=np.float32)

        # Successful episode: distribute reward
        episode_reward = float(episode_reward)
        base_share = 0.5
        final_share = 0.5

        step_rewards = np.full(T, episode_reward * (base_share / T), dtype=np.float32)
        step_rewards[-1] += episode_reward * final_share
        return step_rewards