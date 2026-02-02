import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional

# verl utilities are provided in the evaluation environment
try:
    from verl.utils.torch_functional import masked_mean
except ImportError:
    # Fallback implementation for local testing if verl is not installed.
    def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        """Computes the mean of a tensor over a masked region."""
        tensor = tensor * mask
        # Add a small epsilon to the denominator to avoid division by zero.
        denominator = mask.sum(dim=dim).clamp(min=1e-8)
        if dim is None:
            return tensor.sum() / denominator
        return tensor.sum(dim=dim) / denominator

class Solution:
    """
    Implements a complete reinforcement learning algorithm based on GiGPO (Group-wise
    improvement-guided Policy Optimization) for training LLM agents.

    The algorithm consists of three main components:
    1. Reward Assignment: A sparse reward is assigned to the final step of a
       successful trajectory.
    2. Advantage Computation: GiGPO-style advantage estimation is used. It computes
       step-level returns and normalizes them within groups of trajectories that
       share the same starting state (episode_index) and an intermediate state
       representation (anchor_observation). This provides a more stable and
       fine-grained advantage signal compared to trajectory-level normalization.
    3. Policy Loss: The standard PPO-clip objective function is used for policy
       updates, ensuring stable learning by limiting the change in the policy
       at each step.
    """

    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm and set hyperparameters. This method is called
        once before the start of training.
        """
        self.config = {
            # Discount factor for future rewards. A high value is chosen as
            # ALFWorld tasks can have up to 50 steps.
            "gamma": 0.99,
            # PPO clipping ratio to stabilize training. 0.2 is a standard value.
            "clip_ratio": 0.2,
            # Whether to use a KL divergence penalty to further regularize
            # policy updates. Disabled by default for simplicity.
            "use_kl_loss": False,
            # Coefficient for the KL penalty if it is used.
            "kl_loss_coef": 0.01,
        }
        return self

    def assign_step_rewards(
        self,
        episode_reward: float,
        trajectory_length: int,
        step_observations: list,
        step_actions: list,
        **kwargs
    ) -> np.ndarray:
        """
        Distributes the final episode reward to individual steps. For sparse-reward
        tasks like ALFWorld, the entire reward is assigned to the final step
        if the episode was successful.
        """
        step_rewards = np.zeros(trajectory_length, dtype=np.float32)
        # A positive episode_reward indicates a successful trajectory.
        if episode_reward > 0:
            step_rewards[-1] = float(episode_reward)
        return step_rewards

    def compute_advantage(
        self,
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        episode_index: np.ndarray,
        trajectory_index: np.ndarray,
        step_rewards: Optional[torch.Tensor] = None,
        anchor_observations: Optional[np.ndarray] = None,
        gamma: float = 0.99,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes per-token advantages and returns. This implementation uses
        the GiGPO-style group normalization at the step level.
        """
        device = token_level_rewards.device
        batch_size, seq_len = token_level_rewards.shape

        # First, compute standard token-level discounted returns. This will be
        # one of the outputs of this function.
        token_returns = torch.zeros_like(token_level_rewards)
        current_return = torch.zeros(batch_size, device=device)
        for t in range(seq_len - 1, -1, -1):
            current_return = token_level_rewards[:, t] + gamma * current_return
            token_returns[:, t] = current_return
        token_returns *= response_mask

        # GiGPO requires step-level information. If not provided, fallback to
        # GRPO (normalizing token-level returns by episode group).
        if step_rewards is None or anchor_observations is None:
            advantages = torch.zeros_like(token_returns)
            unique_episodes = np.unique(episode_index)
            for ep_idx in unique_episodes:
                group_mask = (episode_index == ep_idx)
                group_returns = token_returns[group_mask]
                group_response_mask = response_mask[group_mask]
                
                masked_values = group_returns[group_response_mask]
                if masked_values.numel() > 1:
                    mean = masked_values.mean()
                    std = masked_values.std()
                    group_adv = (group_returns - mean) / (std + 1e-8)
                else:
                    group_adv = torch.zeros_like(group_returns)
                
                advantages[group_mask] = group_adv * group_response_mask
            return advantages, token_returns

        # -- GiGPO Advantage Calculation --
        # 1. Compute step-level discounted returns from the provided step_rewards.
        max_steps = step_rewards.shape[1]
        step_returns = torch.zeros_like(step_rewards)
        current_step_return = torch.zeros(batch_size, device=device)
        for t in range(max_steps - 1, -1, -1):
            current_step_return = step_rewards[:, t] + gamma * current_step_return
            step_returns[:, t] = current_step_return

        # 2. Normalize step-level returns within state-action groups to get advantages.
        step_advantages = torch.zeros_like(step_returns)
        valid_step_mask = torch.from_numpy(anchor_observations != "").to(device)

        for t in range(max_steps):
            active_mask_t = valid_step_mask[:, t]
            if not active_mask_t.any():
                continue

            active_indices = active_mask_t.nonzero().squeeze(-1)
            active_ep_indices_np = episode_index[active_indices.cpu().numpy()]
            active_anchors_np = anchor_observations[active_indices.cpu().numpy(), t]
            active_returns_t = step_returns[active_indices, t]

            groups = defaultdict(list)
            group_locs = defaultdict(list)
            for i, (ep_idx, anchor) in enumerate(zip(active_ep_indices_np, active_anchors_np)):
                key = (ep_idx, anchor)
                groups[key].append(active_returns_t[i])
                group_locs[key].append(i)

            for key in groups:
                locs_in_active = group_locs[key]
                returns_in_group = torch.stack(groups[key])

                if len(returns_in_group) > 1:
                    mean = returns_in_group.mean()
                    std = returns_in_group.std()
                    advs_in_group = (returns_in_group - mean) / (std + 1e-8)
                else: # Advantage is 0 if a group has only one member
                    advs_in_group = torch.zeros_like(returns_in_group)
                
                original_batch_indices = active_indices[locs_in_active]
                step_advantages[original_batch_indices, t] = advs_in_group

        # 3. Broadcast step-level advantages to all tokens within that step.
        advantages = torch.zeros_like(token_level_rewards)
        for i in range(batch_size):
            step_end_indices = (token_level_rewards[i] != 0).nonzero(as_tuple=False).flatten()
            start_token_idx = 0
            for step_k, end_token_idx in enumerate(step_end_indices):
                if step_k < max_steps:
                    advantages[i, start_token_idx : end_token_idx + 1] = step_advantages[i, step_k]
                start_token_idx = end_token_idx + 1
        
        advantages *= response_mask

        return advantages, token_returns

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
        Computes the PPO-clip policy loss. This loss function encourages actions
        that have a high advantage while preventing excessively large policy
        updates, which leads to more stable training.
        """
        log_ratio = log_prob - old_log_prob
        ratio = torch.exp(log_ratio)

        # Unclipped policy objective
        unclipped_loss = ratio * advantages
        
        # Clipped policy objective
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        clipped_loss = clipped_ratio * advantages

        # The PPO objective is the minimum of the unclipped and clipped objectives
        policy_objective = torch.min(unclipped_loss, clipped_loss)
        
        # The loss is the negative of the mean objective over valid tokens
        loss = -masked_mean(policy_objective, response_mask)

        # Optionally, add a KL divergence penalty to regularize the policy
        if self.config.get("use_kl_loss", False):
            # Approximate KL(pi_old || pi_new)
            kl_div = old_log_prob - log_prob 
            kl_loss = masked_mean(kl_div, response_mask)
            loss += self.config.get("kl_loss_coef", 0.01) * kl_loss

        # Compute metrics for logging and analysis
        with torch.no_grad():
            clip_frac = masked_mean((torch.abs(ratio - 1.0) > clip_ratio).float(), response_mask)

        metrics = {
            "clip_frac": clip_frac.item(),
            "policy_ratio": masked_mean(ratio, response_mask).item(),
            "advantages": masked_mean(advantages, response_mask).item(),
            "policy_loss": loss.item(),
        }
        if self.config.get("use_kl_loss", False) and 'kl_loss' in locals():
            metrics["kl_loss"] = kl_loss.item()
            
        return loss, metrics