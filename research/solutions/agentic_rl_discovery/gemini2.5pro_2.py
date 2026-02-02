import torch
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional

# verl utilities (optional)
from verl.utils.torch_functional import masked_mean, masked_whiten, entropy_from_logits

class Solution:
    """
    Implements a GRPO-style reinforcement learning algorithm for LLM agents.
    This solution is based on the principles of Group-wise Reward Policy Optimization,
    which has been shown to be effective for training LLM agents.

    The algorithm consists of three main components:
    1.  **Reward Assignment**: Uses a sparse reward scheme where the final episode reward is
        assigned only to the last step of a successful trajectory. This provides a clear
        and unbiased success signal.

    2.  **Advantage Computation**: Implements a core idea from GRPO. It first calculates
        the standard discounted returns for each trajectory. Then, to stabilize training
        and reduce variance, it normalizes these returns within groups of trajectories
        that start from the same initial state. These normalized returns serve as the
        advantages for the policy update.

    3.  **Policy Loss**: Employs the Proximal Policy Optimization (PPO) algorithm with a
        clipping objective. This ensures that the policy updates are not too large,
        leading to more stable and reliable learning.
    """
    def solve(self, spec_path: str = None) -> "Solution":
        """
        Initialize the algorithm and set hyperparameters.
        """
        self.config = {
            "gamma": 0.99,           # Discount factor for future rewards.
            "clip_ratio": 0.2,       # PPO clipping parameter for stabilizing updates.
            "use_kl_loss": False,    # Whether to use a KL-divergence penalty.
            "kl_loss_coef": 0.01,    # Coefficient for the KL penalty if used.
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
        Distributes the final episode reward to individual steps within a trajectory.

        This implementation uses a sparse reward approach. The total episode reward (1.0 for
        success in ALFWorld, 0.0 otherwise) is assigned exclusively to the final step of
        the trajectory. All other steps receive a reward of 0. This encourages the agent
        to learn sequences of actions that lead to task completion.
        """
        step_rewards = np.zeros(trajectory_length, dtype=np.float32)
        if episode_reward > 0:
            step_rewards[-1] = episode_reward
        return step_rewards

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
        """
        Computes advantages and returns using a group-wise normalization strategy (GRPO-style).
        
        The process is as follows:
        1. If step-level rewards are not provided, they are summed from token-level rewards.
        2. Steps are grouped by their trajectory, and discounted returns (G_t) are calculated
           for each step by accumulating future rewards with a discount factor (gamma).
        3. The computed returns are then grouped by the initial episode state. Within each group,
           the returns are normalized (whitened) to have a mean of 0 and a standard deviation of 1.
           These normalized returns serve as the advantages. This critical step reduces variance
           and compares trajectories against others from the same starting point.
        4. Finally, the step-level advantages and returns are broadcast to the token level,
           masked to apply only to the agent's response tokens.
        """
        if step_rewards is None:
            step_rewards = token_level_rewards.sum(dim=1)

        device = step_rewards.device
        batch_size, seq_len = response_mask.shape

        trajectories = defaultdict(list)
        for i in range(batch_size):
            key = (episode_index[i], trajectory_index[i])
            trajectories[key].append(i)
        
        traj_indices_tensors = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in trajectories.items()}

        step_returns = torch.zeros_like(step_rewards)
        for traj_indices in traj_indices_tensors.values():
            r_t = step_rewards[traj_indices]
            
            traj_returns_t = torch.zeros_like(r_t)
            next_return = 0.0
            for t in reversed(range(len(r_t))):
                traj_returns_t[t] = r_t[t] + gamma * next_return
                next_return = traj_returns_t[t]
            
            step_returns[traj_indices] = traj_returns_t

        step_advantages = torch.zeros_like(step_returns)
        unique_episodes = np.unique(episode_index)

        for ep_id in unique_episodes:
            mask = torch.from_numpy(episode_index == ep_id).to(device)
            group_returns = step_returns[mask]
            
            if group_returns.numel() > 1:
                mean = group_returns.mean()
                std = group_returns.std()
                eps = 1e-8
                normalized_advantages = (group_returns - mean) / (std + eps)
                step_advantages[mask] = normalized_advantages

        advantages = step_advantages.unsqueeze(1).expand(-1, seq_len) * response_mask.float()
        returns = step_returns.unsqueeze(1).expand(-1, seq_len) * response_mask.float()

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
        Computes the policy loss using the PPO-Clip objective function.

        The PPO-Clip loss is designed to prevent destructively large policy updates. It works by:
        1. Calculating the probability ratio between the new and old policies.
        2. Creating a surrogate objective that is clipped to be within a small range
           `[1-clip_ratio, 1+clip_ratio]` around 1.
        3. Taking the element-wise minimum of the unclipped and clipped objectives, ensuring
           the update does not deviate too far from the previous policy.
        4. The final loss is the negative of this objective, averaged over all valid response tokens.
        """
        old_log_prob = old_log_prob.detach()

        log_ratio = log_prob - old_log_prob
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        
        policy_loss_per_token = -torch.min(surr1, surr2)
        
        masked_loss = policy_loss_per_token * response_mask
        num_response_tokens = response_mask.sum()
        
        if num_response_tokens > 0:
            loss = masked_loss.sum() / num_response_tokens
        else:
            loss = (log_prob * 0.0).sum()

        metrics = {}
        with torch.no_grad():
            if num_response_tokens > 0:
                clip_mask = ((ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)).float() * response_mask
                clip_frac = clip_mask.sum() / num_response_tokens
                metrics["clip_frac"] = clip_frac.item()
            else:
                metrics["clip_frac"] = 0.0

        use_kl_loss = kwargs.get("use_kl_loss", False)
        if use_kl_loss and num_response_tokens > 0:
            kl_loss_coef = kwargs.get("kl_loss_coef", 0.01)
            kl_div_per_token = old_log_prob - log_prob
            masked_kl_div = kl_div_per_token * response_mask
            kl_loss = (masked_kl_div.sum() / num_response_tokens) * kl_loss_coef
            loss = loss + kl_loss
            metrics["kl_loss"] = kl_loss.item()
            
        return loss, metrics