import numpy as np


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gpt5.1_acrobot_0

        Potential-based shaping on tip height with gentle velocity damping near the goal.
        Ground-truth reward is -1 per step until the tip height exceeds 1.0.

        State (6D):
            0: cos(theta1)
            1: sin(theta1)
            2: cos(theta2)
            3: sin(theta2)
            4: theta1_dot
            5: theta2_dot
        """

        def height(s: np.ndarray) -> float:
            cos_t1, sin_t1, cos_t2, sin_t2 = float(s[0]), float(s[1]), float(s[2]), float(s[3])
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            return -cos_t1 - cos_t1_plus_t2  # in roughly [-2, 2]

        h_curr = height(state)
        h_next = height(next_state)

        # Next-step velocities (used for damping near the goal)
        d1, d2 = float(next_state[4]), float(next_state[5])

        # Potential-based shaping (policy-invariant under ideal conditions)
        gamma = 0.99
        shaping = (gamma * h_next - h_curr) * 12.0

        # Base step cost (aligns with ground truth)
        reward = -1.0 + shaping

        # When we're already high, discourage large velocities to make it easier
        # to stay above the 1.0 height threshold instead of wildly overshooting.
        if h_next > 0.8:
            vel_penalty = 0.01 * (d1 * d1 + 0.5 * d2 * d2)
            reward -= vel_penalty

        # Strong bonus for successful swing-up (tip height > 1.0)
        if done and h_next > 1.0:
            reward += 60.0

        return float(reward)


