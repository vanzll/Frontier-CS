import numpy as np


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gpt5.1_acrobot_2

        Potential-based shaping on a combined (height - velocity) potential.
        Encourages fast swing-up while softly penalizing unnecessary motion
        near the goal. Adds a small bias against zero-torque dithering.
        """

        def height(s: np.ndarray) -> float:
            cos_t1, sin_t1, cos_t2, sin_t2 = float(s[0]), float(s[1]), float(s[2]), float(s[3])
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            return -cos_t1 - cos_t1_plus_t2

        def phi(s: np.ndarray) -> float:
            h = height(s)
            d1, d2 = float(s[4]), float(s[5])
            # Higher is better: large height, moderate velocities.
            return h - 0.05 * (abs(d1) + 0.5 * abs(d2))

        gamma = 0.99
        shaping = (gamma * phi(next_state) - phi(state)) * 20.0

        reward = -1.0 + shaping

        # Very small penalty on "do nothing" torque to reduce dithering,
        # but still allow it when beneficial.
        if action == 1:
            reward -= 0.01

        h_next = height(next_state)
        if done and h_next > 1.0:
            reward += 70.0

        return float(reward)


