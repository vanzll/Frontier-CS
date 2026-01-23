import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Gemini3_pro_1: PBRS (height potential) + terminal bonus.
        Very strong and usually stable for PPO.
        """
        def height(s):
            cos_t1, sin_t1, cos_t2, sin_t2 = s[0], s[1], s[2], s[3]
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            return -cos_t1 - cos_t1_plus_t2  # tip height proxy in [-2,2]

        h = height(state)
        h2 = height(next_state)

        gamma = 0.99
        # Potential-based shaping (policy-invariant under ideal conditions)
        shaping = (gamma * h2 - h) * 10.0

        base = -1.0
        bonus = 0.0
        if done and h2 > 1.0:
            bonus = 50.0  # keep <= 100 after clipping

        return base + shaping + bonus