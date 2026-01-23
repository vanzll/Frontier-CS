import numpy as np


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gpt5_acrobot_1

        Two-phase energy shaping: encourage kinetic build-up when low, then stabilize height.
        """

        def height(s):
            cos_t1, sin_t1, cos_t2, sin_t2 = s[0], s[1], s[2], s[3]
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            return -cos_t1 - cos_t1_plus_t2

        h0 = height(state)
        h1 = height(next_state)
        d1, d2 = float(next_state[4]), float(next_state[5])

        reward = -1.0

        # progress shaping
        reward += (h1 - h0) * 18.0

        if h1 < 0.5:
            # reward building energy
            reward += 0.02 * (d1 * d1 + 0.5 * d2 * d2)
            if action == 1:
                reward -= 0.05
        else:
            # stabilize near-goal
            reward -= 0.02 * (d1 * d1 + 0.5 * d2 * d2)

        if done and h1 > 1.0:
            reward += 80.0

        return float(reward)


