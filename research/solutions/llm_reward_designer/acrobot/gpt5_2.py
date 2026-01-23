import numpy as np


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gpt5_acrobot_2

        Lyapunov-inspired reward: negative change in a designed energy function.
        """

        def height(s):
            cos_t1, sin_t1, cos_t2, sin_t2 = s[0], s[1], s[2], s[3]
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            return -cos_t1 - cos_t1_plus_t2

        def V(s):
            h = height(s)
            d1, d2 = float(s[4]), float(s[5])
            potential = (2.0 - h) ** 2 * 5.0
            kinetic = 0.5 * d1 * d1 + 0.2 * d2 * d2
            return potential + kinetic

        V0 = V(state)
        V1 = V(next_state)

        reward = -1.0
        reward += -(V1 - V0) * 2.0

        if action != 1:
            reward -= 0.02

        h1 = height(next_state)
        if done and h1 > 1.0:
            reward += 100.0

        return float(reward)


