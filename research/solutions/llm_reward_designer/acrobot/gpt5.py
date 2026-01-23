import numpy as np


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gpt5_acrobot_0

        PBRS on tip height with light damping; strong shaping to help PPO discover swing-up.
        """

        def height(s):
            cos_t1, sin_t1, cos_t2, sin_t2 = s[0], s[1], s[2], s[3]
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            return -cos_t1 - cos_t1_plus_t2

        h0 = height(state)
        h1 = height(next_state)

        gamma = 0.99
        shaping = (gamma * h1 - h0) * 12.0

        reward = -1.0 + shaping

        # velocity damping near goal
        d1, d2 = float(next_state[4]), float(next_state[5])
        if h1 > 0.8:
            reward -= 0.01 * (d1 * d1 + 0.5 * d2 * d2)

        if done and h1 > 1.0:
            reward += 60.0

        return float(reward)


