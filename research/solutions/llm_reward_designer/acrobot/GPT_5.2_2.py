import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Gemini3_pro_2: PBRS on (height + kinetic energy) + control cost.
        Helps exploration early (build swing energy), then converts to height.
        """
        def height(s):
            cos_t1, sin_t1, cos_t2, sin_t2 = s[0], s[1], s[2], s[3]
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            return -cos_t1 - cos_t1_plus_t2

        def phi(s):
            h = height(s)
            d1, d2 = float(s[4]), float(s[5])
            kinetic = d1 * d1 + 0.25 * d2 * d2
            return h + 0.05 * kinetic

        gamma = 0.99
        shaping = (gamma * phi(next_state) - phi(state)) * 25.0

        base = -1.0

        # small torque usage penalty (discourage chattering; still allows swing)
        control = 0.0
        if action != 1:  # 0:-1, 1:0, 2:+1
            control = -0.02

        bonus = 0.0
        h2 = height(next_state)
        if done and h2 > 1.0:
            bonus = 60.0

        return base + shaping + control + bonus