import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Gemini3_pro_3: Progress shaping + velocity damping near goal + terminal bonus.
        This often improves final steps (reduces overshoot / chaotic swinging).
        """
        def height(s):
            cos_t1, sin_t1, cos_t2, sin_t2 = s[0], s[1], s[2], s[3]
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            return -cos_t1 - cos_t1_plus_t2

        h = height(state)
        h2 = height(next_state)

        base = -1.0

        # reward progress in height directly (dense)
        progress = (h2 - h) * 15.0

        # once we're high, damp velocities to stabilize the final "tip above 1.0" condition
        d1, d2 = float(next_state[4]), float(next_state[5])
        damp = 0.0
        if h2 > 0.8:
            damp = -0.02 * (d1 * d1 + 0.5 * d2 * d2)

        # encourage decisive torque away from dithering a bit (can help exploration)
        torque_bonus = 0.0
        if action == 0 or action == 2:
            torque_bonus = 0.01

        bonus = 0.0
        if done and h2 > 1.0:
            bonus = 70.0

        return base + progress + damp + torque_bonus + bonus