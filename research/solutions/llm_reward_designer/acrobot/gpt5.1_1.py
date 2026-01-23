import numpy as np


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gpt5.1_acrobot_1

        Direct progress shaping on tip height + phase-dependent velocity terms.
        Simple, dense signal: reward height increases, build energy when low,
        and damp velocities once near the goal.
        """

        def height(s: np.ndarray) -> float:
            cos_t1, sin_t1, cos_t2, sin_t2 = float(s[0]), float(s[1]), float(s[2]), float(s[3])
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            return -cos_t1 - cos_t1_plus_t2

        h_curr = height(state)
        h_next = height(next_state)

        d1, d2 = float(next_state[4]), float(next_state[5])

        # Base ground-truth aligned cost
        reward = -1.0

        # 1) Reward progress in height (very dense shaping)
        reward += (h_next - h_curr) * 18.0

        # 2) When still low, encourage kinetic energy (swing-building)
        if h_next < 0.5:
            energy = d1 * d1 + 0.5 * d2 * d2
            reward += 0.02 * energy

        # 3) When close to the goal, penalize high velocities for stabilization
        if h_next > 0.9:
            reward -= 0.02 * (d1 * d1 + 0.5 * d2 * d2)

        # 4) Terminal bonus for actually reaching the goal height
        if done and h_next > 1.0:
            reward += 80.0

        return float(reward)


