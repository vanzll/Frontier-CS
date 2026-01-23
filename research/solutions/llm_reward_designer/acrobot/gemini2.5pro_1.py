import numpy as np


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gemini2.5pro_1: Acrobot Reward Function 1

        Direct shaping based on height improvement, with stronger kinetic energy
        incentive for exploration and a small penalty for zero torque action.
        """

        def get_height(s: np.ndarray) -> float:
            cos_t1, sin_t1, cos_t2, sin_t2 = float(s[0]), float(s[1]), float(s[2]), float(s[3])
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            return -cos_t1 - cos_t1_plus_t2

        h_curr = get_height(state)
        h_next = get_height(next_state)

        dt1, dt2 = float(next_state[4]), float(next_state[5])
        kinetic_energy_proxy = dt1**2 + dt2**2

        reward = -1.0  # Base time penalty

        # Reward direct progress in height
        reward += (h_next - h_curr) * 20.0

        # Stronger incentive for kinetic energy when far from the goal
        if h_next < 0.7:  # A bit closer than the previous, still encourages exploration
            reward += 0.03 * kinetic_energy_proxy

        # Penalty for 'do nothing' action to avoid getting stuck or dithering
        if action == 1:  # Torque 0
            reward -= 0.05

        # Terminal bonus
        if done and h_next > 1.0:
            reward += 85.0

        return float(reward)
