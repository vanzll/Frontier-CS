import numpy as np


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gemini2.5pro: Acrobot Reward Function 0

        Potential-based shaping on tip height, with an added incentive for kinetic energy
        when the acrobot is far from the goal. Damps velocities near the target height.
        """

        def get_height(s: np.ndarray) -> float:
            # Reconstruct height from trig observations
            cos_t1, sin_t1, cos_t2, sin_t2 = float(s[0]), float(s[1]), float(s[2]), float(s[3])
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            return -cos_t1 - cos_t1_plus_t2  # Range is roughly [-2.0, 2.0]

        h_curr = get_height(state)
        h_next = get_height(next_state)

        dt1, dt2 = float(next_state[4]), float(next_state[5])
        kinetic_energy_proxy = dt1**2 + dt2**2

        # Base reward: -1.0 per step (time penalty)
        reward = -1.0

        # Potential-based shaping for height
        gamma = 0.99
        shaping_reward = (gamma * h_next - h_curr) * 15.0
        reward += shaping_reward

        # Incentive for kinetic energy when still low (to build swing)
        if h_next < 0.5:  # Far from the goal
            reward += 0.01 * kinetic_energy_proxy

        # Damping for velocities when near the goal height to promote stability
        if h_next > 0.8:  # Close to goal
            reward -= 0.02 * kinetic_energy_proxy

        # Terminal bonus for successful swing-up
        if done and h_next > 1.0:
            reward += 75.0

        return float(reward)
