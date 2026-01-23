import numpy as np


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gemini2.5pro_2: Acrobot Reward Function 2

        A more nuanced potential function combining height and a damped velocity term.
        Encourages both reaching height and then stabilizing with minimal velocity.
        Also includes a small bonus for non-zero torque to encourage active control.
        """

        def get_height(s: np.ndarray) -> float:
            cos_t1, sin_t1, cos_t2, sin_t2 = float(s[0]), float(s[1]), float(s[2]), float(s[3])
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            return -cos_t1 - cos_t1_plus_t2

        def get_potential(s: np.ndarray) -> float:
            h = get_height(s)
            dt1, dt2 = float(s[4]), float(s[5])
            # Combine height with a velocity damping term: higher is better
            return h - 0.08 * (dt1**2 + 0.5 * dt2**2)  # Penalize high velocities slightly

        pot_curr = get_potential(state)
        pot_next = get_potential(next_state)

        # Base reward: -1.0 per step
        reward = -1.0

        # Potential-based shaping
        gamma = 0.99
        shaping_reward = (gamma * pot_next - pot_curr) * 25.0
        reward += shaping_reward

        # Small bonus for applying torque (active control is often better than passive)
        if action != 1:  # Action 1 is zero torque
            reward += 0.02

        # Terminal bonus
        h_next = get_height(next_state)
        if done and h_next > 1.0:
            reward += 90.0

        return float(reward)
