import math

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:

        x, x_dot, theta, theta_dot = next_state

        # quadratic "energy"
        V = (
            0.5 * (x / 2.4) ** 2 +
            0.1 * (x_dot ** 2) +
            5.0 * (theta / (12 * 2 * math.pi / 360)) ** 2 +
            0.5 * (theta_dot ** 2)
        )

        # survival bonus encourages long episodes, -V pushes to stable equilibrium
        reward = 1.0 - 2.0 * V

        # terminal penalty
        x_th = 2.4
        th_th = 12 * 2 * math.pi / 360
        if done and (abs(x) >= x_th or abs(theta) >= th_th):
            reward -= 10.0

        return float(reward)