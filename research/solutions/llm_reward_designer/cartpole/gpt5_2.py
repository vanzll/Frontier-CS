import math


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gpt5_cartpole_2

        Lyapunov-like quadratic energy shaping to drive state to zero.
        """

        x, x_dot, theta, theta_dot = next_state
        x_th = 2.4
        th_th = 12 * 2 * math.pi / 360.0

        V = 0.5 * (x / x_th) ** 2 + 0.1 * (x_dot ** 2) + 6.0 * (theta / th_th) ** 2 + 0.5 * (theta_dot ** 2)
        reward = 1.0 - 2.0 * V

        if done and (abs(x) >= x_th or abs(theta) >= th_th):
            reward -= 12.0

        return float(reward)


