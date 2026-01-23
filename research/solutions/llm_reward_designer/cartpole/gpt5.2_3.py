import math

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:

        def phi(s):
            x, x_dot, theta, theta_dot = s
            x_th = 2.4
            th_th = 12 * 2 * math.pi / 360
            # higher phi is better: close to center + upright + low velocities
            return (
                1.0
                - 0.5 * (abs(x) / x_th)
                - 1.0 * (abs(theta) / th_th)
                - 0.02 * (x_dot * x_dot)
                - 0.01 * (theta_dot * theta_dot)
            )

        gamma = 0.99
        shaping = gamma * phi(next_state) - phi(state)

        # keep a small survival bonus so "stay alive" remains primary
        reward = 0.2 + 5.0 * shaping

        # terminal penalty
        x, _, theta, _ = next_state
        x_th = 2.4
        th_th = 12 * 2 * math.pi / 360
        if done and (abs(x) >= x_th or abs(theta) >= th_th):
            reward -= 5.0

        return float(reward)