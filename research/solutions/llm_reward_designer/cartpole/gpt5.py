import math


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gpt5_cartpole_0

        PBRS on centerness and uprightness with velocity damping.
        """

        def phi(s):
            x, x_dot, theta, theta_dot = s
            x_th = 2.4
            th_th = 12 * 2 * math.pi / 360.0
            cent = max(0.0, 1.0 - abs(x) / x_th)
            upr = max(0.0, 1.0 - abs(theta) / th_th)
            vel_pen = 0.02 * x_dot * x_dot + 0.01 * theta_dot * theta_dot
            return cent + upr - vel_pen

        gamma = 0.99
        shaping = gamma * phi(next_state) - phi(state)

        reward = 1.0 + 3.0 * shaping

        if done:
            x, _, theta, _ = next_state
            x_th = 2.4
            th_th = 12 * 2 * math.pi / 360.0
            if abs(x) >= x_th or abs(theta) >= th_th:
                reward -= 10.0

        return float(reward)


