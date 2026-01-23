import math


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gpt5_cartpole_1

        Dense multiplicative shaping: centerness * uprightness minus velocity penalty.
        """

        x, x_dot, theta, theta_dot = next_state
        x_th = 2.4
        th_th = 12 * 2 * math.pi / 360.0

        cent = max(0.0, 1.0 - abs(x) / x_th)
        upr = max(0.0, 1.0 - abs(theta) / th_th)

        vel_pen = 0.01 * x_dot * x_dot + 0.02 * theta_dot * theta_dot

        reward = 1.0 * (cent * upr) - vel_pen

        if done and (abs(x) >= x_th or abs(theta) >= th_th):
            reward -= 8.0

        return float(reward)


