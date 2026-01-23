import math

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:

        x, x_dot, theta, theta_dot = next_state

        x_th = 2.4
        th_th = 12 * 2 * math.pi / 360  # same as env

        # Smooth dense terms in [0,1]
        centerness = max(0.0, 1.0 - abs(x) / x_th)
        uprightness = max(0.0, 1.0 - abs(theta) / th_th)

        # Small damping to reduce oscillations
        vel_pen = 0.01 * (x_dot * x_dot) + 0.005 * (theta_dot * theta_dot)

        reward = 1.0 * (centerness * uprightness) - vel_pen

        # If we terminated by violation, add a penalty to speed up learning
        if done and (abs(x) >= x_th or abs(theta) >= th_th):
            reward -= 5.0

        return float(reward)