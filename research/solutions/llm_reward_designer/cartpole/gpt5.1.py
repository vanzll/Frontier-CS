import math


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gpt5.1_cartpole_0

        Potential-based shaping on a simple "goodness" potential:
            Phi(s) = centerness + uprightness - velocity_penalty

        This adds a dense gradient while keeping survival (+1 per step) as the core.
        """

        def phi(s):
            x, x_dot, theta, theta_dot = s

            x_th = 2.4
            th_th = 12 * 2 * math.pi / 360.0

            centerness = max(0.0, 1.0 - abs(x) / x_th)
            uprightness = max(0.0, 1.0 - abs(theta) / th_th)

            vel_pen = 0.02 * (x_dot * x_dot) + 0.01 * (theta_dot * theta_dot)

            # Higher phi is better.
            return centerness + uprightness - vel_pen

        gamma = 0.99
        shaping = gamma * phi(next_state) - phi(state)

        # Base survival reward from ground truth
        reward = 1.0

        # Add reasonably strong shaping
        reward += 3.0 * shaping

        # Extra penalty on true failure terminations (out of bounds or fallen)
        if done:
            x, _, theta, _ = next_state
            x_th = 2.4
            th_th = 12 * 2 * math.pi / 360.0
            if abs(x) >= x_th or abs(theta) >= th_th:
                reward -= 10.0

        return float(reward)


