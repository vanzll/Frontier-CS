import math


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gpt5.1_cartpole_2

        Lyapunov-like shaping on quadratic energy:
            V(s) = ax^2 + bx_dot^2 + c theta^2 + d theta_dot^2
            reward = 1.0 - k * V(s)

        Encourages the system to converge to (0,0,0,0) with a very simple form.
        """

        x, x_dot, theta, theta_dot = next_state

        # Scale by environment thresholds
        x_th = 2.4
        th_th = 12 * 2 * math.pi / 360.0

        V = (
            0.5 * (x / x_th) ** 2
            + 0.1 * (x_dot ** 2)
            + 6.0 * (theta / th_th) ** 2
            + 0.5 * (theta_dot ** 2)
        )

        reward = 1.0 - 2.0 * V

        # Failure penalty to speed up learning of safe region
        if done and (abs(x) >= x_th or abs(theta) >= th_th):
            reward -= 12.0

        return float(reward)


