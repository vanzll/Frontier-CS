import math


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gpt5.1_cartpole_1

        Direct dense reward:
            reward = survival_bonus * (centerness * uprightness) - velocity_penalty

        No PBRS; very simple and strong shaping for PPO.
        """

        x, x_dot, theta, theta_dot = next_state

        x_th = 2.4
        th_th = 12 * 2 * math.pi / 360.0

        centerness = max(0.0, 1.0 - abs(x) / x_th)
        uprightness = max(0.0, 1.0 - abs(theta) / th_th)

        velocity_penalty = 0.01 * (x_dot * x_dot) + 0.02 * (theta_dot * theta_dot)

        # Survival-style base term
        base = 1.0
        shaped = base * (centerness * uprightness) - velocity_penalty

        reward = shaped

        # Extra penalty on clear failures
        if done and (abs(x) >= x_th or abs(theta) >= th_th):
            reward -= 8.0

        return float(reward)


