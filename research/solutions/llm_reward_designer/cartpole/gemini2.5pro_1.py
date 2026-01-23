import math


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gemini2.5pro_1: CartPole Reward Function 1

        Direct dense reward focusing on keeping the pole upright and centered,
        with additional penalties for high velocities to promote stability.
        """

        x, x_dot, theta, theta_dot = next_state

        x_threshold = 2.4
        theta_threshold = 12 * 2 * math.pi / 360.0

        # Penalize distance from center and angle from vertical
        position_penalty = (x / x_threshold) ** 2 * 2.0
        angle_penalty = (theta / theta_threshold) ** 2 * 5.0

        # Penalize high velocities
        velocity_penalty = (x_dot**2) * 0.1 + (theta_dot**2) * 0.5

        reward = 1.0  # Base survival reward
        reward -= position_penalty
        reward -= angle_penalty
        reward -= velocity_penalty

        # Stronger penalty for definitive failures
        if done and (abs(x) >= x_threshold or abs(theta) >= theta_threshold):
            reward -= 20.0

        return float(reward)

