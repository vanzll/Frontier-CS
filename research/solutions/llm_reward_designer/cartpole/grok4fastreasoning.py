import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Grok4FastReasoning Solution: Enhanced Stability Shaping for CartPole

        State: [x, x_dot, theta, theta_dot]

        Goal: Keep pole upright (theta ≈ 0) and cart centered (x ≈ 0)

        Strategy: Strong potential-based shaping with position and angle penalties
        """

        x, x_dot, theta, theta_dot = next_state

        # Thresholds (from CartPole environment)
        x_threshold = 2.4
        theta_threshold = 12 * 2 * np.pi / 360  # ≈ 0.2094 rad

        # 1. Base survival reward (align with ground truth +1)
        reward = 1.0

        # 2. Position shaping (quadratic penalty, stronger near boundaries)
        x_penalty = (x / x_threshold) ** 2
        reward -= x_penalty * 2.0

        # 3. Angle shaping (quadratic penalty, critical for stability)
        theta_penalty = (theta / theta_threshold) ** 2
        reward -= theta_penalty * 5.0

        # 4. Velocity damping (prevent oscillations)
        vel_penalty = (abs(x_dot) * 0.1 + abs(theta_dot) * 0.5)
        reward -= vel_penalty

        # 5. Terminal penalty for failure
        if done:
            # Check if it's a real failure (not just timeout)
            if abs(x) >= x_threshold or abs(theta) >= theta_threshold:
                reward -= 20.0

        return reward
