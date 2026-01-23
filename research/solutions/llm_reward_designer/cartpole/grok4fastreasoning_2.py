import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Grok4FastReasoning Solution 2: Multi-Stage Dense Control for CartPole

        State: [x, x_dot, theta, theta_dot]

        Strategy: Adaptive reward based on current stability level
        - When unstable: Focus on angle correction
        - When stable: Focus on centering and fine-tuning
        """

        x, x_dot, theta, theta_dot = next_state

        # Thresholds
        x_threshold = 2.4
        theta_threshold = 12 * 2 * np.pi / 360

        # Base survival reward
        reward = 1.0

        # Adaptive reward based on current angle stability
        abs_theta = abs(theta)
        abs_theta_dot = abs(theta_dot)

        if abs_theta > theta_threshold * 0.5:  # Very tilted
            # Emergency mode: heavily penalize angle and angular velocity
            angle_penalty = (theta / theta_threshold) ** 2 * 8.0
            ang_vel_penalty = (theta_dot / (4 * np.pi)) ** 2 * 4.0  # Max ang vel ~4pi
            reward -= angle_penalty + ang_vel_penalty

            # Ignore position when pole is falling
            position_penalty = 0.0

        elif abs_theta > theta_threshold * 0.2:  # Moderately tilted
            # Recovery mode: balance angle and position
            angle_penalty = (theta / theta_threshold) ** 2 * 4.0
            ang_vel_penalty = abs_theta_dot * 2.0
            position_penalty = (x / x_threshold) ** 2 * 1.0
            reward -= angle_penalty + ang_vel_penalty + position_penalty

        else:  # Relatively stable
            # Fine-tuning mode: optimize centering and smoothness
            position_penalty = (x / x_threshold) ** 2 * 3.0
            vel_penalty = abs(x_dot) * 0.3 + abs_theta_dot * 1.0
            angle_penalty = abs_theta * 5.0  # Linear for precision
            reward -= position_penalty + vel_penalty + angle_penalty

            # Bonus for being very centered and stable
            if abs(x) < 0.1 and abs_theta < 0.01:
                reward += 0.5

        # Terminal penalty
        if done:
            if abs(x) >= x_threshold or abs(theta) >= theta_threshold:
                reward -= 25.0

        return reward
