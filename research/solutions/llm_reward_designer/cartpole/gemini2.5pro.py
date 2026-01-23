import math


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gemini2.5pro: CartPole Reward Function 0

        Potential-based shaping using a weighted combination of `centerness` and `uprightness`.
        This provides a dense signal guiding the pole to the center and upright position,
        while maintaining the base survival reward.
        """

        def get_potential(s: np.ndarray) -> float:
            x, _, theta, _ = s
            x_threshold = 2.4
            theta_threshold = 12 * 2 * math.pi / 360.0  # ~0.209 radians

            # We want to maximize centerness and uprightness.
            # Potential should be higher for better states.
            centerness = max(0.0, 1.0 - abs(x) / x_threshold)
            uprightness = max(0.0, 1.0 - abs(theta) / theta_threshold)

            return (centerness * 0.5) + (uprightness * 0.5)  # Simple average

        gamma = 0.99
        phi_curr = get_potential(state)
        phi_next = get_potential(next_state)

        shaping_reward = (gamma * phi_next - phi_curr) * 10.0

        reward = 1.0  # Base survival reward (ground truth)
        reward += shaping_reward

        # Additional penalty for terminal states that are failures
        if done:
            x, _, theta, _ = next_state
            x_threshold = 2.4
            theta_threshold = 12 * 2 * math.pi / 360.0
            if abs(x) >= x_threshold or abs(theta) >= theta_threshold:
                reward -= 15.0  # Significant penalty for failing

        return float(reward)

