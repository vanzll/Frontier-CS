import math

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Gemini3 Pro Solution 2: Minimalist Continuous Control
        
        Instead of complex penalties, we just reward the 'Goodness' of the state continuously.
        Goodness = (Uprightness) * (Centerness)
        
        This dense reward signal guides the agent to the perfect state (0,0,0,0) at every step.
        """
        
        x, x_dot, theta, theta_dot = next_state
        
        # 1. Uprightness: cos(theta). Since theta is small, use linear approximation.
        # theta range is approx +- 0.2 rad (12 deg) before done usually, or +-0.4
        # We want to maximize 1 - |theta|/threshold
        theta_threshold = 0.2095
        uprightness = max(0.0, 1.0 - abs(theta) / theta_threshold)
        
        # 2. Centerness: 1 - |x|/2.4
        x_threshold = 2.4
        centerness = max(0.0, 1.0 - abs(x) / x_threshold)
        
        # 3. Combine
        # Dense reward per step.
        # If perfect: 1.0 * 1.0 = 1.0.
        # If falling: approaches 0.
        reward = uprightness * centerness
        
        # 4. Terminal Penalty
        # If we fail (done but steps < max), we get 0 reward for future steps.
        # PPO will naturally try to maximize sum of rewards, i.e., stay alive.
        # But we can add a small crash penalty to be sure.
        if done:
             # Just checking bounds roughly
             if abs(x) >= x_threshold or abs(theta) >= theta_threshold:
                 reward -= 10.0

        return reward
