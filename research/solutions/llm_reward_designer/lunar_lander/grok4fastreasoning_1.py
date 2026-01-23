import math

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Grok4FastReasoning Solution 1: Multi-Phase Landing Control for LunarLander

        State: [x, y, vx, vy, theta, vtheta, l_contact, r_contact]

        Strategy: Different control objectives for different phases:
        - High altitude: Get above landing pad, control descent
        - Medium altitude: Align with pad, reduce horizontal velocity
        - Low altitude: Final approach with precise control
        """

        x, y, vx, vy, theta, vtheta, _, _ = next_state

        # Determine phase based on altitude
        reward = 0.0

        if y > 0.8:  # High altitude phase
            # Focus on getting roughly above the pad and controlling descent
            dist_to_pad = abs(x)  # Horizontal distance to center
            reward -= dist_to_pad * 1.5  # Get above pad

            # Control vertical velocity (don't fall too fast)
            if vy < -1.0:
                reward -= (abs(vy) - 1.0) * 2.0

            # Keep somewhat upright
            reward -= abs(theta) * 1.0

            # Reward reducing horizontal speed
            reward -= abs(vx) * 0.5

        elif y > 0.3:  # Medium altitude phase
            # Focus on alignment and slowing down
            reward -= abs(x) * 3.0  # Stronger centering
            reward -= abs(vx) * 2.0  # Reduce horizontal speed
            reward -= abs(theta) * 2.0  # Better orientation

            # Control vertical speed more tightly
            if vy < -0.8:
                reward -= (abs(vy) - 0.8) * 3.0

            reward -= abs(vtheta) * 1.0

        else:  # Low altitude / landing phase
            # Precision landing mode
            reward -= abs(x) * 5.0  # Must be very centered
            reward -= abs(theta) * 4.0  # Must be very upright
            reward -= abs(vx) * 3.0  # Must be very slow horizontally
            reward -= max(0, -vy - 0.3) * 4.0  # Don't go down too fast, but allow controlled descent
            reward -= abs(vtheta) * 2.0

            # Bonus for being in landing position
            if abs(x) < 0.15 and abs(theta) < 0.15 and abs(vx) < 0.3 and abs(vy) < 0.4:
                reward += 2.0

        # Fuel costs (scaled appropriately)
        fuel_cost = 0.0
        if action == 2:  # Main engine
            fuel_cost = 0.25  # Slightly less than ground truth for training
        elif action == 1 or action == 3:  # Side engines
            fuel_cost = 0.025

        # Terminal rewards
        if done:
            if y <= 0.0:
                if (abs(x) < 0.2 and abs(theta) < 0.2 and
                    abs(vy) < 0.5 and abs(vx) < 0.5):
                    reward += 100.0
                else:
                    reward += -100.0
            elif abs(x) > 1.5 or y > 2.5:
                reward += -100.0

        return reward - fuel_cost
