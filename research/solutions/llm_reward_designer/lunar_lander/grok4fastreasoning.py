import math

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Grok4FastReasoning Solution: Advanced Potential-Based Shaping for LunarLander

        State: [x, y, vx, vy, theta, vtheta, l_contact, r_contact]

        Goal: Land at (0,0) with low velocity and upright orientation

        Strategy: Strong PBRS with distance/velocity/angle potentials + fuel costs
        """

        x, y, vx, vy, theta, vtheta, _, _ = next_state

        # 1. Distance potential (closer to (0,0) is better)
        distance = math.sqrt(x*x + y*y)
        dist_potential = -distance * 2.0

        # 2. Velocity potential (lower speed is better, especially near ground)
        speed = math.sqrt(vx*vx + vy*vy)
        vel_potential = -speed * 1.5

        # 3. Angle potential (upright is better)
        angle_potential = -abs(theta) * 3.0

        # 4. Angular velocity potential
        ang_vel_potential = -abs(vtheta) * 1.0

        # Combine potentials (gamma = 0.99 for PBRS)
        gamma = 0.99
        current_potential = -math.sqrt(state[0]**2 + state[1]**2) * 2.0 - math.sqrt(state[2]**2 + state[3]**2) * 1.5 - abs(state[4]) * 3.0 - abs(state[5]) * 1.0
        next_potential = dist_potential + vel_potential + angle_potential + ang_vel_potential

        shaping_reward = gamma * next_potential - current_potential
        shaping_reward *= 2.0  # Scale up

        # 5. Fuel costs (align with ground truth)
        fuel_cost = 0.0
        if action == 2:  # Main engine
            fuel_cost = 0.3
        elif action == 1 or action == 3:  # Side engines
            fuel_cost = 0.03

        # 6. Terminal rewards (align with ground truth)
        terminal_reward = 0.0
        if done:
            if y <= 0.0:  # On ground
                if (abs(x) < 0.2 and abs(theta) < 0.2 and
                    abs(vy) < 0.5 and abs(vx) < 0.5):
                    terminal_reward = 100.0  # Successful landing
                else:
                    terminal_reward = -100.0  # Crash
            elif abs(x) > 1.5 or y > 2.5:  # Out of bounds
                terminal_reward = -100.0

        return shaping_reward - fuel_cost + terminal_reward
