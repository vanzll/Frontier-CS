import math

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Gemini3 Pro Solution 3: Kinetic Energy Damping
        
        The main reason CartPole fails is oscillation getting out of control.
        We explicitly punish Kinetic Energy (Total Energy of the system relative to upright equilibrium).
        
        Lyapunov-like reward function.
        V(s) = 0.5 * x^2 + 0.5 * x_dot^2 + 0.5 * theta^2 + 0.5 * theta_dot^2
        Reward = -V(s) + Survival Bonus
        """
        
        x, x_dot, theta, theta_dot = next_state
        
        # Base Survival Reward
        reward = 1.0
        
        # Energy / Error Penalty
        # We weigh theta much higher because it's critical.
        penalty = (
            0.1 * (x ** 2) + 
            0.1 * (x_dot ** 2) + 
            10.0 * (theta ** 2) + 
            1.0 * (theta_dot ** 2)
        )
        
        reward -= penalty
        
        # Terminal Penalty
        if done:
            # Check for failure conditions
            x_threshold = 2.4
            theta_threshold = 0.2095
            if abs(x) >= x_threshold or abs(theta) >= theta_threshold:
                reward -= 50.0 # Heavy penalty for crashing
                
        return reward

