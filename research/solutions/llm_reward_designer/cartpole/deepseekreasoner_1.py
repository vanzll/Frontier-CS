import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Deepseek Reasoner Solution 1: Precision Multiplier for CartPole.
        
        Uses a sharp, peaked reward function that rewards being very close 
        to the ideal equilibrium state (0,0,0,0).
        """
        x, x_dot, theta, theta_dot = next_state
        
        # Survival base
        reward = 1.0
        
        # State goodness: exponential decay centered at 0
        # More robust than quadratic for high-precision tasks
        goodness = np.exp(-5.0 * abs(theta)) * np.exp(-0.5 * abs(x))
        
        # Add a bonus for low-velocity equilibrium
        stability = np.exp(-0.1 * abs(theta_dot)) * np.exp(-0.05 * abs(x_dot))
        
        reward += 2.0 * goodness * stability
        
        # Crash penalty
        if done and (abs(theta) > 0.209 or abs(x) > 2.4):
            reward -= 10.0
            
        return float(reward)

