import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Deepseek Reasoner Solution 2: Kinetic Energy Damping.
        
        Explicitly penalizes the total energy of the system to prevent 
        oscillations from growing out of control.
        """
        x, x_dot, theta, theta_dot = next_state
        
        # Base reward
        reward = 1.0
        
        # Kinetic energy terms
        # Rotational kinetic energy is most critical
        ke_rot = 0.5 * (theta_dot**2)
        ke_trans = 0.5 * (x_dot**2)
        
        # Potential energy (deviation from vertical)
        pe = 1.0 - np.cos(theta)
        
        # Total deviation cost
        energy_cost = 20.0 * pe + 2.0 * ke_rot + 0.1 * ke_trans
        
        # Add position cost to prevent drifting
        pos_cost = 0.5 * (x**2)
        
        reward -= (energy_cost + pos_cost)
        
        if done and (abs(theta) > 0.209 or abs(x) > 2.4):
            reward -= 50.0 # Heavy penalty for losing control
            
        return float(reward)

