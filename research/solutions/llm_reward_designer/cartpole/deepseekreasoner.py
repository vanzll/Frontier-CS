import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Deepseek Reasoner Solution: Dense Stability Shaping for CartPole.
        
        Penalizes deviations from the upright position and center of the track, 
        providing a dense signal to keep the pole balanced perfectly.
        """
        x, x_dot, theta, theta_dot = next_state
        
        # 1. Base survival reward
        reward = 1.0
        
        # 2. Angle penalty (Quadratic to punish larger deviations more)
        # Threshold is ~0.209 rad (12 degrees)
        theta_penalty = (theta**2) * 10.0
        
        # 3. Position penalty (Stay near the center)
        # Threshold is 2.4
        x_penalty = (x**2) * 0.5
        
        # 4. Velocity damping (Encourage stability)
        vel_penalty = (theta_dot**2) * 0.1 + (x_dot**2) * 0.01
        
        reward -= (theta_penalty + x_penalty + vel_penalty)
        
        # 5. Terminal penalty if it failed (don't penalize timeout if possible, 
        # but in CartPole 'done' usually means failure)
        if done and (abs(theta) > 0.209 or abs(x) > 2.4):
            reward -= 20.0
            
        return float(reward)

