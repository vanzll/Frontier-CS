import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Deepseek Reasoner Solution 1: Vector Field Alignment for LunarLander.
        
        Calculates a 'target velocity vector' pointing towards the landing 
        pad and rewards the agent for aligning its current velocity with it.
        """
        x, y, vx, vy, theta, vtheta, l_leg, r_leg = next_state
        
        # 1. Calculate target velocity
        # We want to move towards x=0, and descend at a speed proportional to height
        target_vx = -x * 2.0
        target_vy = -y * 1.5
        
        # Cap max target speeds
        target_vx = np.clip(target_vx, -1.0, 1.0)
        target_vy = np.clip(target_vy, -1.0, 0.0) # Always down or neutral
        
        # 2. Alignment reward (Negative MSE between current and target velocity)
        vel_alignment = -((vx - target_vx)**2 + (vy - target_vy)**2) * 20.0
        
        # 3. Attitude control (Strongly upright)
        attitude = - (theta**2 * 50.0 + vtheta**2 * 10.0)
        
        # 4. Leg contact
        contact = 10.0 * (l_leg + r_leg)
        
        reward = vel_alignment + attitude + contact
        
        # 5. Success/Crash terminal
        if done:
            if y <= 0.0:
                 if abs(x) < 0.2 and abs(theta) < 0.2 and abs(vy) < 0.5 and abs(vx) < 0.5:
                     reward += 500.0
                 else:
                     reward -= 200.0
            else: # Out of bounds
                reward -= 200.0
                
        return float(reward)

