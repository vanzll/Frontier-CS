import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Deepseek Reasoner Solution 2: Phase-Based Landing Guidance.
        
        Separates the flight into 'Descent' and 'Touchdown' phases with 
        different reward priorities for each.
        """
        x, y, vx, vy, theta, vtheta, l_leg, r_leg = next_state
        
        reward = 0.0
        
        if y > 0.4:
            # Phase 1: Controlled Descent
            # Focus on horizontal positioning and keeping vertical speed in check
            reward -= abs(x) * 5.0
            reward -= abs(vx) * 2.0
            reward -= abs(theta) * 10.0
            # Punish fast falling
            if vy < -1.0:
                reward -= abs(vy) * 10.0
        else:
            # Phase 2: Touchdown
            # High precision required. Zero horizontal movement, very slow vertical.
            reward -= abs(x) * 20.0
            reward -= abs(vx) * 10.0
            reward -= abs(theta) * 30.0
            reward -= abs(vy) * 20.0
            reward += 10.0 * (l_leg + r_leg) # Bonus for legs
            
        # Fuel regularization
        if action != 0:
            reward -= 0.05
            
        # Terminal rewards
        if done:
            if y <= 0.0:
                if abs(x) < 0.2 and abs(theta) < 0.2 and abs(vy) < 0.5 and abs(vx) < 0.5:
                    reward += 1000.0 # Very large success signal
                else:
                    reward -= 100.0
            else:
                reward -= 100.0
                
        return float(reward)

