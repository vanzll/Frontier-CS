import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Deepseek Reasoner Solution: Dense Guidance for LunarLander.
        
        Provides dense rewards for getting closer to the pad, maintaining 
        upright orientation, and reducing velocity as the lander descends.
        """
        x, y, vx, vy, theta, vtheta, l_leg, r_leg = next_state
        
        # 1. Distance to target (0,0)
        dist = np.sqrt(x**2 + y**2)
        dist_shaping = -dist * 10.0
        
        # 2. Velocity shaping (Reward slowing down near the ground)
        vel = np.sqrt(vx**2 + vy**2)
        vel_shaping = -vel * 2.0
        if y < 0.5:
             # Heavier penalty for fast descent near ground
             vel_shaping -= (abs(vy) * 5.0 + abs(vx) * 2.0)
        
        # 3. Angle and angular velocity
        angle_shaping = -abs(theta) * 15.0 - abs(vtheta) * 2.0
        
        # 4. Leg contact (Bonus for touching ground)
        leg_bonus = 5.0 * (l_leg + r_leg)
        
        # 5. Fuel usage (Action-based penalty)
        # Action 2 is main engine, 1 and 3 are side engines
        fuel_penalty = 0.0
        if action == 2:
            fuel_penalty = 0.1
        elif action in [1, 3]:
            fuel_penalty = 0.01
            
        reward = dist_shaping + vel_shaping + angle_shaping + leg_bonus - fuel_penalty
        
        # 6. Terminal Rewards (Align with Ground Truth but scale up)
        if done:
            if y <= 0.0 and abs(x) < 0.2 and abs(theta) < 0.2 and abs(vy) < 0.5 and abs(vx) < 0.5:
                reward += 200.0 # Landing success
            elif y <= 0.0 or abs(x) > 1.5 or y > 2.5:
                reward -= 200.0 # Crash or OOB
                
        return float(reward)

