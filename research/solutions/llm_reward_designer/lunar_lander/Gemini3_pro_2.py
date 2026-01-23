import math

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Gemini3 Pro Solution 2: Safety Monitor
        
        Focuses entirely on safety constraints.
        Instead of rewarding "going to goal", we heavily punish "unsafe states".
        Safe state:
        - Upright angle (theta ~ 0)
        - Low angular velocity
        - Low vertical speed (don't fall too fast)
        
        If safety is maintained, we give a small drip reward.
        """
        
        x, y, vx, vy, theta, vtheta, _, _ = next_state
        
        # 1. Survival Reward (Drip)
        reward = 0.1
        
        # 2. Safety Penalties
        # Angle
        if abs(theta) > 0.2: # Tilted
            reward -= 1.0
        
        # Angular Velocity (Spinning is bad)
        if abs(vtheta) > 0.2:
            reward -= 0.5
            
        # Falling too fast?
        if vy < -1.5: # Falling fast
            reward -= 0.5
            
        # 3. Distance Guidance (Weak)
        # Just enough to nudge it towards center
        dist = math.sqrt(x*x + y*y)
        reward -= dist * 0.1
        
        # 4. Terminal Logic
        if done:
            # Re-implement ground truth logic
            landed = False
            crashed = False
            
            if y <= 0.0:
                # Landed or Crashed
                if abs(x) < 0.2 and abs(theta) < 0.2 and abs(vy) < 0.5 and abs(vx) < 0.5:
                    landed = True
                else:
                    crashed = True
            elif abs(x) > 1.5 or y > 2.5:
                crashed = True
                
            if landed:
                reward += 100.0
            elif crashed:
                reward -= 100.0
                
        return reward

