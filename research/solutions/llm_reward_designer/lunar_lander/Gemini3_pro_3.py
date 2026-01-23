import math

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Gemini3 Pro Solution 3: Vector Field Guidance
        
        We construct an ideal velocity vector V_ideal at every state.
        V_ideal points towards the landing pad (0,0).
        Reward is based on alignment: Dot(V_current, V_ideal).
        """
        
        x, y, vx, vy, theta, vtheta, _, _ = next_state
        
        # 1. Calculate Ideal Velocity Vector
        # We want to move towards (0,0).
        # Vector D = (0-x, 0-y) = (-x, -y).
        # We want V to be proportional to D, but clamped.
        # Desired V = k * D
        
        k = 1.0
        target_vx = -x * k
        target_vy = -y * k
        
        # But we can't go through the ground, so target_vy shouldn't be too negative if close to ground?
        # Actually, simple proportional control is fine for descent.
        
        # 2. Velocity Alignment Reward
        # Minimize error: (vx - target_vx)^2 + (vy - target_vy)^2
        vel_error_sq = (vx - target_vx)**2 + (vy - target_vy)**2
        reward = -vel_error_sq
        
        # 3. Angle Control
        # We want theta = 0.
        reward -= 5.0 * (theta ** 2)
        reward -= 1.0 * (vtheta ** 2)
        
        # 4. Terminal Rewards (Critical)
        if done:
            landed = False
            crashed = False
            if y <= 0.0:
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
                
        # Scale down the shaping reward to not overshadow the terminal reward too much
        # Current shaping is approx -(1.5^2 + 1.5^2) ~= -4.5 per step. 
        # 200 steps -> -900. Might be too large compared to +100.
        # Scale it down.
        reward *= 0.1
        
        return reward

