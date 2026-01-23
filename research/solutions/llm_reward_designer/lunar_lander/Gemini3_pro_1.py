import math

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Strong Reward Shaping for Simplified LunarLander.
        
        State: x, y, vx, vy, theta, vtheta, ...
        
        Goals:
        1. Get close to (0,0) -> Minimize distance.
        2. Land slowly -> Minimize velocity when close to ground.
        3. Stay upright -> Minimize theta and vtheta.
        """
        
        # Unpack state
        # next_state is the state AFTER the action
        x, y, vx, vy, theta, vtheta, _, _ = next_state
        
        # 1. Distance Shaping
        # We want to minimize distance to (0,0).
        # Use potential-based shaping if possible, or just dense reward.
        # Let's use dense penalty for distance.
        dist = math.sqrt(x*x + y*y)
        
        # 2. Velocity Shaping
        # We want small velocity, ESPECIALLY when y is small.
        # But we need high downward velocity when high up to save time/fuel? 
        # Actually PPO will learn to fall fast if we don't punish it.
        # Let's punish total velocity.
        vel = math.sqrt(vx*vx + vy*vy)
        
        # 3. Angle Shaping
        # Punish tilting.
        angle_penalty = abs(theta)
        
        # Reward components
        reward = 0.0
        
        # Dense shaping
        # Normalized roughly: dist ~ [0, 2.5], vel ~ [0, 2.0], angle ~ [0, 0.5]
        reward -= dist * 4.0        # Go to center!
        reward -= vel * 2.0         # Slow down!
        reward -= angle_penalty * 10.0 # Stay upright!
        
        # Fuel penalty (implicit in PPO exploration? No, we should add it)
        # But action is not easily accessible as "engine power" here without looking at action index.
        # action 2 is main engine.
        # If we can't see action semantics easily, we ignore fuel for now in shaping
        # and let the ground truth fuel penalty (which is small) handle it?
        # No, better to guide it.
        if action == 2: # Main engine
            reward -= 0.1
        elif action == 1 or action == 3: # Side engines
            reward -= 0.01
            
        # Terminal Rewards
        # We MUST align with Ground Truth
        if done:
            # Re-check landing condition manually to be sure
            # Condition: y <= 0, abs(x)<0.2, abs(theta)<0.2, abs(vy)<0.5, abs(vx)<0.5
            if y <= 0.0:
                if abs(x) < 0.2 and abs(theta) < 0.2 and abs(vy) < 0.5 and abs(vx) < 0.5:
                    reward += 100.0 # Big bonus for success
                else:
                    reward -= 100.0 # Crash penalty
            elif abs(x) > 1.5 or y > 2.5:
                reward -= 100.0 # OOB penalty
                
        # Shaping for "Flying towards center" (Velocity direction alignment)
        # If position is (x,y), we want velocity to be (-x, -y) roughly.
        # This is advanced, maybe too much.
        
        # Let's add a "Survival" bonus to encourage not crashing immediately?
        # Or a "Approaching" bonus.
        # Shaping: R = old_dist - new_dist
        # This is strictly better mathematically.
        
        return reward

