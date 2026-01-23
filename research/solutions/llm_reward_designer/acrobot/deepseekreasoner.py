import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Deepseek Reasoner Solution: Potential-Based Reward Shaping for Acrobot.
        
        Uses height as the potential function to provide a dense gradient 
        towards the goal while remaining policy-invariant.
        """
        def get_height(s):
            cos_t1 = s[0]
            sin_t1 = s[1]
            cos_t2 = s[2]
            sin_t2 = s[3]
            # cos(t1 + t2) = cos(t1)cos(t2) - sin(t1)sin(t2)
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            return -cos_t1 - cos_t1_plus_t2

        h_curr = get_height(state)
        h_next = get_height(next_state)
        
        # 1. PBRS: F = gamma * Phi(s') - Phi(s)
        gamma = 0.99
        shaping = (gamma * h_next - h_curr) * 15.0
        
        # 2. Survival penalty (Ground Truth)
        reward = -1.0 + shaping
        
        # 3. Success Bonus
        if done and h_next > 1.0:
            reward += 100.0
            
        return float(reward)

