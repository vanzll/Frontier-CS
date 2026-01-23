import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Potential-Based Reward Shaping (PBRS) for Acrobot.
        
        Phi(s) = -Distance_to_Goal
        Target height is > 1.0. Max height is 2.0.
        Let's define potential as simply the height.
        """
        
        def get_height(s):
            cos_t1 = s[0]
            sin_t1 = s[1]
            cos_t2 = s[2]
            sin_t2 = s[3]
            return -cos_t1 - (cos_t1 * cos_t2 - sin_t1 * sin_t2)

        h_curr = get_height(state)
        h_next = get_height(next_state)
        
        # PBRS formula: F = gamma * Phi(s') - Phi(s)
        gamma = 0.99
        shaping = gamma * h_next - h_curr
        
        # Base Reward
        reward = -1.0
        
        # Add shaping (scaled significantly to compete with -1)
        reward += shaping * 10.0
        
        # Success Bonus
        if done and h_next > 1.0:
            reward += 100.0
            
        return reward

