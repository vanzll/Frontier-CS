import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Reward function for Acrobot-v1.
        
        State layout:
        0: cos(theta1)
        1: sin(theta1)
        2: cos(theta2)
        3: sin(theta2)
        4: theta1_dot
        5: theta2_dot
        """
        
        def get_height(s):
            # Reconstruct height from trig observations
            cos_t1 = s[0]
            sin_t1 = s[1]
            cos_t2 = s[2]
            sin_t2 = s[3]
            
            # cos(t1 + t2) = cos(t1)cos(t2) - sin(t1)sin(t2)
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            
            # Tip height formula from environment: -cos(t1) - cos(t1 + t2)
            # Range is roughly [-2.0, 2.0]
            return -cos_t1 - cos_t1_plus_t2

        h_current = get_height(state)
        h_next = get_height(next_state)
        
        # 1. Potential-based shaping: Reward improvement in height
        # Scale factor 10.0 makes the shaping signal significant compared to the -1 step penalty
        shaping_reward = (h_next - h_current) * 10.0
        
        # 2. Base reward: -1.0 per step (Time penalty)
        # This aligns with the Ground Truth objective: finish as fast as possible
        base_reward = -1.0
        
        # 3. Terminal bonus
        # If successfully finished (height > 1.0), give a large bonus
        bonus = 0.0
        if done:
            # Check if it was a success termination (height > 1.0) or timeout
            # The environment logic says termination happens when height > 1.0
            if h_next > 1.0:
                bonus = 100.0
        
        return base_reward + shaping_reward + bonus