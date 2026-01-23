import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Deepseek Reasoner Solution 2: Stabilized Height Shaping.
        
        Focuses on reaching the target height while penalizing excessive 
        angular velocity when near the goal to prevent "flying past" it.
        """
        def get_height(s):
            cos_t1, sin_t1, cos_t2, sin_t2 = s[0], s[1], s[2], s[3]
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            return -cos_t1 - cos_t1_plus_t2

        h_curr = get_height(state)
        h_next = get_height(next_state)
        t1_dot_next = next_state[4]
        t2_dot_next = next_state[5]
        
        # 1. Height progress
        reward = (h_next - h_curr) * 25.0 - 1.0
        
        # 2. Stability penalty when high
        # If height > 0.5, start penalizing very high velocities to stabilize
        if h_next > 0.5:
            reward -= 0.05 * (abs(t1_dot_next) + abs(t2_dot_next))
            
        # 3. Success Bonus
        if done and h_next > 1.0:
            reward += 120.0
            
        return float(reward)

