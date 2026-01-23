import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Deepseek Reasoner Solution 1: Energy-Based Shaping for Acrobot.
        
        Encourages the agent to build mechanical energy (kinetic + potential) 
        to reach the required height.
        """
        def get_mechanical_energy(s):
            cos_t1 = s[0]
            sin_t1 = s[1]
            cos_t2 = s[2]
            sin_t2 = s[3]
            t1_dot = s[4]
            t2_dot = s[5]
            
            # Potential height proxy
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            height = -cos_t1 - cos_t1_plus_t2
            
            # Kinetic energy proxy (sum of squared velocities)
            kinetic = 0.5 * (t1_dot**2 + t2_dot**2)
            
            # Weight height more as it's the primary goal
            return 2.0 * height + 0.1 * kinetic

        e_curr = get_mechanical_energy(state)
        e_next = get_mechanical_energy(next_state)
        
        # Progress-based shaping
        shaping = (e_next - e_curr) * 20.0
        
        # Base reward
        reward = -1.0 + shaping
        
        # Success Bonus
        if done:
            # Re-calculate height for success check
            cos_t1 = next_state[0]
            sin_t1 = next_state[1]
            cos_t2 = next_state[2]
            sin_t2 = next_state[3]
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            if -cos_t1 - cos_t1_plus_t2 > 1.0:
                reward += 150.0
                
        return float(reward)

