import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Energy-Based Reward Shaping for Acrobot-v1.
        
        The goal is to swing the end effector up to a height of at least 1.0.
        The state contains [cos(t1), sin(t1), cos(t2), sin(t2), dt1, dt2].
        
        We can encourage the agent to increase the total energy of the system.
        Total Energy = Kinetic Energy + Potential Energy.
        """
        
        # Unpack state (next_state)
        # s[0]=cos(t1), s[1]=sin(t1), s[2]=cos(t2), s[3]=sin(t2), s[4]=dt1, s[5]=dt2
        cos_t1 = next_state[0]
        sin_t1 = next_state[1]
        cos_t2 = next_state[2]
        sin_t2 = next_state[3]
        dt1 = next_state[4]
        dt2 = next_state[5]
        
        # Calculate Height (Potential Energy Proxy)
        # Tip y-coordinate = -cos(t1) - cos(t1 + t2)
        # cos(t1+t2) = cos(t1)cos(t2) - sin(t1)sin(t2)
        height = -cos_t1 - (cos_t1 * cos_t2 - sin_t1 * sin_t2)
        
        # Kinetic Energy Proxy
        # Real KE is complex, but dt1^2 + dt2^2 is a decent proxy for "movement"
        kinetic = dt1**2 + dt2**2
        
        # Base Reward: -1 per step (Time penalty)
        reward = -1.0
        
        # Shaping: Reward being high
        # Height is in [-2, 2]. Target > 1.0.
        # We give a small dense reward for height to guide it up.
        reward += height * 0.1
        
        # Shaping: Reward energy (swinging) when low
        if height < 0.5:
            reward += kinetic * 0.01
            
        # Terminal Bonus
        if done and height > 1.0:
            reward += 100.0
            
        return reward

