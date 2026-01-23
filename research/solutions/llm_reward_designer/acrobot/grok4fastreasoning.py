import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Grok4FastReasoning Solution: Enhanced Potential-Based Shaping for Acrobot

        State: [cos(t1), sin(t1), cos(t2), sin(t2), t1_dot, t2_dot]

        Goal: Maximize tip height h = -cos(t1) - cos(t1+t2) > 1.0

        Strategy: Strong PBRS with height potential + velocity damping + terminal bonus
        """

        def get_height(s):
            cos_t1, sin_t1, cos_t2, sin_t2 = s[0], s[1], s[2], s[3]
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            return -cos_t1 - cos_t1_plus_t2

        h_current = get_height(state)
        h_next = get_height(next_state)

        # 1. Potential-based shaping (gamma=0.99, strong scale)
        gamma = 0.99
        shaping = gamma * h_next - h_current
        shaping_reward = shaping * 15.0  # Stronger than baseline

        # 2. Base time penalty (align with ground truth)
        base_reward = -1.0

        # 3. Velocity damping when high (prevent overshoot)
        _, _, _, _, dt1, dt2 = next_state
        vel_penalty = 0.0
        if h_next > 0.8:  # Near goal region
            vel_penalty = 0.1 * (dt1**2 + 0.5 * dt2**2)

        # 4. Terminal bonus (stronger than baseline)
        bonus = 0.0
        if done and h_next > 1.0:
            bonus = 200.0  # Larger bonus for success

        return base_reward + shaping_reward - vel_penalty + bonus
