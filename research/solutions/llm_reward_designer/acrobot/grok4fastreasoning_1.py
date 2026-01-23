import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Grok4FastReasoning Solution 1: Energy-Based Multi-Stage Learning for Acrobot

        State: [cos(t1), sin(t1), cos(t2), sin(t2), t1_dot, t2_dot]

        Strategy: Two-phase reward design
        - Phase 1 (low height): Reward swing energy to build momentum
        - Phase 2 (high height): Focus on height stabilization

        This creates a natural curriculum from swinging to balancing.
        """

        def get_height(s):
            cos_t1, sin_t1, cos_t2, sin_t2 = s[0], s[1], s[2], s[3]
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            return -cos_t1 - cos_t1_plus_t2

        h_current = get_height(state)
        h_next = get_height(next_state)

        # Extract velocities
        _, _, _, _, dt1, dt2 = next_state
        kinetic_energy = dt1**2 + dt2**2

        # Base time penalty
        reward = -1.0

        # Multi-stage reward design
        if h_next < 0.5:
            # Phase 1: Build swinging energy when low
            # Reward height progress + kinetic energy to encourage swinging
            height_progress = (h_next - h_current) * 8.0
            energy_bonus = kinetic_energy * 0.05
            reward += height_progress + energy_bonus

            # Slight penalty for zero torque (encourage action)
            if action == 1:  # Zero torque
                reward -= 0.1

        else:
            # Phase 2: Focus on height when high
            # Strong height shaping + velocity control for stability
            height_progress = (h_next - h_current) * 12.0
            vel_penalty = kinetic_energy * 0.1  # Damp oscillations
            reward += height_progress - vel_penalty

        # Terminal bonus
        if done and h_next > 1.0:
            reward += 150.0

        return reward
