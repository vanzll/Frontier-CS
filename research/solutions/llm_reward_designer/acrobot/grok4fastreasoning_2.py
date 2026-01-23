import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Grok4FastReasoning Solution 2: Lyapunov-Inspired Control for Acrobot

        State: [cos(t1), sin(t1), cos(t2), sin(t2), t1_dot, t2_dot]

        Strategy: Design a Lyapunov function V that decreases towards the goal
        V(s) = potential + kinetic energy terms
        Reward = -dV/dt (energy dissipation) + terminal bonus

        This creates a stable control-like reward that naturally guides to the upright position.
        """

        def get_height(s):
            cos_t1, sin_t1, cos_t2, sin_t2 = s[0], s[1], s[2], s[3]
            cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
            return -cos_t1 - cos_t1_plus_t2

        def lyapunov_function(s):
            """Lyapunov candidate: higher when far from goal"""
            h = get_height(s)
            cos_t1, _, _, _, dt1, dt2 = s

            # Potential energy (distance from target height 2.0)
            potential = (2.0 - h)**2 * 5.0

            # Kinetic energy (penalize high velocities)
            kinetic = dt1**2 * 0.5 + dt2**2 * 0.2

            # Angle deviation (penalize when not upright)
            angle_penalty = (1.0 + cos_t1)**2 * 2.0  # cos(t1) = -1 when upright

            return potential + kinetic + angle_penalty

        V_current = lyapunov_function(state)
        V_next = lyapunov_function(next_state)

        # Lyapunov-based reward: negative derivative encourages stability
        lyapunov_reward = -(V_next - V_current) * 2.0

        # Base time penalty
        base_reward = -1.0

        # Action cost (small penalty for non-zero torque to encourage efficiency)
        action_cost = 0.0
        if action != 1:  # Non-zero torque
            action_cost = 0.05

        # Terminal bonus
        bonus = 0.0
        h_next = get_height(next_state)
        if done and h_next > 1.0:
            bonus = 180.0

        return base_reward + lyapunov_reward - action_cost + bonus
