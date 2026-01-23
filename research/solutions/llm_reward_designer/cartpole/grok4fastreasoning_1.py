import numpy as np

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Grok4FastReasoning Solution 1: Energy-Based Control for CartPole

        State: [x, x_dot, theta, theta_dot]

        Strategy: Lyapunov function based on total system energy
        V = potential_energy + kinetic_energy + position_penalty

        Reward = -dV/dt to encourage energy dissipation towards equilibrium
        """

        x, x_dot, theta, theta_dot = next_state

        # Thresholds
        x_threshold = 2.4
        theta_threshold = 12 * 2 * np.pi / 360

        def lyapunov_energy(s):
            """Lyapunov candidate: energy-like function"""
            x_pos, x_vel, theta_ang, theta_vel = s

            # Potential energy (angle deviation + position)
            potential = (
                (theta_ang / theta_threshold) ** 2 * 10.0 +  # Angle cost
                (x_pos / x_threshold) ** 2 * 2.0              # Position cost
            )

            # Kinetic energy
            kinetic = x_vel**2 * 0.5 + theta_vel**2 * 2.0

            return potential + kinetic

        # Energy-based reward (negative energy change)
        V_current = lyapunov_energy(state)
        V_next = lyapunov_energy(next_state)
        energy_reward = -(V_next - V_current) * 3.0

        # Base survival reward
        base_reward = 1.0

        # Action efficiency bonus (reward using appropriate force)
        action_bonus = 0.0
        if abs(x) > 0.5:  # Cart is off-center, reward corrective action
            if (x > 0 and action == 0) or (x < 0 and action == 1):  # Correct direction
                action_bonus = 0.1

        # Terminal penalty
        if done:
            if abs(x) >= x_threshold or abs(theta) >= theta_threshold:
                base_reward -= 15.0

        return base_reward + energy_reward + action_bonus
