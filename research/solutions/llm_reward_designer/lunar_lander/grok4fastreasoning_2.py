import math

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Grok4FastReasoning Solution 2: Vector Field Guidance for LunarLander

        State: [x, y, vx, vy, theta, vtheta, l_contact, r_contact]

        Strategy: Define desired velocity vector pointing to landing pad
        Reward based on alignment between current velocity and desired velocity

        This creates a natural flow field that guides the lander to the target.
        """

        x, y, vx, vy, theta, vtheta, _, _ = next_state

        # 1. Calculate desired velocity vector
        # Target position: (0, 0)
        # Desired velocity should point towards target with appropriate speed
        dist_to_target = math.sqrt(x*x + y*y)

        if dist_to_target > 0.01:  # Avoid division by zero
            # Desired direction (normalized)
            desired_vx = -x / dist_to_target
            desired_vy = -y / dist_to_target

            # Desired speed (slower when closer to target)
            desired_speed = min(1.0, dist_to_target * 0.5)
            desired_vx *= desired_speed
            desired_vy *= desired_speed

            # Special handling near ground: reduce downward component
            if y < 0.2:
                desired_vy = max(desired_vy, -0.2)  # Don't want to go down too fast near ground

            # 2. Velocity alignment reward
            current_speed = math.sqrt(vx*vx + vy*vy)
            if current_speed > 0.01:  # Current velocity exists
                # Cosine similarity between current and desired velocity
                dot_product = vx * desired_vx + vy * desired_vy
                alignment = dot_product / (current_speed * desired_speed)
                vel_reward = alignment * 2.0  # Scale up
            else:
                vel_reward = 0.0

            # 3. Speed matching bonus
            speed_error = abs(current_speed - desired_speed)
            speed_reward = -speed_error * 1.0

        else:
            # Very close to target - focus on stopping
            vel_reward = -math.sqrt(vx*vx + vy*vy) * 3.0
            speed_reward = 0.0

        # 4. Orientation control (should face movement direction roughly)
        # Ideal theta should align with velocity direction
        if abs(vx) > 0.1 or abs(vy) > 0.1:
            desired_theta = math.atan2(vy, vx)  # Direction of movement
            theta_error = abs(theta - desired_theta)
            # Normalize to [-pi, pi]
            theta_error = min(theta_error, 2*math.pi - theta_error)
            orientation_reward = -theta_error * 1.0
        else:
            orientation_reward = -abs(theta) * 1.0  # Just keep upright if not moving

        # 5. Angular velocity damping
        ang_vel_penalty = -abs(vtheta) * 0.5

        # Combine rewards
        reward = vel_reward + speed_reward + orientation_reward + ang_vel_penalty

        # 6. Fuel costs
        fuel_cost = 0.0
        if action == 2:
            fuel_cost = 0.3
        elif action == 1 or action == 3:
            fuel_cost = 0.03

        # 7. Terminal rewards
        if done:
            if y <= 0.0:
                if (abs(x) < 0.2 and abs(theta) < 0.2 and
                    abs(vy) < 0.5 and abs(vx) < 0.5):
                    reward += 100.0
                else:
                    reward += -100.0
            elif abs(x) > 1.5 or y > 2.5:
                reward += -100.0

        return reward - fuel_cost
