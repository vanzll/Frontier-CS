import math


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gpt5_lunar_1

        Velocity-field tracking towards pad; penalize vx/vy deviation and angle.
        """

        x, y, vx, vy, theta, vtheta, _, _ = next_state

        kx = 1.0
        ky = 0.8
        target_vx = -kx * x
        target_vy = -ky * y

        evx = vx - target_vx
        evy = vy - target_vy

        reward = 0.2
        reward -= 0.8 * (evx * evx + evy * evy)
        reward -= 3.0 * (theta * theta)
        reward -= 0.5 * (vtheta * vtheta)

        if y < 0.3:
            reward -= 2.0 * (vx * vx + vy * vy)
            reward -= 5.0 * abs(theta)

        if action == 2:
            reward -= 0.025
        elif action == 1 or action == 3:
            reward -= 0.005

        if done:
            landed = (y <= 0.0 and abs(x) < 0.2 and abs(theta) < 0.2 and abs(vy) < 0.5 and abs(vx) < 0.5)
            crashed = (y <= 0.0 and not landed) or (abs(x) > 1.5 or y > 2.5)
            if landed:
                reward += 90.0
            elif crashed:
                reward -= 90.0

        return float(reward)


