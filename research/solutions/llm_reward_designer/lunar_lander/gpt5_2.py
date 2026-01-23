import math


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gpt5_lunar_2

        Two-phase shaping: high-altitude guidance then precise low-altitude landing control.
        """

        x, y, vx, vy, theta, vtheta, _, _ = next_state

        reward = 0.2
        dist = math.sqrt(x * x + y * y)

        if y > 0.6:
            reward -= 1.5 * abs(x)
            reward -= 0.6 * dist
            reward -= 2.5 * abs(theta)
            reward -= 0.2 * abs(vtheta)
            if vy < -2.0:
                reward -= 0.6
        else:
            reward -= 3.0 * abs(x)
            reward -= 2.0 * abs(theta)
            reward -= 1.5 * abs(vx)
            reward -= 2.5 * max(0.0, -vy)
            reward -= 0.25 * abs(vtheta)
            if abs(x) < 0.2 and abs(theta) < 0.2 and abs(vy) < 0.5 and abs(vx) < 0.5:
                reward += 1.2

        if action == 2:
            reward -= 0.02
        elif action == 1 or action == 3:
            reward -= 0.004

        if done:
            landed = (y <= 0.0 and abs(x) < 0.2 and abs(theta) < 0.2 and abs(vy) < 0.5 and abs(vx) < 0.5)
            crashed = (y <= 0.0 and not landed) or (abs(x) > 1.5 or y > 2.5)
            if landed:
                reward += 95.0
            elif crashed:
                reward -= 95.0

        return float(reward)


