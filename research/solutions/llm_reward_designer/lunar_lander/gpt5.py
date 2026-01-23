import math


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gpt5_lunar_0

        PBRS on a hand-crafted landing potential (distance + speed + angle) + fuel cost.
        """

        def phi(s):
            x, y, vx, vy, theta, vtheta, _, _ = s
            return -(
                2.0 * abs(x)
                + 1.5 * max(0.0, y)
                + 0.7 * abs(vx)
                + 0.7 * abs(vy)
                + 2.0 * abs(theta)
                + 0.3 * abs(vtheta)
            )

        gamma = 0.99
        shaping = gamma * phi(next_state) - phi(state)
        reward = 0.2 + 5.0 * shaping

        if action == 2:
            reward -= 0.02
        elif action == 1 or action == 3:
            reward -= 0.004

        x, y, vx, vy, theta, _, _, _ = next_state
        if done:
            landed = (y <= 0.0 and abs(x) < 0.2 and abs(theta) < 0.2 and abs(vy) < 0.5 and abs(vx) < 0.5)
            crashed = (y <= 0.0 and not landed) or (abs(x) > 1.5 or y > 2.5)
            if landed:
                reward += 80.0
            elif crashed:
                reward -= 80.0

        return float(reward)


