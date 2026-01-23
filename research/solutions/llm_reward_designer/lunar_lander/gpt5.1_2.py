import math


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gpt5.1_lunar_2

        Simple two-phase shaping:
          - High altitude: move above pad, keep roughly upright.
          - Low altitude: behave like a precise landing controller.
        """

        x, y, vx, vy, theta, vtheta, _, _ = next_state

        reward = 0.2

        dist = math.sqrt(x * x + y * y)

        if y > 0.6:
            # high altitude: primarily navigate over pad and avoid wild attitudes
            reward -= 1.5 * abs(x)
            reward -= 0.6 * dist
            reward -= 2.5 * abs(theta)
            reward -= 0.2 * abs(vtheta)

            # discourage extreme downward speed far from surface
            if vy < -2.0:
                reward -= 0.6
        else:
            # low altitude: landing controller
            reward -= 3.0 * abs(x)
            reward -= 2.0 * abs(theta)
            reward -= 1.5 * abs(vx)
            reward -= 2.5 * max(0.0, -vy)  # strong on downward speed
            reward -= 0.25 * abs(vtheta)

            # dense local bonus when we are already in a safe landing region
            if (
                abs(x) < 0.2
                and abs(theta) < 0.2
                and abs(vy) < 0.5
                and abs(vx) < 0.5
            ):
                reward += 1.2

        # fuel usage penalty
        if action == 2:
            reward -= 0.02
        elif action == 1 or action == 3:
            reward -= 0.004

        # terminal bonus/penalty
        if done:
            landed = (
                y <= 0.0
                and abs(x) < 0.2
                and abs(theta) < 0.2
                and abs(vy) < 0.5
                and abs(vx) < 0.5
            )
            crashed = (y <= 0.0 and not landed) or (abs(x) > 1.5 or y > 2.5)

            if landed:
                reward += 95.0
            elif crashed:
                reward -= 95.0

        return float(reward)


