import math

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:

        x, y, vx, vy, theta, vtheta, _, _ = next_state
        dist = math.sqrt(x*x + y*y)
        speed = math.sqrt(vx*vx + vy*vy)

        reward = 0.2

        if y > 0.6:
            # high altitude: get roughly above the pad, keep upright, don't spin too much
            reward -= 1.2 * abs(x)
            reward -= 0.6 * dist
            reward -= 2.5 * abs(theta)
            reward -= 0.2 * abs(vtheta)
            # allow speed a bit (don’t over-penalize), but prevent extreme falling
            if vy < -2.0:
                reward -= 0.5
        else:
            # low altitude: behave like a landing controller
            reward -= 3.0 * abs(x)
            reward -= 2.0 * abs(theta)
            reward -= 1.5 * abs(vx)
            reward -= 2.0 * max(0.0, -vy)  # punish downward speed strongly
            reward -= 0.2 * abs(vtheta)

            # extra dense bonus when already satisfying landing thresholds (even before done)
            if abs(x) < 0.2 and abs(theta) < 0.2 and abs(vy) < 0.5 and abs(vx) < 0.5:
                reward += 1.0

        # fuel cost
        if action == 2:
            reward -= 0.015
        elif action == 1 or action == 3:
            reward -= 0.003

        # terminal
        if done:
            landed = (y <= 0.0 and abs(x) < 0.2 and abs(theta) < 0.2 and abs(vy) < 0.5 and abs(vx) < 0.5)
            reward += 95.0 if landed else -95.0

        return float(reward)