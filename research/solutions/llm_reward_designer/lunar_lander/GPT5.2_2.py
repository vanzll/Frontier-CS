import math

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:

        x, y, vx, vy, theta, vtheta, _, _ = next_state

        # guidance: move toward (0,0)
        vx_star = -1.2 * x
        vy_star = -1.0 * y  # descend rate proportional to height (slower near ground)

        # tracking errors
        evx = vx - vx_star
        evy = vy - vy_star

        # base shaping (keep per-step magnitude moderate)
        reward = 0.2
        reward -= 0.6 * (evx*evx + evy*evy)
        reward -= 1.5 * (theta*theta) - 0.2 * (vtheta*vtheta)

        # landing phase tightening
        if y < 0.3:
            reward -= 2.0 * (vx*vx + vy*vy)
            reward -= 4.0 * abs(theta)

        # fuel cost (discourage always-thrust)
        if action == 2:
            reward -= 0.02
        elif action == 1 or action == 3:
            reward -= 0.004

        # terminal
        if done:
            landed = (y <= 0.0 and abs(x) < 0.2 and abs(theta) < 0.2 and abs(vy) < 0.5 and abs(vx) < 0.5)
            reward += 90.0 if landed else -90.0

        return float(reward)