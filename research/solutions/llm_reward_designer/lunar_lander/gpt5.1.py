import math


class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        gpt5.1_lunar_0

        Potential-based shaping on a simple landing potential:
            Phi(s) = -(w1*|x| + w2*y + w3*|vx| + w4*|vy| + w5*|theta| + w6*|vtheta|)

        Ground-truth evaluation uses only +100/-100 and fuel; this shapes a dense
        curriculum toward (x,y)=(0,0), small velocities, and upright attitude.
        """

        def phi(s):
            x, y, vx, vy, theta, vtheta, _, _ = s

            # weights chosen so typical per-step shaping magnitude is O(1)
            return -(
                2.0 * abs(x)
                + 1.5 * max(0.0, y)  # distance above ground
                + 0.7 * abs(vx)
                + 0.7 * abs(vy)
                + 2.0 * abs(theta)
                + 0.3 * abs(vtheta)
            )

        gamma = 0.99
        shaping = gamma * phi(next_state) - phi(state)

        # small survival bonus + shaping
        reward = 0.2 + 5.0 * shaping

        # simple fuel cost aligned with environment semantics
        if action == 2:
            reward -= 0.02
        elif action == 1 or action == 3:
            reward -= 0.004

        # reconstruct terminal success / failure consistent with environment.py
        x, y, vx, vy, theta, _, _, _ = next_state
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
                reward += 80.0
            elif crashed:
                reward -= 80.0

        return float(reward)


