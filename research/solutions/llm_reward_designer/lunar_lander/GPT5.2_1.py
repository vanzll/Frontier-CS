import math

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        GPT-5.2 LunarLander #1: PBRS on a shaped potential (distance + speed + angle) + terminal bonus.
        通常对 PPO 最稳，探索更容易收敛到“往目标靠近+减速+竖直”。
        """
        def phi(s):
            x, y, vx, vy, theta, vtheta, _, _ = s
            dist = math.sqrt(x*x + y*y)
            speed = math.sqrt(vx*vx + vy*vy)
            # potential: higher is better
            return -(2.0*dist + 0.6*speed + 2.0*abs(theta) + 0.2*abs(vtheta))

        gamma = 0.99
        shaping = gamma * phi(next_state) - phi(state)

        # small survival drip so longer episodes help
        reward = 0.2 + 5.0 * shaping

        # fuel cost (match env magnitudes; very small per step)
        if action == 2:
            reward -= 0.01
        elif action == 1 or action == 3:
            reward -= 0.002

        # terminal bonus/penalty (reconstruct landing condition)
        x, y, vx, vy, theta, _, _, _ = next_state
        if done:
            landed = (y <= 0.0 and abs(x) < 0.2 and abs(theta) < 0.2 and abs(vy) < 0.5 and abs(vx) < 0.5)
            if landed:
                reward += 80.0
            else:
                reward -= 80.0

        return float(reward)