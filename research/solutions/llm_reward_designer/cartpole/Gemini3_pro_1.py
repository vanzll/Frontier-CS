import math

class Solution:
    def reward_function(self, state, action, next_state, done) -> float:
        """
        Gemini3 Pro Designed Reward Function for CartPole-v1.
        
        State layout:
        0: x (Cart Position)
        1: x_dot (Cart Velocity)
        2: theta (Pole Angle)
        3: theta_dot (Pole Angular Velocity)
        """
        
        x, x_dot, theta, theta_dot = next_state
        
        # 1. Base Survival Reward (Alignment with Ground Truth)
        # We want to encourage the agent to stay alive as long as possible.
        reward = 1.0
        
        # 2. Position Shaping (Encourage Centering)
        # Being near the center (x=0) is safer than near the edges (x=+-2.4).
        # We use a quadratic penalty normalized by the threshold.
        x_threshold = 2.4
        dist_penalty = (x / x_threshold) ** 2
        
        # 3. Angle Shaping (Encourage Verticality)
        # Being upright (theta=0) is much safer.
        # Threshold is roughly 12 degrees (~0.209 rad).
        theta_threshold = 0.2095
        angle_penalty = (theta / theta_threshold) ** 2
        
        # 4. Velocity Shaping (Encourage Stability)
        # High velocity usually means instability or impending crash.
        # We penalize high angular velocity heavily.
        velocity_penalty = (theta_dot ** 2) * 0.1 + (x_dot ** 2) * 0.01
        
        # Combine penalties
        # We subtract these from the base reward.
        # Weights are tuned to not overpower the survival signal but provide strong guidance.
        reward -= dist_penalty * 0.5
        reward -= angle_penalty * 0.5
        reward -= velocity_penalty
        
        # 5. Terminal Penalty
        # If the episode ends early (failure), apply a penalty.
        # Note: We don't have 'step count' here, so we just penalize 'done' state
        # unless it's a timeout (which PPO handles via time limit usually).
        # But simply penalizing 'done' is risky if max_steps is reached.
        # However, in CartPole, 'done' usually means failure (fell or out of bounds).
        if done:
            # Check if it was likely a failure (angle or position violation)
            # The environment thresholds are hardcoded, but we can infer.
            if abs(x) >= x_threshold or abs(theta) >= theta_threshold:
                reward -= 10.0
                
        return reward