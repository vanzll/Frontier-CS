#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
import numpy as np
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
COMMON_DIR = HERE / "../common"
RESOURCES_DIR = HERE / "resources"
sys.path.insert(0, str(COMMON_DIR))
sys.path.insert(0, str(RESOURCES_DIR))

from cleanrl_ppo import train, Args
# Import the custom environment wrapper (actually CleanRL PPO will use gym.make, 
# so we need to register our custom env or patch make_env)

# Wait, cleanrl_ppo.py uses `gym.make(env_id)`.
# Since we have a custom python class `LunarLanderEnv` in environment.py,
# we need to adapt cleanrl_ppo.py or our usage of it to accept a custom env class.
# CleanRL's make_env function calls `gym.make`.

# Solution: We will Monkey Patch gym.make or create a custom wrapper in this script
# that intercepts the env_id "LunarLander-Custom".

import gymnasium as gym
from environment import LunarLanderEnv

class CustomEnvWrapper(gym.Env):
    def __init__(self):
        self.env = LunarLanderEnv()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.env.observation_space_shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.env.action_space_n)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        return self.env.reset(), {}
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

# Register the custom environment
gym.register(
    id="LunarLander-Custom-v0",
    entry_point=CustomEnvWrapper,
)

def load_solution_module(solution_path: Path) -> ModuleType:
    if not solution_path.exists():
        raise FileNotFoundError(f"solution.py not found at {solution_path}")
    spec = importlib.util.spec_from_file_location("submitted_solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def evaluate(solution_path: Path) -> dict:
    try:
        module = load_solution_module(solution_path)
        solution = module.Solution()
        
        args = Args(
            env_id="LunarLander-Custom-v0",
            total_timesteps=300000, # Needs more time
            learning_rate=2.5e-4,
            num_envs=4,
            num_steps=128,
            update_epochs=4,
            batch_size=512,
            cuda=False
        )
        
        score = train(args, reward_fn=solution.reward_function)
        
        # Max score: ~100 per episode (if perfect). Min: -100 or worse.
        # Normalize: 0 -> 0, 100 -> 100.
        # Score = max(0, score) ? No, let's map [-100, 100] to [0, 100]
        # (score + 100) / 2
        normalized_score = max(0, min(100, (score + 100) / 2.0))
        
        return {
            "status": "success",
            "score": float(normalized_score),
            "score_unbounded": float(score),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "score": 0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution-path", type=Path, default=Path("solution.py"))
    parser.add_argument("--output-path", type=Path, default=Path("result.json"))
    args = parser.parse_args()
    
    result = evaluate(args.solution_path)
    
    with args.output_path.open("w") as f:
        json.dump(result, f, indent=2)
        
    if result["status"] == "success":
        print(f"Evaluation completed. Score: {result['score']:.2f}")
        print(f"{result['score']:.2f} {result['score_unbounded']:.2f}")
    else:
        print(f"Error: {result.get('error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()

