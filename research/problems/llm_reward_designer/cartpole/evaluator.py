#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
import numpy as np
from pathlib import Path
from types import ModuleType

# Add common resources to path
HERE = Path(__file__).resolve().parent
COMMON_DIR = HERE / "../common"
sys.path.insert(0, str(COMMON_DIR))

# Import modified CleanRL PPO
from cleanrl_ppo import train, Args

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
        
        # Setup Args for CartPole-v1
        # Using default CleanRL hyperparameters for CartPole
        args = Args(
            env_id="CartPole-v1",
            total_timesteps=200000, # CleanRL usually does 500k
            learning_rate=2.5e-4,
            num_envs=4,
            num_steps=128,
            update_epochs=4,
            batch_size=512, # 4 * 128
            cuda=False # Force CPU for stability in docker
        )
        
        # Train and get score
        score = train(args, reward_fn=solution.reward_function)
        
        # Max score on CartPole is 500.
        # Normalize: 475+ is solved. Cap at 500.
        final_score = min(score, 500.0)
        normalized_score = (final_score / 500.0) * 100.0
        
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
