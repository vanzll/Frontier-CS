#!/usr/bin/env python3
"""
Evaluator for Agentic RL Algorithm Discovery problem.

This evaluator:
1. Loads the user's Solution class
2. Runs verl-agent training with the custom algorithm
3. Evaluates the trained agent on test tasks
4. Computes and outputs the score
"""
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path


def load_solution(solution_path: str, resources_dir: Path):
    """Load and initialize the user's Solution class."""
    # Add resources directory to path for solution_loader
    sys.path.insert(0, str(resources_dir))

    from solution_loader import load_solution as _load
    return _load(solution_path)


def run_training(solution_path: str, resources_dir: Path) -> dict:
    """
    Run verl-agent training with the user's Solution.

    Returns:
        Training results including final success rate
    """
    verl_agent_dir = resources_dir / "verl-agent"

    # Set up environment
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{verl_agent_dir}:{resources_dir}:{env.get('PYTHONPATH', '')}"
    env["SOLUTION_PATH"] = str(solution_path)

    # Run training script
    training_script = resources_dir / "run_training.sh"
    if training_script.exists():
        result = subprocess.run(
            ["bash", str(training_script)],
            cwd=str(verl_agent_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=21000  # ~5.8 hours, leave buffer for evaluation
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"Training failed: {result.stderr}", file=sys.stderr)
            return {"success_rate": 0.0, "error": result.stderr}
    else:
        # Fallback: run verl-agent directly
        print("Warning: run_training.sh not found, using direct training")
        # This would require more setup - for now, return error
        return {"success_rate": 0.0, "error": "run_training.sh not found"}

    # Parse training results
    results_file = verl_agent_dir / "results" / "final_results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)

    # Try to parse from stdout
    success_rate = parse_success_rate_from_output(result.stdout)
    return {"success_rate": success_rate}


def parse_success_rate_from_output(output: str) -> float:
    """Parse success rate from training output."""
    # Look for patterns like "success_rate: 0.86" or "Success Rate: 86%"
    import re

    patterns = [
        r"success[_\s]rate[:\s]+(\d+\.?\d*)",
        r"Success[_\s]Rate[:\s]+(\d+\.?\d*)%?",
        r"val/success[:\s]+(\d+\.?\d*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            rate = float(match.group(1))
            if rate > 1:  # It's a percentage
                rate /= 100
            return rate

    return 0.0


def compute_score(success_rate: float, epochs_to_converge: int = 150) -> tuple:
    """
    Compute bounded and unbounded scores.

    Args:
        success_rate: Success rate on test tasks [0, 1]
        epochs_to_converge: Epoch when 95% of final performance was reached

    Returns:
        (bounded_score, unbounded_score)
    """
    # Bounded score: 0-100
    bounded_score = success_rate * 100

    # Unbounded score: includes efficiency bonus
    total_epochs = 150
    efficiency_bonus = max(0, (1 - epochs_to_converge / total_epochs) * 20)
    unbounded_score = bounded_score + efficiency_bonus

    return bounded_score, unbounded_score


def main(resources_dir: Path = None):
    """Main evaluator entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", type=str, default="solution.py",
                        help="Path to solution file")
    args, _ = parser.parse_known_args()

    # Determine paths
    if resources_dir is None:
        resources_dir = Path(__file__).parent / "resources"

    solution_path = Path(args.solution).resolve()
    if not solution_path.exists():
        # Try relative to workspace
        solution_path = Path("/workspace") / args.solution
        if not solution_path.exists():
            print(f"Error: Solution file not found: {args.solution}", file=sys.stderr)
            print("0")  # Output zero score
            return

    print(f"Loading solution from: {solution_path}")

    try:
        # Load and validate solution
        solution = load_solution(str(solution_path), resources_dir)
        print(f"Solution loaded successfully")
        print(f"Config: {getattr(solution, 'config', {})}")

        # Run training and evaluation
        results = run_training(str(solution_path), resources_dir)
        success_rate = results.get("success_rate", 0.0)
        epochs_to_converge = results.get("epochs_to_converge", 150)

        # Compute scores
        bounded_score, unbounded_score = compute_score(success_rate, epochs_to_converge)

        # Output scores (format expected by Frontier-CS)
        print(f"\n=== Evaluation Results ===")
        print(f"Success Rate: {success_rate:.4f}")
        print(f"Bounded Score: {bounded_score:.2f}")
        print(f"Unbounded Score: {unbounded_score:.2f}")
        print(f"\n{bounded_score:.2f} {unbounded_score:.2f}")

    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        print("0 0")  # Output zero scores


if __name__ == "__main__":
    main()
