"""
Dynamic loader for user Solution class.
This module loads and manages user-defined RL algorithm implementations.

Place at: verl/trainer/ppo/solution_loader.py (or import from resources/)
"""
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Any


_solution_instance: Optional[Any] = None


def load_solution(solution_path: str) -> Any:
    """
    Dynamically load a user Solution class from a Python file.

    Args:
        solution_path: Path to the Python file containing the Solution class

    Returns:
        Initialized Solution instance (after calling solve())

    Raises:
        FileNotFoundError: If solution file doesn't exist
        AttributeError: If Solution class is not found in the file
        Exception: If solve() method fails
    """
    global _solution_instance

    solution_path = Path(solution_path).resolve()
    if not solution_path.exists():
        raise FileNotFoundError(f"Solution file not found: {solution_path}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("user_solution", solution_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec from {solution_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["user_solution"] = module
    spec.loader.exec_module(module)

    # Get the Solution class
    if not hasattr(module, "Solution"):
        raise AttributeError(f"Solution class not found in {solution_path}")

    solution_class = module.Solution

    # Instantiate and initialize
    _solution_instance = solution_class()
    _solution_instance.solve()

    # Log the loaded configuration
    config = getattr(_solution_instance, "config", {})
    print(f"[SolutionLoader] Loaded Solution from {solution_path}")
    print(f"[SolutionLoader] Solution config: {config}")

    return _solution_instance


def get_solution() -> Optional[Any]:
    """
    Get the currently loaded Solution instance.

    Returns:
        The Solution instance, or None if not loaded
    """
    return _solution_instance


def reset_solution() -> None:
    """Reset the solution instance (for testing purposes)."""
    global _solution_instance
    _solution_instance = None


class SolutionInterface:
    """
    Base class defining the Solution interface.
    Users should implement this interface in their solution files.
    """

    def solve(self, spec_path: str = None) -> "SolutionInterface":
        """
        Initialize the algorithm. Called once before training.

        Set hyperparameters in self.config:
            self.config = {
                "gamma": 0.95,
                "clip_ratio": 0.2,
                "use_kl_loss": False,
                "kl_loss_coef": 0.01,
                "entropy_coef": 0.0,
            }

        Returns:
            self
        """
        self.config = {}
        return self

    def compute_advantage(
        self,
        token_level_rewards,
        response_mask,
        episode_index,
        trajectory_index,
        step_rewards=None,
        anchor_observations=None,
        gamma=0.95,
        **kwargs
    ):
        """
        Compute advantages and returns for policy update.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
        """
        raise NotImplementedError("compute_advantage must be implemented")

    def compute_policy_loss(
        self,
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        clip_ratio=0.2,
        **kwargs
    ):
        """
        Compute policy gradient loss.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: (loss, metrics)
        """
        raise NotImplementedError("compute_policy_loss must be implemented")

    def assign_step_rewards(
        self,
        episode_reward,
        trajectory_length,
        step_observations,
        step_actions,
        **kwargs
    ):
        """
        Distribute episode reward to individual steps.

        Returns:
            np.ndarray: step_rewards of shape (trajectory_length,)
        """
        raise NotImplementedError("assign_step_rewards must be implemented")
