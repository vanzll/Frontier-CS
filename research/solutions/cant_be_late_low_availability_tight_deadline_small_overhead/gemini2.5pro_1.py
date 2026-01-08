import argparse

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "heuristic_ema_v1"

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        # --- Hyperparameters ---
        # Alpha for the Exponential Moving Average of spot availability probability.
        self.ema_alpha: float = 0.01
        # Initial guess for spot availability. Problem states 4-40%.
        self.p_spot_initial_guess: float = 0.20
        # A factor to make the safety buffer more conservative. > 1.0 is safer.
        self.safety_buffer_factor: float = 1.1

        # --- Internal State ---
        self.p_spot_ema: float = self.p_spot_initial_guess

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        Must return self.
        """
        self.p_spot_ema = self.p_spot_initial_guess
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step. Return which cluster type to use next.
        """
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        # If the task is finished, do nothing.
        if work_remaining <= 1e-9:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds

        # If deadline has passed, try to make progress (though it's likely too late).
        if time_to_deadline <= 0:
            return ClusterType.ON_DEMAND

        # Update our estimate of spot availability using an EMA.
        current_spot_availability = 1.0 if has_spot else 0.0
        self.p_spot_ema = (self.ema_alpha * current_spot_availability +
                           (1 - self.ema_alpha) * self.p_spot_ema)

        # Define a "panic threshold" to decide when to switch to ON_DEMAND.
        # This is the point where if a SPOT attempt failed, we wouldn't have
        # enough time to finish the job even on guaranteed on-demand instances.
        time_cost_of_spot_failure = self.env.gap_seconds + self.restart_overhead
        safety_buffer = self.safety_buffer_factor * time_cost_of_spot_failure
        panic_threshold = time_to_deadline - safety_buffer

        # If remaining work exceeds the panic threshold, we must use a guaranteed resource.
        if work_remaining >= panic_threshold:
            return ClusterType.ON_DEMAND

        # If we are in "safe mode" (i.e., not in panic mode):
        if has_spot:
            # Spot is available and cheap; it's the best choice.
            return ClusterType.SPOT
        else:
            # Spot is not available. Decide between waiting (NONE) or
            # buying guaranteed progress (ON_DEMAND).

            # Calculate the time we can afford to wait before hitting the panic threshold.
            max_waitable_time = panic_threshold - work_remaining
            
            # Calculate the expected time we would have to wait for spot to become available.
            p_spot = max(self.p_spot_ema, 1e-6)  # Avoid division by zero
            expected_wait_time = ((1 / p_spot) - 1) * self.env.gap_seconds

            # CORE HEURISTIC: Compare what we can afford with what we expect.
            if max_waitable_time > expected_wait_time:
                # We can afford to wait, so we do to save money.
                return ClusterType.NONE
            else:
                # The expected wait is too long. The risk of hitting the panic
                # threshold is too high, so we use ON_DEMAND to make progress.
                return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        """
        A required classmethod for the evaluator to instantiate the solution.
        """
        args, _ = parser.parse_known_args()
        return cls(args)