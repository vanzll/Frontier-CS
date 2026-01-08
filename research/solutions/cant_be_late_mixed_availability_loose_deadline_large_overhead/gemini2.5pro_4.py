from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    # --- Tunable Parameters ---

    # Safety buffer factor: We enter "panic mode" (always ON_DEMAND) if the
    # time left to deadline is less than (work_remaining + safety_buffer).
    # The buffer is calculated as: self.restart_overhead * SAFETY_BUFFER_FACTOR.
    # A value of 1.5 means we reserve enough slack to handle 1.5 future
    # preemption events without missing the deadline.
    SAFETY_BUFFER_FACTOR = 1.5

    # Wait threshold factor: When Spot is unavailable, we choose to wait (NONE)
    # only if our current slack is greater than a threshold.
    # The threshold is: self.restart_overhead * WAIT_THRESHOLD_FACTOR.
    # This ensures we only wait if we have enough slack to absorb at least
    # one future preemption plus a margin.
    WAIT_THRESHOLD_FACTOR = 1.2

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        No trace-specific initialization is needed for this heuristic.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Makes a decision at each time step on which cluster type to use.
        """
        # 1. Calculate current progress and time remaining.
        # The sum() operation is efficient enough for the expected scale of the
        # problem and keeps the logic stateless between steps.
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        # If the job is already finished, do nothing to save costs.
        if work_remaining <= 0:
            return ClusterType.NONE

        current_time = self.env.elapsed_seconds
        time_to_deadline = self.deadline - current_time

        # 2. Check if we are in "Panic Mode".
        # This is the critical condition where we must use the reliable
        # On-Demand option to guarantee finishing before the deadline.
        safety_buffer = self.restart_overhead * self.SAFETY_BUFFER_FACTOR
        critical_time_needed = work_remaining + safety_buffer

        if time_to_deadline <= critical_time_needed:
            # Not enough slack to risk using Spot or waiting.
            return ClusterType.ON_DEMAND

        # 3. If not in panic mode, operate in "Standard Mode".
        if has_spot:
            # Spot is available and we have a time buffer, so use the cheap option.
            return ClusterType.SPOT
        else:
            # Spot is not available. Decide between waiting (NONE) or using
            # On-Demand to make progress.
            slack = time_to_deadline - work_remaining
            wait_threshold = self.restart_overhead * self.WAIT_THRESHOLD_FACTOR

            if slack > wait_threshold:
                # We have a comfortable amount of slack. It's better to wait
                # for Spot to potentially become available again than to pay
                # for On-Demand.
                return ClusterType.NONE
            else:
                # Slack is getting tight. The risk of falling behind by waiting
                # is too high. Use On-Demand to guarantee progress.
                return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        """Required method for evaluator instantiation."""
        # No custom command-line arguments are needed for this strategy.
        args, _ = parser.parse_known_args()
        return cls(args)