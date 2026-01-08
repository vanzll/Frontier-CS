import argparse

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initializes the strategy with tunable parameters.
        """
        # Safety buffer: A portion of the total slack reserved as a hard safety
        # margin. An 8-hour buffer (out of 22) is a conservative choice
        # given the high penalty for failure and low-availability regions.
        self.safety_buffer_hours = 8.0

        # Panic buffer factor: A multiplier for the restart overhead to create
        # a time buffer in the final moments to absorb a last-minute
        # preemption without failing the deadline.
        self.panic_buffer_factor = 2.0

        # Flag to perform lazy initialization on the first step.
        self.initialized = False
        return self

    def _initialize_params(self):
        """
        Initializes parameters that depend on environment attributes.
        """
        self.safety_buffer = self.safety_buffer_hours * 3600.0
        self.target_finish_time = self.deadline - self.safety_buffer
        self.panic_buffer = self.panic_buffer_factor * self.restart_overhead
        # Epsilon for robust floating-point comparisons.
        self.epsilon = 1e-9
        self.initialized = True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        The main decision-making logic, called at each time step.
        """
        if not self.initialized:
            self._initialize_params()

        current_time = self.env.elapsed_seconds
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        # 1. Job completed: Switch to NONE to stop incurring costs.
        if work_remaining <= self.epsilon:
            return ClusterType.NONE

        # 2. Panic Mode: Activate if critically close to the deadline.
        time_to_deadline = self.deadline - current_time
        time_needed_on_demand = work_remaining + self.env.remaining_restart_overhead

        if time_to_deadline <= time_needed_on_demand + self.panic_buffer:
            return ClusterType.ON_DEMAND

        # 3. Proportional Control Mode.
        if current_time >= self.target_finish_time:
            # Past our soft deadline; all buffer is consumed. Must make progress.
            return ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND

        # Safeguard for the ideal_work_done calculation.
        if self.target_finish_time <= self.epsilon:
            return ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND

        # Calculate ideal progress based on a linear path to the soft deadline.
        ideal_work_done = self.task_duration * (current_time / self.target_finish_time)
        is_behind_schedule = work_done < ideal_work_done

        if has_spot:
            # Always take cheap progress when it's available.
            return ClusterType.SPOT
        else:
            # Spot is unavailable. Decide whether to pay for On-Demand or wait.
            if is_behind_schedule:
                # Behind our target; use On-Demand to catch up.
                return ClusterType.ON_DEMAND
            else:
                # Ahead of schedule; we can afford to wait for Spot.
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        """Required method for evaluator instantiation."""
        args, _ = parser.parse_known_args()
        return cls(args)