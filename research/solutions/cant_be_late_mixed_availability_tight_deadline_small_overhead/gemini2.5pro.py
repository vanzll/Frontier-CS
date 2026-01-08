import argparse

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        Read spec_path for configuration if needed.
        Must return self.
        """
        # --- Tunable Parameters ---

        # Initial conservative guess for spot efficiency. Based on the problem
        # spec, low-availability regions are 4-40%. We pick a pessimistic
        # value to be cautious at the start.
        self.initial_spot_efficiency = 0.35

        # Additive smoothing factor. This represents an initial "belief" about
        # spot performance, equivalent to 1 hour of observation time. It helps
        # stabilize the efficiency estimate at the beginning of the run.
        self.smoothing_time_seconds = 3600.0

        # --- State Variables for Online Estimation ---
        self.smoothing_work_seconds = self.smoothing_time_seconds * self.initial_spot_efficiency

        # Track total time spent attempting to use spot instances.
        self.time_spent_on_spot = 0.0

        # Track effective work accomplished by spot instances.
        self.work_done_on_spot = 0.0

        # Store the total work done at the previous step to calculate delta.
        self.last_total_work_done = 0.0

        return self

    def _get_total_work_done(self) -> float:
        """Helper to calculate total completed work from segments."""
        if not self.task_done_time:
            return 0.0
        return sum(end - start for start, end in self.task_done_time)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step. Return which cluster type to use next.

        Args:
            last_cluster_type: The cluster type used in the previous step
            has_spot: Whether spot instances are available this step

        Returns:
            ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        # 1. Update online estimates based on the outcome of the last step.
        current_total_work_done = self._get_total_work_done()
        work_done_last_step = current_total_work_done - self.last_total_work_done

        if last_cluster_type == ClusterType.SPOT:
            self.time_spent_on_spot += self.env.gap_seconds
            self.work_done_on_spot += work_done_last_step

        self.last_total_work_done = current_total_work_done

        # 2. Calculate current job status.
        work_remaining = self.task_duration - current_total_work_done

        # If the job is done, do nothing to save costs.
        if work_remaining <= 1e-9:
            return ClusterType.NONE

        time_available = self.deadline - self.env.elapsed_seconds

        # If deadline is missed or imminent, use On-Demand as a last resort.
        # This is a safeguard; the main logic should prevent this state.
        if time_available <= 1e-9:
            return ClusterType.ON_DEMAND

        # 3. Core decision logic: Compare required progress rate vs. estimated spot efficiency.

        # The required progress rate is the amount of work we must complete per
        # second of wall-clock time to finish exactly at the deadline.
        required_progress_rate = work_remaining / time_available

        # Estimate the effective progress rate of spot instances using our online data.
        # An efficiency of 0.5 means for every hour on Spot, we get 30 mins of work.
        denominator = self.time_spent_on_spot + self.smoothing_time_seconds
        spot_efficiency_estimate = (self.work_done_on_spot + self.smoothing_work_seconds) / denominator

        # Decision: If the required rate to meet the deadline is higher than what Spot
        # can reliably provide, we must switch to On-Demand.
        if required_progress_rate >= spot_efficiency_estimate:
            # We don't have enough time slack to absorb potential Spot issues.
            # Using On-Demand is necessary to guarantee progress.
            return ClusterType.ON_DEMAND
        else:
            # We have a sufficient time buffer. It's cost-effective to try using Spot.
            if has_spot:
                # Spot is our preferred cheap option and it's available.
                return ClusterType.SPOT
            else:
                # Spot is preferred, but currently unavailable. Since our schedule has
                # enough slack, we can afford to wait for it. This saves money
                # compared to using expensive On-Demand.
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser):  # REQUIRED: For evaluator instantiation
        """
        Instantiates the strategy from command-line arguments.
        """
        args, _ = parser.parse_known_args()
        return cls(args)