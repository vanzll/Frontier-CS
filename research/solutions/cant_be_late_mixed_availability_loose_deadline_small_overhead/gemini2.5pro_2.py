import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    # --- Tunable Parameters ---
    # The strategy aims to finish this many hours before the hard deadline,
    # creating a "soft deadline" for its pacing calculations.
    DEADLINE_BUFFER_HOURS = 12.0

    # The strategy enters a "panic mode" and uses only On-Demand if the
    # remaining time is less than the work left plus a buffer. This buffer is
    # defined as a number of potential preemption overheads.
    PANIC_PREEMPTIONS = 60

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize calculated constants for the strategy based on the
        task specification and tunable parameters. This is called once
        before the simulation begins.
        """
        self.deadline_buffer_seconds = self.DEADLINE_BUFFER_HOURS * 3600.0
        self.soft_deadline = self.deadline - self.deadline_buffer_seconds

        # Sanity check: ensure the soft deadline is feasible. If the buffer is
        # too aggressive, fall back to a 10% time buffer over the task duration.
        if self.soft_deadline <= self.task_duration:
            self.soft_deadline = self.task_duration * 1.1

        # The panic buffer is a fixed time value for the final safety check.
        self.panic_buffer_seconds = self.PANIC_PREEMPTIONS * self.restart_overhead
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide which cluster type to use for the next time step.
        This implements the core logic of the scheduling strategy.
        """
        work_done = sum(end - start for start, end in self.task_done_time)
        work_remaining = self.task_duration - work_done

        # If the job is complete, do nothing to save cost.
        if work_remaining <= 0:
            return ClusterType.NONE

        current_time = self.env.elapsed_seconds
        time_to_hard_deadline = self.deadline - current_time

        # 1. Panic Check (Point of No Return): If we are running out of time
        # to finish even on guaranteed resources, switch to On-Demand.
        if work_remaining + self.panic_buffer_seconds >= time_to_hard_deadline:
            return ClusterType.ON_DEMAND

        # 2. Main Heuristic: Balance cost vs. progress.
        if has_spot:
            # Always prefer cheap Spot instances when available and not in panic mode.
            return ClusterType.SPOT
        else:
            # Spot is unavailable. Decide whether to use expensive On-Demand or wait.
            # This decision is based on progress towards our "soft" deadline.
            if self.soft_deadline <= 0:
                # Failsafe for invalid configuration.
                return ClusterType.ON_DEMAND

            target_work_done = current_time * (self.task_duration / self.soft_deadline)

            if work_done < target_work_done:
                # We are behind our soft schedule; use On-Demand to catch up.
                return ClusterType.ON_DEMAND
            else:
                # We are ahead of our soft schedule; we can afford to wait for
                # Spot to become available again to save costs.
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser: "argparse.ArgumentParser"):
        """
        Required method for the evaluator to instantiate the strategy.
        This solution does not use command-line arguments.
        """
        args, _ = parser.parse_known_args()
        return cls(args)