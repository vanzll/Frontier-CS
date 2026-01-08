from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateStrategy"

    def __init__(self, args):
        super().__init__(args)
        self.cached_work_done = 0.0
        self.cached_idx = 0

    def solve(self, spec_path: str) -> "Solution":
        # Reset cache at the start of evaluation
        self.cached_work_done = 0.0
        self.cached_idx = 0
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide the cluster type for the next step.
        Strategy: Use Spot instances to minimize cost, but switch to On-Demand
        if the remaining time gets dangerously close to the time required to complete the work.
        """
        # Efficiently calculate total work done by caching the sum
        # self.task_done_time is a list of completed work segments (durations)
        current_len = len(self.task_done_time)
        if current_len > self.cached_idx:
            for i in range(self.cached_idx, current_len):
                self.cached_work_done += self.task_done_time[i]
            self.cached_idx = current_len

        remaining_work = self.task_duration - self.cached_work_done
        elapsed = self.env.elapsed_seconds
        remaining_time = self.deadline - elapsed

        # Calculate the panic threshold.
        # If we switch to OD, we incur restart_overhead.
        # We also add a buffer of 2 * gap_seconds to handle step granularity safety.
        # Threshold = Remaining Work + Overhead + Safety Buffer
        safety_buffer = self.restart_overhead + (2.0 * self.env.gap_seconds)
        panic_threshold = remaining_work + safety_buffer

        # 1. Panic Mode: Deadline is approaching.
        # If remaining time is less than what's needed for OD (plus overhead/buffer),
        # we must use OD immediately to guarantee completion.
        if remaining_time <= panic_threshold:
            return ClusterType.ON_DEMAND

        # 2. Standard Mode: We have slack.
        # If Spot is available, use it (cheapest option).
        if has_spot:
            return ClusterType.SPOT

        # 3. Wait Mode: Spot unavailable, but we have slack.
        # Do nothing (NONE) to save cost. We burn slack time but save money.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)