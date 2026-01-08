from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        deadline = self.deadline
        total_duration = self.task_duration
        restart_overhead = self.restart_overhead

        work_done = sum(self.task_done_time)
        work_remaining = total_duration - work_done

        # If work is effectively done, stop
        if work_remaining <= 1e-6:
            return ClusterType.NONE

        time_remaining = deadline - elapsed
        
        # Margin to handle simulation step granularity
        margin = 1.5 * gap

        # Calculate the latest possible time we must act to guarantee completion on OD.
        # We assume we might need to incur restart overhead if we are not already running OD efficiently.
        # Even if we are on OD, we use this threshold to lock us in.
        panic_threshold = work_remaining + restart_overhead + margin

        # Priority 1: Meet the hard deadline.
        if time_remaining <= panic_threshold:
            return ClusterType.ON_DEMAND

        # Priority 2: Use Spot instances to minimize cost.
        if has_spot:
            # If we are currently on On-Demand, switching to Spot incurs a restart overhead (cost/time).
            # We only switch if we have enough slack to absorb this penalty and potential failure.
            if last_cluster_type == ClusterType.ON_DEMAND:
                # Buffer: overhead to switch, plus overhead to switch back if Spot fails immediately
                switch_safety_buffer = 3.0 * restart_overhead + margin
                if time_remaining > work_remaining + switch_safety_buffer:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        # Priority 3: Wait for Spot (NONE) to save money.
        # If we are currently on OD but Spot is unavailable, we might pause.
        # However, pausing means we will eventually pay restart overhead to resume.
        # We only pause if we have substantial slack.
        if last_cluster_type == ClusterType.ON_DEMAND:
            # Hysteresis to prevent rapid toggling and overhead accumulation
            stop_od_buffer = 2.0 * restart_overhead + margin
            if time_remaining > panic_threshold + stop_od_buffer:
                return ClusterType.NONE
            else:
                return ClusterType.ON_DEMAND

        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)