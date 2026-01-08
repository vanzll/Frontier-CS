from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "cant_be_late_solution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = self.env.elapsed_seconds
        done = sum(self.task_done_time)
        remaining_work = self.task_duration - done

        if remaining_work <= 0:
            return ClusterType.NONE

        remaining_time = self.deadline - elapsed
        slack = remaining_time - remaining_work
        overhead = self.restart_overhead
        gap = self.env.gap_seconds

        # Panic Threshold: Minimum slack required to start On-Demand and finish on time.
        # We need 'overhead' time just to start. We add a buffer (0.5*overhead + gap) 
        # to ensure we don't miss the window due to discrete time steps.
        panic_threshold = (overhead * 1.5) + gap

        # Switch Threshold: Slack required to justify switching from On-Demand to Spot.
        # Switching costs 'overhead' time. We must ensure that after paying this cost,
        # we still have enough slack (> panic_threshold) to recover if Spot fails immediately.
        switch_threshold = panic_threshold + overhead

        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND:
                # Only switch from expensive OD to Spot if we have ample slack
                if slack > switch_threshold:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            else:
                # Currently NONE or SPOT, and Spot is available -> take it
                return ClusterType.SPOT
        else:
            # Spot not available
            if last_cluster_type == ClusterType.ON_DEMAND:
                # Already on OD, stay on OD to avoid paying restart overhead later
                return ClusterType.ON_DEMAND
            else:
                # Currently NONE or SPOT (just lost)
                # Wait for Spot unless we are dangerously close to the deadline
                if slack < panic_threshold:
                    return ClusterType.ON_DEMAND
                else:
                    return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)