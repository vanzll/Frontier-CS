from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "OptimizedHysteresisStrategy"

    def solve(self, spec_path: str) -> "Solution":
        # No pre-computation needed
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate current progress
        work_done = sum(self.task_done_time)
        work_rem = self.task_duration - work_done
        time_rem = self.deadline - self.env.elapsed_seconds
        
        # Calculate slack: the amount of time we can afford to lose (overhead or waiting)
        # Slack = Time_Remaining - Work_Remaining
        slack = time_rem - work_rem
        
        overhead = self.restart_overhead
        
        # Define Thresholds
        
        # CRITICAL_SLACK: 1.5 * overhead
        # If we are running on Spot and get preempted, we incur 'overhead' to restart on OD.
        # If slack < overhead, a preemption means we fail the deadline.
        # We enforce OD usage if slack drops below this safety margin to guarantee completion.
        CRITICAL_SLACK = 1.5 * overhead
        
        # SWITCH_TO_SPOT_THRESHOLD: 4.0 * overhead
        # Switching from OD to Spot incurs an immediate 'overhead' cost in time.
        # We should only switch if we have substantial slack to absorb this cost 
        # and withstand potential future failures.
        SWITCH_TO_SPOT_THRESHOLD = 4.0 * overhead

        # Logic Implementation
        
        # 1. Safety First: If slack is too low, stay/switch to On-Demand to ensure finish.
        if slack < CRITICAL_SLACK:
            return ClusterType.ON_DEMAND

        # 2. Resource Selection
        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND:
                # We are currently on OD. Applying hysteresis avoids thrashing.
                # Only switch to Spot if we have excess slack.
                if slack > SWITCH_TO_SPOT_THRESHOLD:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            else:
                # We are on Spot or None (starting).
                # Since slack is healthy, use/keep Spot to minimize cost.
                return ClusterType.SPOT
        else:
            # Spot unavailable.
            # Waiting (NONE) reduces slack directly. 
            # Given the high penalty for failure and limited total slack (4h),
            # we immediately fall back to OD.
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)