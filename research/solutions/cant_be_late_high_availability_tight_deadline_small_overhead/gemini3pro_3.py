from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateStrategy"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate total work completed and remaining
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        
        # If task is already complete, stop using resources
        if remaining_work <= 0:
            return ClusterType.NONE
            
        # Calculate time remaining until deadline
        time_until_deadline = self.deadline - self.env.elapsed_seconds
        
        # Calculate slack: the amount of time we can afford to waste (wait or restart)
        # before we jeopardize the deadline.
        slack = time_until_deadline - remaining_work
        
        # Define a safety buffer.
        # We need to account for the restart overhead (latency to start a new instance/recover).
        # We also need to account for the simulation step gap.
        # Factor of 2.5 on overhead ensures we have enough time to handle a transition to OD
        # or a final spot failure without missing the hard deadline.
        # We assume worst case: we might need to switch to OD immediately, incurring overhead.
        buffer = (2.5 * self.restart_overhead) + (2.0 * self.env.gap_seconds)
        
        # Safety Logic:
        # If our slack drops below the safety buffer, we are dangerously close to the deadline.
        # We must switch to On-Demand (guaranteed availability, no preemption) to ensure we finish.
        if slack < buffer:
            return ClusterType.ON_DEMAND
            
        # Cost Optimization Logic:
        # If we have sufficient slack, we prioritize minimizing cost.
        if has_spot:
            # Spot is available and we have a safety buffer. Use Spot (Cheapest).
            return ClusterType.SPOT
        else:
            # Spot is unavailable, but we still have slack.
            # We choose to WAIT (NONE) rather than paying for On-Demand.
            # Burning slack is "free" monetarily, whereas OD is expensive.
            # We only switch to OD when the slack is consumed (handled by Safety Logic above).
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)