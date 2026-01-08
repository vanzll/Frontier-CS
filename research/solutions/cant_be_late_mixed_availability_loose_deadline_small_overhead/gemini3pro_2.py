from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        Read spec_path for configuration if needed.
        Must return self.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step. Return which cluster type to use next.

        Args:
            last_cluster_type: The cluster type used in the previous step
            has_spot: Whether spot instances are available this step

        Returns:
            ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        # Calculate total work done and remaining
        work_done = sum(self.task_done_time) if self.task_done_time else 0.0
        work_remaining = max(0.0, self.task_duration - work_done)

        # If work is completed, pause (env usually handles this, but for safety)
        if work_remaining <= 0:
            return ClusterType.NONE

        # Current time and time remaining until deadline
        time_now = self.env.elapsed_seconds
        time_remaining = self.deadline - time_now

        # Calculate the time required to finish the job using On-Demand instances.
        # If we are not currently on On-Demand, we must account for the restart overhead
        # required to switch to or start an On-Demand instance.
        # If we are already on On-Demand, we assume no new overhead is incurred to continue.
        switch_overhead = 0.0
        if last_cluster_type != ClusterType.ON_DEMAND:
            switch_overhead = self.restart_overhead
        
        required_time_od = work_remaining + switch_overhead

        # Define a safety margin. Since we only make decisions every `gap_seconds`,
        # we must ensure that we don't cross the point of no return while waiting for the next step.
        # A margin of 2.0 * gap_seconds provides a robust buffer against time quantization.
        margin = 2.0 * self.env.gap_seconds

        # Panic Logic:
        # If the time remaining is getting close to the minimum time required to finish via OD,
        # we must switch to OD immediately to guarantee meeting the hard deadline.
        if time_remaining <= required_time_od + margin:
            return ClusterType.ON_DEMAND

        # Cost Minimization Logic:
        # If we have slack (not in panic mode), we prefer using Spot instances (cheaper).
        if has_spot:
            return ClusterType.SPOT
        
        # If Spot is unavailable but we still have slack, we choose NONE (pause).
        # This saves money compared to running OD prematurely, preserving the budget 
        # for when Spot becomes available or for the mandatory OD finish.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)