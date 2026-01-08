from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        This method sets up the thresholds for the scheduling policy
        based on the problem's parameters.
        """
        initial_slack = self.deadline - self.task_duration
        
        # Estimate the time lost to a single preemption event.
        # self.env.gap_seconds is not available here, so we assume a common value.
        assumed_gap_seconds = 60
        cost_per_preemption = self.restart_overhead + assumed_gap_seconds

        # Safety threshold: A buffer to withstand several preemptions.
        # If slack drops below this, we enter a "panic mode" and only use
        # guaranteed On-Demand instances.
        self.SAFETY_THRESHOLD = 7.5 * cost_per_preemption
        
        # Aggressive threshold: A level of slack above which we can afford
        # to wait for cheap Spot instances. This is set to half the initial
        # slack, balancing cost savings against the risk of burning slack.
        self.AGGRESSIVE_THRESHOLD = initial_slack / 2.0
        
        # Ensure thresholds are logical, even with unusual problem parameters.
        if self.AGGRESSIVE_THRESHOLD < self.SAFETY_THRESHOLD:
            self.AGGRESSIVE_THRESHOLD = self.SAFETY_THRESHOLD

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step. Return which cluster type to use next.

        This strategy uses a three-mode policy based on the current "slack",
        which is the amount of time we can afford to not make progress
        and still finish before the deadline.

        1. Aggressive Mode (high slack): Prioritize cost savings. Use Spot
           if available, otherwise wait (NONE) for it to become available.
        2. Cautious Mode (medium slack): Balance cost and progress. Use Spot
           if available, but use On-Demand if not, to avoid losing more slack.
        3. Panic Mode (low slack): Prioritize finishing on time. Always use
           On-Demand for guaranteed progress.
        """
        # 1. Calculate current state variables
        current_time = self.env.elapsed_seconds
        
        work_done = sum(end - start for start, end in self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        time_to_deadline = self.deadline - current_time
        
        if work_remaining >= time_to_deadline:
             return ClusterType.ON_DEMAND

        current_slack = time_to_deadline - work_remaining

        # 2. Apply the three-mode, two-threshold policy
        if current_slack <= self.SAFETY_THRESHOLD:
            # Panic Mode: Critically low slack, must use On-Demand.
            return ClusterType.ON_DEMAND
        
        elif current_slack <= self.AGGRESSIVE_THRESHOLD:
            # Cautious Mode: Use Spot if possible, otherwise On-Demand.
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        
        else:
            # Aggressive Mode: Plenty of slack, can afford to wait for Spot.
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        """
        Required classmethod for evaluator instantiation.
        """
        args, _ = parser.parse_known_args()
        return cls(args)