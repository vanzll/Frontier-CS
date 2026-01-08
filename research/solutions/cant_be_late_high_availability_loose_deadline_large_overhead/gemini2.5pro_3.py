import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args=None):
        super().__init__()
        # Hyperparameters for the strategy, tunable via command-line arguments.
        self.slack_threshold_factor = args.slack_threshold_factor if args else 10.0
        self.safety_margin_factor = args.safety_margin_factor if args else 1.0

        # These will be initialized in solve() based on environment parameters.
        self.safety_margin = 0
        self.slack_threshold = 0

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        Read spec_path for configuration if needed.
        Must return self.
        """
        # Initialize thresholds based on the specific task's restart overhead.
        self.safety_margin = self.safety_margin_factor * self.restart_overhead
        self.slack_threshold = self.slack_threshold_factor * self.restart_overhead
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
        # Calculate remaining work.
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        # If the job is done, switch to NONE to minimize cost.
        if work_remaining <= 0:
            return ClusterType.NONE

        # Calculate the time needed to finish if we use On-Demand from now on.
        # This is our worst-case time to completion.
        time_needed_on_demand = work_remaining + self.env.remaining_restart_overhead
        
        # Project the finish time assuming we switch to On-Demand immediately.
        projected_finish_time_od = self.env.elapsed_seconds + time_needed_on_demand
        
        # Calculate the available time buffer against the deadline in the worst-case scenario.
        current_slack = self.deadline - projected_finish_time_od

        # --- Decision Logic ---

        # 1. Panic Mode: If the slack is below a critical safety margin, we must use
        #    On-Demand to guarantee finishing before the deadline.
        if current_slack <= self.safety_margin:
            return ClusterType.ON_DEMAND

        # 2. Normal Mode: We have some slack to work with.
        if has_spot:
            # If Spot instances are available, they are the most cost-effective choice.
            return ClusterType.SPOT
        else:
            # Spot is not available. We must choose between waiting (NONE) or paying for
            # guaranteed progress (ON_DEMAND).
            if current_slack <= self.slack_threshold:
                # If our slack buffer is running low, it's too risky to wait.
                # Use On-Demand to make progress and preserve the remaining slack.
                return ClusterType.ON_DEMAND
            else:
                # If we have a comfortable amount of slack, we can afford to wait
                # for Spot instances to become available again.
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):  # REQUIRED: For evaluator instantiation
        """
        Allows the evaluation framework to instantiate the class and pass arguments.
        """
        parser.add_argument(
            '--slack-threshold-factor', 
            type=float, 
            default=10.0,
            help='Factor of restart_overhead to set the slack threshold.'
        )
        parser.add_argument(
            '--safety-margin-factor', 
            type=float, 
            default=1.0,
            help='Factor of restart_overhead to set the safety margin.'
        )
        args, _ = parser.parse_known_args()
        return cls(args)