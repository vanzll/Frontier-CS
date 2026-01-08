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
        # --- Hyperparameters for the slack-based policy ---
        # If slack is below this (in hours), always use On-Demand as a safety buffer.
        self.critical_slack_hours = 1.5
        # If slack is below this (in hours), use On-Demand if Spot is unavailable.
        self.comfortable_slack_hours = 10.0
        
        # Convert hours to seconds for efficiency in the _step method.
        self.critical_slack_seconds = self.critical_slack_hours * 3600
        self.comfortable_slack_seconds = self.comfortable_slack_hours * 3600

        # Small tolerance for floating point comparisons.
        self.epsilon = 1e-6
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
        # Calculate the total amount of work successfully completed.
        work_done = sum(end - start for start, end in self.task_done_time)

        # Calculate the remaining work required to finish the task.
        work_remaining = self.task_duration - work_done

        # If the task is completed, do nothing to save costs.
        if work_remaining <= self.epsilon:
            return ClusterType.NONE

        # Calculate the time remaining until the hard deadline.
        current_time = self.env.elapsed_seconds
        time_to_deadline = self.deadline - current_time

        # Slack is the time buffer we have if we complete the rest of the job
        # using reliable on-demand instances.
        slack_seconds = time_to_deadline - work_remaining

        # --- Three-Zone Decision Logic based on Slack ---

        # 1. Critical Zone: Slack is below the safety threshold.
        # Must use on-demand to guarantee completion.
        if slack_seconds < self.critical_slack_seconds:
            return ClusterType.ON_DEMAND

        # 2. Cautious Zone: Slack is moderate.
        # Prioritize progress. Use Spot if available, otherwise use on-demand
        # to avoid burning slack by waiting.
        if slack_seconds < self.comfortable_slack_seconds:
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND

        # 3. Comfortable Zone: Slack is abundant.
        # Be aggressive with cost-saving. Use Spot when available, and wait
        # (NONE) if it's not.
        else:
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):  # REQUIRED: For evaluator instantiation
        args, _ = parser.parse_known_args()
        return cls(args)