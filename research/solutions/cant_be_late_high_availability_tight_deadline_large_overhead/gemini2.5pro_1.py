from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    # This factor determines the size of the "panic" buffer, relative to restart overhead.
    # If current slack is below this buffer, we use ON_DEMAND exclusively.
    URGENT_BUFFER_FACTOR = 1.5

    # This fraction determines how much of our "safe" slack (total slack minus the urgent buffer)
    # we are willing to spend waiting for spot instances to become available.
    WAIT_BUDGET_FRACTION = 0.5

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the strategy with calculated thresholds based on the problem spec.
        This method is called once before the simulation starts.
        """
        initial_slack = self.deadline - self.task_duration

        # Define the "panic" threshold. This is the point where we can no longer
        # risk using spot instances or waiting.
        self.buffer_urgent = self.URGENT_BUFFER_FACTOR * self.restart_overhead

        # Add a guardrail in case of very large overhead or very small initial slack.
        # The urgent buffer cannot be larger than the total available slack.
        if initial_slack > 0:
            self.buffer_urgent = min(self.buffer_urgent, initial_slack)
        else:
            # If there's no slack to begin with, we are always in urgent mode.
            self.buffer_urgent = 0

        # Define the "wait" threshold. This divides the "safe" zone into a
        # "patient" zone (where we wait for spot) and a "cautious" zone
        # (where we use on-demand if spot is unavailable).
        safe_slack = max(0, initial_slack - self.buffer_urgent)
        wait_budget = safe_slack * self.WAIT_BUDGET_FRACTION
        self.buffer_wait = self.buffer_urgent + wait_budget
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decision-making logic for each time step.
        The core of the strategy is to partition the decision space into three zones
        based on the current slack time.
        """
        # Calculate current work progress.
        work_done_seconds = sum(end - start for start, end in self.task_done_time)
        work_remaining_seconds = self.task_duration - work_done_seconds

        # If the job is finished, do nothing to save costs.
        if work_remaining_seconds <= 0:
            return ClusterType.NONE

        # Calculate time remaining until the deadline.
        time_to_deadline_seconds = self.deadline - self.env.elapsed_seconds
        
        # Slack is the key metric: the buffer between the time we have and the time we need.
        current_slack_seconds = time_to_deadline_seconds - work_remaining_seconds

        # --- Three-Zone Decision Logic ---

        # 1. PANIC ZONE (slack <= buffer_urgent)
        # Slack is critically low. We must use the reliable on-demand instance
        # to guarantee progress and finish before the deadline.
        if current_slack_seconds <= self.buffer_urgent:
            return ClusterType.ON_DEMAND

        # If not in the panic zone, our preferred option is always the cheap spot instance.
        if has_spot:
            return ClusterType.SPOT

        # If spot is not available, our decision depends on which of the other two zones we are in.
        
        # 2. PATIENT ZONE (slack > buffer_wait)
        # We have a comfortable amount of slack. It's cost-effective to wait for a
        # spot instance to become available rather than paying for on-demand.
        if current_slack_seconds > self.buffer_wait:
            return ClusterType.NONE
        
        # 3. CAUTIOUS ZONE (buffer_urgent < slack <= buffer_wait)
        # Our slack buffer is shrinking. We can no longer afford to wait.
        # We must make progress, so we use an on-demand instance.
        else:
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        """
        Required classmethod for evaluator instantiation.
        """
        args, _ = parser.parse_known_args()
        return cls(args)