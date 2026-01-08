import collections
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    """
    An adaptive strategy that balances cost and deadline risk based on "time slack".

    Core Idea:
    The strategy's main metric is "time slack", defined as:
    Slack = (Time Until Deadline) - (Work Remaining)
    This represents the idle time we can afford before risking the deadline.

    Decision Logic:
    - If `slack` is below an adaptive threshold, we are at risk. We switch to
      the reliable but expensive ON_DEMAND instance to guarantee progress.
    - If `slack` is above the threshold, we have a comfortable buffer. We
      prioritize cost-saving: use SPOT if available, otherwise wait (NONE).

    Adaptive Threshold:
    The threshold adapts to the recent historical availability of SPOT instances.
    - High SPOT availability lowers the threshold, allowing more risk-taking.
    - Low SPOT availability raises the threshold, prompting a more
      conservative, earlier switch to ON_DEMAND.

    This adaptability allows the strategy to perform well across different
    cloud environments (traces) with varying SPOT instance reliability.
    """
    NAME = "my_solution"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        """
        Initializes the strategy. Initialization of parameters is deferred
        to the first _step call to access environment variables.
        """
        self.initialized = False
        return self

    def _initialize(self):
        """
        One-time initialization on the first call to _step, which allows access
        to `self.env` and other instance attributes like `restart_overhead`.
        """
        # We estimate spot availability over a 2-hour rolling window.
        history_window_duration_s = 2 * 3600
        self.history_window_steps = max(1, int(history_window_duration_s / self.env.gap_seconds))
        
        self.spot_availability_history = collections.deque(maxlen=self.history_window_steps)
        
        # Base slack threshold: The minimum safety buffer, used when SPOT is 100%
        # available. Set to be slightly larger than one restart_overhead.
        self.base_slack_threshold = self.restart_overhead * 1.25

        # Max slack threshold: Used when SPOT is 0% available. It determines how
        # much slack we're willing to burn waiting for SPOT. We set this to
        # burn up to 3 hours of slack.
        max_wait_duration_s = 3 * 3600
        self.max_slack_threshold = self.base_slack_threshold + max_wait_duration_s

        # A burn-in period for the availability estimator to gather initial data.
        self.burn_in_steps = self.history_window_steps // 4

        self.initialized = True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step to decide the next cluster type to use.
        """
        if not self.initialized:
            self._initialize()

        # 1. Update knowledge base with current spot availability.
        self.spot_availability_history.append(1 if has_spot else 0)

        # 2. Assess current state.
        work_done = sum(end - start for start, end in self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        if work_remaining <= 1e-6:
            return ClusterType.NONE  # Job is done.

        time_to_deadline = self.deadline - self.env.elapsed_seconds
        
        # 3. Calculate time slack, our primary decision metric.
        slack = time_to_deadline - work_remaining
        
        # 4. Determine the adaptive slack threshold based on recent availability.
        num_history_points = len(self.spot_availability_history)
        if num_history_points < self.burn_in_steps:
            # During burn-in, use a conservative default availability assumption.
            recent_availability = 0.30
        else:
            recent_availability = sum(self.spot_availability_history) / num_history_points

        # Linearly interpolate the threshold.
        slack_threshold = self.max_slack_threshold - \
            (self.max_slack_threshold - self.base_slack_threshold) * recent_availability

        # 5. Make the decision.
        if slack <= slack_threshold:
            # Slack is below our safety margin; use ON_DEMAND.
            return ClusterType.ON_DEMAND
        else:
            # We have enough slack; prioritize cost.
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):  # REQUIRED: For evaluator instantiation
        args, _ = parser.parse_known_args()
        return cls(args)