import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    """
    A strategy that aims to minimize cost by maximizing Spot instance usage
    while maintaining a safety buffer of time to ensure completion before the
    deadline.
    """
    NAME = "my_solution"

    # This is a tunable hyperparameter. It represents the fraction of the
    # initial total slack time that the strategy will attempt to preserve as a
    # safety buffer. A value of 0.3 means it will start using expensive
    # On-Demand instances only when 70% of the initial slack has been consumed
    # by waiting for Spot or by restart overheads.
    DEFAULT_SLACK_FRACTION_TO_KEEP = 0.3

    def solve(self, spec_path: str) -> "Solution":
        """
        Initializes the strategy. This method is called once before the
        simulation starts. It calculates the safety buffer threshold based on
        the task parameters.
        """
        # Retrieve the hyperparameter for this instance. It might have been
        # set by _from_args. If not, use the class default.
        slack_fraction = getattr(self, "slack_fraction_to_keep", self.DEFAULT_SLACK_FRACTION_TO_KEEP)

        # The initial total slack is the time between the deadline and the
        # raw task duration.
        initial_slack = self.deadline - self.task_duration

        # Calculate the absolute time buffer to maintain. The core of the
        # strategy is to not let the current slack drop below this threshold.
        self.buffer_threshold = initial_slack * slack_fraction

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Makes a decision at each time step on which cluster type to use.
        The logic is as follows:
        1. If Spot is available, always use it (greedy for cost savings).
        2. If Spot is not available, check the current time slack.
        3. If slack is below a safety threshold, use On-Demand to guarantee progress.
        4. If slack is healthy, wait (use None) for a Spot instance to appear.
        """
        # Priority 1: Always use cheap Spot instances when they are available.
        if has_spot:
            return ClusterType.SPOT

        # If Spot is not available, we must decide between expensive progress
        # (On-Demand) or free waiting (None). The decision is based on deadline
        # pressure.

        # Calculate the total work completed so far.
        # self.task_done_time is a list of [start, end] tuples.
        work_done = sum(end - start for start, end in self.task_done_time)

        # Calculate remaining work and time.
        work_remaining = self.task_duration - work_done
        time_remaining = self.deadline - self.env.elapsed_seconds

        # The current slack is the amount of "extra" time we have.
        # It's the time remaining minus the work remaining. If this is negative,
        # we are already behind schedule.
        current_slack = time_remaining - work_remaining

        # Priority 2: If our time buffer is depleted, we must act.
        # By comparing the current slack to our pre-calculated safety buffer,
        # we decide if the situation is critical.
        if current_slack <= self.buffer_threshold:
            # Slack is too low; we must use On-Demand to make guaranteed progress.
            return ClusterType.ON_DEMAND
        else:
            # Slack is sufficient; we can afford to wait for a cheaper Spot instance.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        """
        This method allows the evaluation environment to instantiate the
        solution and pass command-line arguments for hyperparameter tuning.
        """
        parser.add_argument(
            '--slack-fraction-to-keep',
            type=float,
            default=cls.DEFAULT_SLACK_FRACTION_TO_KEEP,
            help='Fraction of the initial slack to keep as a safety buffer.'
        )

        args, _ = parser.parse_known_args()

        # The base Strategy constructor is expected to accept the parsed args.
        instance = cls(args)

        # Store the specific hyperparameter on the instance for use in solve().
        instance.slack_fraction_to_keep = args.slack_fraction_to_keep

        return instance