import collections
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    """
    An adaptive strategy for the Cant-Be-Late Scheduling Problem.

    The core idea is a two-level thresholding policy based on the concept of "slack",
    which is the amount of time available beyond the bare minimum required to finish
    the job using reliable on-demand instances.

    1.  **Panic Threshold**: A hard threshold to guarantee completion. If the time
        remaining until the deadline is less than the work remaining plus a small
        safety buffer (the "panic buffer"), the strategy switches to On-Demand
        and stays there. This is the last resort to avoid failing the deadline.

    2.  **Patience Threshold**: A soft, adaptive threshold to decide between waiting
        for cheap Spot instances (NONE) or making progress with expensive On-Demand
        instances when Spot is unavailable. This is the main cost-optimization logic.
        - The strategy maintains a running estimate of the spot instance availability.
        - Based on this estimate, it calculates a "patience buffer". If spot
          availability is high, it's worth waiting longer, so the patience buffer
          is small. If availability is low, waiting is risky (burns too much slack),
          so the patience buffer is large, leading to a quicker switch to On-Demand.
        - The decision is: if the current `slack < patience_buffer`, use On-Demand;
          otherwise, wait (NONE).

    This approach aims to aggressively use cheap Spot instances when available and
    intelligently fall back to On-Demand, adapting its risk tolerance based on
    observed market conditions (spot availability).
    """
    NAME = "my_solution"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        """
        Initializes the strategy's parameters based on the problem specification.
        This method is called once before the simulation begins.
        """
        # A safety buffer for the panic mode, slightly larger than one restart
        # overhead to account for the time step and potential preemption.
        self.panic_buffer = self.restart_overhead * 1.1

        # A tunable factor that determines the baseline "patience" of the strategy.
        # This value was chosen as a reasonable heuristic based on the initial
        # total slack available in the problem. It's used to scale the adaptive
        # patience buffer.
        self.patience_factor = 12.0

        # A deque to maintain a sliding window of recent spot availability,
        # used for calculating a running estimate.
        self.spot_history = collections.deque(maxlen=500)

        # An initial guess for spot availability, based on the problem description (4-40%).
        # We start with the midpoint. This value adapts as the simulation runs.
        self.p_spot_estimate = 0.22

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        The main decision-making function, called at each time step.
        """
        # 1. Update the running estimate of spot availability.
        self.spot_history.append(1 if has_spot else 0)
        if len(self.spot_history) > 0:
            self.p_spot_estimate = sum(self.spot_history) / len(self.spot_history)

        # 2. Calculate the current state of the job.
        total_work_done = sum(end - start for start, end in self.task_done_time)
        remaining_work = self.task_duration - total_work_done

        # If the job is complete, we no longer need any resources.
        if remaining_work <= 0:
            return ClusterType.NONE

        time_left = self.deadline - self.env.elapsed_seconds

        # 3. Apply the decision logic.

        # 3a. Panic Check: If finishing on time is at risk, we must use On-Demand.
        # This condition triggers when the time left is barely enough to complete the
        # remaining work, including a small safety buffer.
        if remaining_work + self.panic_buffer >= time_left:
            return ClusterType.ON_DEMAND

        # 3b. Ideal Case: If not in panic mode and Spot is available, always use it.
        # It's the most cost-effective way to make progress.
        if has_spot:
            return ClusterType.SPOT

        # 3c. Patience Logic: Spot is unavailable. Decide between waiting (NONE) or
        # using On-Demand to make progress.
        # The decision depends on how much slack we can afford to lose while waiting.

        # An epsilon to prevent division by zero if the spot estimate is transiently zero.
        epsilon = 1e-9
        # Calculate the adaptive patience buffer. It's inversely proportional to the
        # estimated spot availability.
        patience_buffer = (self.patience_factor * self.restart_overhead) / (self.p_spot_estimate + epsilon)

        # Slack is the time we have beyond the minimum needed to finish on On-Demand.
        slack = time_left - remaining_work

        # If our available slack is less than our patience threshold, our patience
        # has run out. We must use On-Demand to guarantee progress.
        if slack <= patience_buffer:
            return ClusterType.ON_DEMAND
        else:
            # We have sufficient slack, so we can afford to wait for Spot to become available.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):  # REQUIRED: For evaluator instantiation
        """
        Class method to instantiate the strategy from command-line arguments.
        """
        args, _ = parser.parse_known_args()
        return cls(args)