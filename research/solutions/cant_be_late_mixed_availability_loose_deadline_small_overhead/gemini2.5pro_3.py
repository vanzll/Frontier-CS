from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        Initializes hyperparameters and state for the strategy.
        """
        # Alpha for the Exponential Moving Average (EMA) of spot availability.
        # A smaller value means a longer memory (smoother, less reactive).
        self.ema_alpha = 0.005

        # A safety factor for the decision logic. We use ON_DEMAND unless our
        # estimated spot availability is safely higher than the required rate.
        # A value > 1.0 makes the strategy more risk-averse.
        self.safety_factor = 1.25

        # Initial estimate for spot availability probability.
        # A pessimistic value makes the strategy cautious at the beginning.
        self.p_spot_est = 0.30

        self.initialized = True
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step. Return which cluster type to use next.
        """
        # Ensure initialization has run.
        if not hasattr(self, 'initialized') or not self.initialized:
            self.solve("")

        # 1. Update the Exponential Moving Average of spot availability.
        self.p_spot_est = (self.ema_alpha * int(has_spot) +
                           (1.0 - self.ema_alpha) * self.p_spot_est)

        # 2. Calculate the current state of the job.
        work_done = sum(end - start for start, end in self.task_done_time)
        work_remaining = self.task_duration - work_done

        # If the job is complete, do nothing to minimize cost.
        if work_remaining <= 1e-9:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds

        # If the deadline has passed, our only hope to finish is ON_DEMAND.
        if time_to_deadline <= 1e-9:
            return ClusterType.ON_DEMAND

        # 3. Point of No Return (PNR) safety check.
        # If the time required to finish (including a potential restart)
        # exceeds the time left, we must use ON_DEMAND.
        if work_remaining + self.restart_overhead >= time_to_deadline:
            return ClusterType.ON_DEMAND

        # 4. Main decision logic.
        if has_spot:
            # Spot instances are cheap and make progress. Almost always the best choice.
            return ClusterType.SPOT
        else:
            # Spot is not available. Decide between ON_DEMAND (costly) and NONE (risky).
            # The decision is based on comparing the required progress rate with our
            # estimate of future spot availability.

            # The average rate of work needed from this point to the deadline.
            required_rate = work_remaining / time_to_deadline

            # If our estimated spot availability (with a safety margin) is not
            # sufficient to meet the required rate, we must use ON_DEMAND.
            if self.p_spot_est * self.safety_factor < required_rate:
                return ClusterType.ON_DEMAND
            else:
                # We have enough slack and a reasonable expectation of future spot
                # availability, so we can afford to wait.
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        """
        Required classmethod for evaluator instantiation.
        """
        args, _ = parser.parse_known_args()
        return cls(args)