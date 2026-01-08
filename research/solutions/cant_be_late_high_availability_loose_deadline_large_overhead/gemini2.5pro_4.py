import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "dynamic_slack_heuristic"

    # --- Tunable Parameters ---

    # Determines the final safety buffer. If slack falls below
    # (FACTOR * restart_overhead), we switch to On-Demand permanently.
    # A larger value is safer and provides a larger buffer against a long
    # series of spot unavailability.
    SAFETY_MARGIN_FACTOR = 5.0

    # Determines our "patience" when spot is unavailable. If normalized
    # slack falls below this threshold, we use On-Demand instead of waiting.
    # A higher value (e.g., 0.8) means we are less patient (safer, but more expensive).
    # A lower value (e.g., 0.3) means we are more patient (cheaper, but riskier).
    WAIT_SLACK_THRESHOLD = 0.7

    # --- Strategy Implementation ---

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize strategy-specific variables and pre-calculate constants.
        This method is called once before the simulation starts.
        """
        # State variable to track the remaining time penalty from preemptions.
        self.pending_overhead = 0.0
        
        # Calculate constants based on the environment specification.
        self.initial_slack = self.deadline - self.task_duration
        self.safety_margin = self.SAFETY_MARGIN_FACTOR * self.restart_overhead
        
        # This is the range of slack we are willing to "risk" by waiting for Spot.
        self.risk_slack_range = self.initial_slack - self.safety_margin
        # Prevent division by zero if the problem has no slack to begin with.
        if self.risk_slack_range <= 0:
            self.risk_slack_range = 1.0

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Main decision logic called at each time step of the simulation.
        """
        # 1. Update internal state: pending_overhead
        # If we were running on any cluster in the previous step, we "pay down"
        # the restart overhead time.
        if last_cluster_type != ClusterType.NONE:
            self.pending_overhead = max(0.0, self.pending_overhead - self.env.gap_seconds)
        
        # Check for a preemption event: we were using a Spot instance, and it has
        # now become unavailable. This incurs a new restart overhead, which
        # replaces any previously remaining overhead.
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self.pending_overhead = self.restart_overhead

        # 2. Assess the current situation by calculating key metrics
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        # The total time we need to finish from this point forward, assuming a
        # 100% reliable cluster. This includes the actual work left and any
        # time penalty from restart overhead.
        effective_work_remaining = work_remaining + self.pending_overhead
        
        time_to_deadline = self.deadline - self.env.elapsed_seconds
        
        # Slack is our critical time buffer. If it's negative, we are behind schedule.
        current_slack = time_to_deadline - effective_work_remaining

        # 3. Apply decision rules based on the calculated slack
        
        # RULE 1: Final Sprint / Emergency Mode
        # If our slack has fallen below the pre-calculated safety margin, we cannot
        # afford any more risks (like another preemption or waiting for Spot).
        # We must run on On-Demand to guarantee completion.
        if current_slack <= self.safety_margin:
            return ClusterType.ON_DEMAND

        # RULE 2: Opportunistic Spot Usage
        # If we are not in the emergency mode and Spot instances are available,
        # we always choose them. They provide the most cost-effective progress.
        if has_spot:
            return ClusterType.SPOT
        
        # RULE 3: Patience vs. Progress Decision
        # At this point, we are not in emergency mode, but Spot is unavailable.
        # We must choose between waiting (NONE) and saving money, or running on
        # On-Demand (ON_DEMAND) to make progress at a higher cost.
        # The decision is based on how much of our "riskable" slack is left.
        
        slack_above_margin = current_slack - self.safety_margin
        normalized_risk_slack = slack_above_margin / self.risk_slack_range
        
        if normalized_risk_slack < self.WAIT_SLACK_THRESHOLD:
            # Our slack buffer is getting low. Our patience has run out.
            # It's now more important to make progress than to save money by waiting.
            # We proactively switch to On-Demand.
            return ClusterType.ON_DEMAND
        else:
            # We still have a comfortable slack buffer. We can afford to be patient
            # and wait for the cheaper Spot instances to become available again,
            # saving costs in the process.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        """
        Required classmethod for the evaluator to instantiate the class.
        """
        args, _ = parser.parse_known_args()
        return cls(args)