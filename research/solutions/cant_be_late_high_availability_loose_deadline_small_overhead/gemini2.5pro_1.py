import sys
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initializes the strategy's state variables and hyperparameters.
        """
        # State variables for online estimation
        self.total_steps = 0
        self.spot_avail_count = 0
        self.spot_chosen_count = 0.0
        self.spot_work_done = 0.0
        self.last_work_total = 0.0

        # --- Hyperparameters ---
        
        # Bayesian prior for Spot availability probability (p_spot_avail).
        # Using a Beta(a, b) distribution prior.
        # An optimistic prior mean encourages initial Spot usage for cost savings.
        prior_mean_avail = 0.75
        prior_strength_avail = 5.0
        self.prior_a = prior_mean_avail * prior_strength_avail
        self.prior_b = (1.0 - prior_mean_avail) * prior_strength_avail

        # Prior for effective Spot speed (accounts for preemptions).
        # Modeled as `N` virtual steps of perfect execution (speed = 1.0).
        self.prior_speed_N = 5.0

        # Safety buffer for the hard backstop rule, in units of restart_overhead.
        self.hard_buffer_factor = 1.0

        # Safety buffer for the main projection-based decision rule.
        self.main_buffer_factor = 3.0

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Makes a decision at each time step based on a projection model.
        """
        # 1. Update internal state based on the outcome of the previous step.
        work_total = sum(self.task_done_time)
        work_delta = work_total - self.last_work_total

        if last_cluster_type == ClusterType.SPOT:
            self.spot_chosen_count += 1
            self.spot_work_done += work_delta
        
        self.total_steps += 1
        if has_spot:
            self.spot_avail_count += 1
            
        # 2. Assess current job status.
        work_remaining = self.task_duration - work_total
        if work_remaining <= 1e-6:  # Job is done (with float tolerance)
            self.last_work_total = work_total
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds

        # 3. Hard Backstop: A non-negotiable switch to On-Demand if completion is
        #    impossible otherwise, preventing failure in extreme cases.
        if work_remaining + self.hard_buffer_factor * self.restart_overhead >= time_to_deadline:
            self.last_work_total = work_total
            return ClusterType.ON_DEMAND

        # 4. Online Estimation: Update model of the environment.
        # Estimate p_spot_avail using a Bayesian update with the Beta prior.
        p_spot_avail = (self.spot_avail_count + self.prior_a) / \
                       (self.total_steps + self.prior_a + self.prior_b)

        # Estimate effective_spot_speed, adding a prior for stability.
        T = self.env.gap_seconds
        total_spot_time_in_gaps = self.spot_chosen_count + self.prior_speed_N
        total_spot_work_with_prior = self.spot_work_done + self.prior_speed_N * T

        effective_spot_speed = total_spot_work_with_prior / (total_spot_time_in_gaps * T)
        effective_spot_speed = max(0.01, min(1.0, effective_spot_speed))

        # 5. Projection: Estimate time to finish under the "Spot-first" policy.
        try:
            spot_state_time_needed = work_remaining / effective_spot_speed
            est_time_to_finish = spot_state_time_needed / p_spot_avail
        except (ZeroDivisionError, OverflowError):
            est_time_to_finish = float('inf')

        # 6. Main Adaptive Decision Rule.
        safety_buffer = self.main_buffer_factor * self.restart_overhead
        
        if est_time_to_finish + safety_buffer >= time_to_deadline:
            # Projection indicates a risk of missing the deadline. Switch to On-Demand.
            choice = ClusterType.ON_DEMAND
        else:
            # Projection shows sufficient slack. Pursue cost-effective Spot-first policy.
            if has_spot:
                choice = ClusterType.SPOT
            else:
                choice = ClusterType.NONE  # Wait for Spot.
        
        # 7. Finalize and return.
        self.last_work_total = work_total
        return choice

    @classmethod
    def _from_args(cls, parser):
        """Required method for evaluator instantiation."""
        args, _ = parser.parse_known_args()
        return cls(args)