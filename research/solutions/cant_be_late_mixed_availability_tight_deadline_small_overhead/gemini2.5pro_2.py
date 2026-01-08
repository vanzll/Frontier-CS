import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initializes the strategy's parameters. Called once before evaluation.
        """
        # EMA parameter for tracking spot availability. Alpha corresponds to a
        # look-back window of roughly 2/alpha steps. N=39 -> alpha~0.05
        self.ema_availability = 0.5  # Start with a neutral assumption
        self.alpha = 0.05

        # --- Parameters for the adaptive wait threshold ---
        # We define a linear mapping from observed availability to the slack
        # threshold (T_wait) at which we stop waiting for Spot.
        
        # Define the expected range of spot availability from problem description.
        avail_min = 0.04  # Low-availability traces
        avail_max = 0.80  # High-availability traces
        
        # Define the corresponding T_wait values.
        # If availability is low, T_wait is high (switch to OD early).
        # If availability is high, T_wait is low (be patient and wait for Spot).
        # Values are chosen relative to the total initial slack of 4 hours.
        T_wait_at_avail_min = 2.5 * 3600  # 2.5 hours
        T_wait_at_avail_max = 0.5 * 3600  # 30 minutes

        # Pre-calculate the slope and intercept for: T_wait = m*avail + c
        self.slope = (T_wait_at_avail_max - T_wait_at_avail_min) / (avail_max - avail_min)
        self.intercept = T_wait_at_avail_max - self.slope * avail_max
        
        # Store bounds for clamping the calculated T_wait value.
        self.T_wait_bounds = tuple(sorted((T_wait_at_avail_min, T_wait_at_avail_max)))

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step to decide which cluster type to use.
        """
        # 1. Update internal metrics
        # Update the EMA of spot availability based on the current observation.
        current_availability_signal = 1.0 if has_spot else 0.0
        self.ema_availability = self.alpha * current_availability_signal + \
                                (1 - self.alpha) * self.ema_availability

        # 2. Calculate current job state
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        # If the job is finished, do nothing to save cost.
        if work_remaining <= 0:
            return ClusterType.NONE

        time_left_to_deadline = self.deadline - self.env.elapsed_seconds
        
        # 3. Core Decision Logic

        # A) CRITICAL SAFETY CHECK
        # If using Spot and getting preempted *once* would cause a deadline miss,
        # we MUST use On-Demand. This is the "Cant-Be-Late" guarantee.
        # We add a buffer of one `gap_seconds` for added safety.
        time_needed_with_one_preemption = work_remaining + self.restart_overhead
        if time_left_to_deadline <= time_needed_with_one_preemption + self.env.gap_seconds:
            return ClusterType.ON_DEMAND

        # B) PREFER SPOT
        # If Spot is available and it's safe to use it (passed check A), always
        # choose it for maximum cost savings.
        if has_spot:
            return ClusterType.SPOT

        # C) SPOT UNAVAILABLE: DECIDE BETWEEN ON-DEMAND AND WAITING (NONE)
        # The choice depends on how much slack we have compared to our adaptive threshold.
        
        # Calculate the adaptive wait threshold for the current EMA of availability.
        T_wait = self.slope * self.ema_availability + self.intercept
        # Ensure the threshold stays within our pre-defined safe bounds.
        T_wait = max(self.T_wait_bounds[0], min(self.T_wait_bounds[1], T_wait))

        # Slack if we were to use On-Demand for the rest of the job.
        on_demand_slack = time_left_to_deadline - work_remaining
        
        if on_demand_slack > T_wait:
            # We have more slack than our current threshold, so we can afford to
            # wait for Spot to become available again.
            return ClusterType.NONE
        else:
            # Our slack is below the threshold. It's too risky to wait.
            # We must make progress using On-Demand.
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        """Required method for evaluator instantiation."""
        args, _ = parser.parse_known_args()
        return cls(args)