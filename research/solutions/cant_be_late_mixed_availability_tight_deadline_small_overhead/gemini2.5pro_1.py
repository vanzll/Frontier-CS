import collections
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    """
    An adaptive strategy that uses a moving window to estimate recent spot
    availability and decides whether to use On-Demand instances based on the
    progress required to meet the deadline.
    """
    NAME = "adaptive_moving_window"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the strategy's parameters and state. This is called once
        before the evaluation starts.
        """
        # --- Parameters for tuning ---

        # The size of the moving window for estimating spot availability.
        # A window of 100 steps, with each step being ~5 minutes, corresponds
        # to looking at the last ~8.3 hours of spot history.
        self.window_size = 100

        # The number of initial steps to use a simpler, more conservative
        # strategy before the moving window has enough data for a reliable estimate.
        self.burn_in_steps = 20

        # The buffer for the initial conservative strategy. If slack falls
        # below this, use On-Demand. 3600s = 1 hour.
        self.caution_buffer_initial = 3600.0

        # A safety multiplier for the critical buffer. We switch to permanent
        # On-Demand when remaining time is less than work_remaining +
        # restart_overhead * factor. A factor > 1 provides a safety margin.
        self.critical_buffer_factor = 2.0

        # A safety fudge factor for the adaptive logic. We use On-Demand if
        # required_progress_rate * factor > estimated_spot_availability.
        # A factor > 1 makes us switch to On-Demand more readily, preserving slack.
        self.required_rate_fudge_factor = 1.2

        # --- State tracking ---

        # Use a deque for an efficient moving window of spot availability history.
        self.history = collections.deque(maxlen=self.window_size)

        # A latch to switch to permanent On-Demand mode when the deadline is critical.
        self.on_demand_mode = False

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        This function is called at each time step to decide which cluster type to use.
        The logic is prioritized as follows:
        1. Safety: Check for task completion or critical deadline proximity.
        2. Cost: If spot is available and we're not in a critical state, always use it.
        3. Risk management: If spot is unavailable, use an adaptive model to decide
           between costly progress (On-Demand) and risky waiting (None).
        """
        # --- 1. Update state and get environment variables ---
        self.history.append(1 if has_spot else 0)

        work_done = sum(seg.duration for seg in self.task_done_time)
        if work_done >= self.task_duration:
            self.on_demand_mode = False
            return ClusterType.NONE

        work_remaining = self.task_duration - work_done
        current_time = self.env.elapsed_seconds
        time_to_deadline = self.deadline - current_time

        # --- 2. Handle the critical "point of no return" ---
        if self.on_demand_mode:
            return ClusterType.ON_DEMAND

        critical_buffer = self.restart_overhead * self.critical_buffer_factor
        if time_to_deadline <= work_remaining + critical_buffer:
            self.on_demand_mode = True
            return ClusterType.ON_DEMAND

        # --- 3. Default to cheap spot instances if available ---
        if has_spot:
            return ClusterType.SPOT

        # --- 4. Adaptive logic for when spot is unavailable ---
        num_steps_in_history = len(self.history)

        # Phase 1: Burn-in period with a simple heuristic
        if num_steps_in_history < self.burn_in_steps:
            slack = time_to_deadline - work_remaining
            if slack <= self.caution_buffer_initial:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.NONE
        
        # Phase 2: Adaptive logic using moving window estimate
        else:
            p_spot_est = sum(self.history) / num_steps_in_history
            
            if time_to_deadline <= 0: # Should be caught by critical check, but for safety
                return ClusterType.ON_DEMAND

            required_rate = work_remaining / time_to_deadline

            # If the required progress rate (with a safety factor) is greater than
            # what we can expect from spot, we must use On-Demand to catch up.
            if required_rate * self.required_rate_fudge_factor > p_spot_est:
                return ClusterType.ON_DEMAND
            else:
                # Our progress is on track, so we can afford to wait for cheaper Spot.
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        """
        Required classmethod for evaluator instantiation.
        """
        args, _ = parser.parse_known_args()
        return cls(args)