from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "adaptive_buffer_strategy"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        Read spec_path for configuration if needed.
        Must return self.
        """
        if not hasattr(self, 'initialized'):
            # --- State for adaptive logic ---
            # EMA for spot availability, starting with a neutral assumption.
            self.rolling_availability = 0.5
            # EMA smoothing factor. 1/0.005 = 200 steps memory.
            self.ema_alpha = 0.005

            # --- Tunable Hyperparameters for safety buffer ---
            # Multiplier for how many "average wait times" for Spot to keep as a buffer.
            self.buffer_wait_time_multiplier = 3.0
            # Multiplier for how many restart overheads to keep as a fixed base buffer.
            self.buffer_overhead_multiplier = 2.0

            self.initialized = True
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
        if not hasattr(self, 'initialized'):
            self.solve("")

        # --- 1. UPDATE STATE & CALCULATE METRICS ---
        current_availability_signal = 1.0 if has_spot else 0.0
        self.rolling_availability = (self.ema_alpha * current_availability_signal) + \
                                    ((1 - self.ema_alpha) * self.rolling_availability)

        work_done = sum(t[1] - t[0] for t in self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 1e-9:
            return ClusterType.NONE

        current_time = self.env.elapsed_seconds
        time_left_to_deadline = self.deadline - current_time

        # --- 2. DYNAMICALLY CALCULATE SAFETY BUFFER ---
        clipped_availability = max(0.01, min(0.99, self.rolling_availability))
        
        avg_unavailability_duration = ((1.0 - clipped_availability) / clipped_availability) * self.env.gap_seconds
        
        safety_buffer = (self.buffer_wait_time_multiplier * avg_unavailability_duration) + \
                        (self.buffer_overhead_multiplier * self.restart_overhead)

        # --- 3. CORE DECISION LOGIC ---
        time_needed_on_demand = work_remaining
        time_needed_with_one_preemption = work_remaining + self.restart_overhead

        if has_spot:
            if time_left_to_deadline <= time_needed_with_one_preemption:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.SPOT
        else:
            effective_slack = time_left_to_deadline - time_needed_on_demand
            if effective_slack <= safety_buffer:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):  # REQUIRED: For evaluator instantiation
        """
        Instantiates the strategy from command-line arguments.
        """
        args, _ = parser.parse_known_args()
        return cls(args)