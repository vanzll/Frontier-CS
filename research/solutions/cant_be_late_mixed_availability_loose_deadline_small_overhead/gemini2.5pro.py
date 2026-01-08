import collections
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        Read spec_path for configuration if needed.
        Must return self.
        """
        # --- Strategy Parameters ---
        # The size of the sliding window to estimate recent spot availability.
        self.HISTORY_WINDOW_SIZE = 500
        # The initial guess for spot availability before any history is collected.
        self.INITIAL_SPOT_AVAILABILITY_ESTIMATE = 0.5

        # --- State Variables ---
        self.spot_history = collections.deque(maxlen=self.HISTORY_WINDOW_SIZE)
        self.work_done_cache = 0.0
        self.last_task_done_len = 0
        
        return self

    def _get_work_done(self) -> float:
        """
        Efficiently calculates the total work done by caching the sum.
        """
        if len(self.task_done_time) > self.last_task_done_len:
            new_segments = self.task_done_time[self.last_task_done_len:]
            self.work_done_cache += sum(end - start for start, end in new_segments)
            self.last_task_done_len = len(self.task_done_time)
        return self.work_done_cache

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step. Return which cluster type to use next.

        Args:
            last_cluster_type: The cluster type used in the previous step
            has_spot: Whether spot instances are available this step

        Returns:
            ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        # 1. Update state from the environment
        self.spot_history.append(1 if has_spot else 0)
        
        work_done = self._get_work_done()
        work_remaining = self.task_duration - work_done

        # 2. Handle job completion
        if work_remaining <= 0:
            return ClusterType.NONE

        # 3. Core decision logic based on a safety buffer
        current_time = self.env.elapsed_seconds
        
        # The safety buffer is the time slack available compared to a pure on-demand run.
        time_to_deadline = self.deadline - current_time
        safety_buffer = time_to_deadline - work_remaining
        
        # The critical buffer is the minimum slack required to survive one spot preemption.
        # A preemption costs time (gap_seconds) and adds work (restart_overhead).
        critical_buffer = self.restart_overhead + self.env.gap_seconds

        # Decision A: Deadline is at risk.
        # If slack is insufficient to cover a preemption, use reliable on-demand.
        if safety_buffer <= critical_buffer:
            return ClusterType.ON_DEMAND

        # Decision B: Safe to use SPOT.
        # If spot is available and there's enough buffer, it's the cheapest option.
        if has_spot:
            return ClusterType.SPOT
        
        # Decision C: SPOT is unavailable. Decide whether to wait (NONE) or use ON_DEMAND.
        
        # Estimate recent spot availability from history.
        if len(self.spot_history) > 0:
            estimated_availability = sum(self.spot_history) / len(self.spot_history)
        else:
            estimated_availability = self.INITIAL_SPOT_AVAILABILITY_ESTIMATE
            
        # Determine if it's worth waiting based on availability and remaining slack.
        # The threshold is the minimum availability needed to justify waiting.
        affordable_slack = safety_buffer - critical_buffer
        
        # Add a small epsilon for floating point stability.
        wait_patience_threshold = self.env.gap_seconds / (affordable_slack + self.env.gap_seconds + 1e-9)

        if estimated_availability >= wait_patience_threshold:
            # Estimated availability is high enough to justify waiting.
            return ClusterType.NONE
        else:
            # Availability is too low to risk waiting; use on-demand to make progress.
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):  # REQUIRED: For evaluator instantiation
        args, _ = parser.parse_known_args()
        return cls(args)