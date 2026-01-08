import collections
import argparse

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "adaptive_slack_manager"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        Read spec_path for configuration if needed.
        Must return self.
        """
        # --- Hyperparameters for Adaptive Thresholds ---

        # Window size for the moving average of spot availability.
        self.HISTORY_WINDOW_SIZE = 200

        # Base thresholds for the DANGER ZONE in seconds. This is the "panic mode"
        # buffer to ensure completion. MIN is for 100% spot availability, MAX for 0%.
        self.MIN_DANGER_S = 2 * 3600  # 2 hours
        self.MAX_DANGER_S = 9 * 3600  # 9 hours

        # Base thresholds for the COMFORT ZONE in seconds. This is the buffer
        # for waiting for Spot. MIN is for 100% spot availability, MAX for 0%.
        self.MIN_COMFORT_S = 4 * 3600  # 4 hours
        self.MAX_COMFORT_S = 18 * 3600 # 18 hours

        # --- Internal State Variables ---

        # A deque to store the recent history of spot availability (1s and 0s).
        self.spot_history = collections.deque(maxlen=self.HISTORY_WINDOW_SIZE)
        
        # Start with a pessimistic assumption about spot availability.
        self.spot_availability_rate = 0.2

        # Cache for memoizing the total work done to avoid re-summing the list.
        self.work_done_cache = 0.0
        self.last_task_done_time_len = 0

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
        # 1. Update internal state.
        self._update_state(has_spot)

        # 2. Calculate key metrics.
        work_remaining = self.task_duration - self.work_done_cache
        if work_remaining <= 0:
            return ClusterType.NONE  # Job finished.

        time_now = self.env.elapsed_seconds
        time_to_deadline = self.deadline - time_now
        
        # Slack is the buffer we have if we were to run on On-Demand from now on.
        # slack = (time until deadline) - (time needed to finish on OD)
        slack = time_to_deadline - work_remaining

        # If slack is negative, we've already missed the deadline.
        # Run OD to minimize the failure.
        if slack < 0:
            return ClusterType.ON_DEMAND

        # 3. Calculate adaptive thresholds based on recent spot availability.
        # Low availability -> factor close to 1 -> higher thresholds (more cautious).
        # High availability -> factor close to 0 -> lower thresholds (more aggressive).
        adjustment_factor = 1.0 - self.spot_availability_rate
        danger_threshold = self.MIN_DANGER_S + adjustment_factor * (self.MAX_DANGER_S - self.MIN_DANGER_S)
        comfort_threshold = self.MIN_COMFORT_S + adjustment_factor * (self.MAX_COMFORT_S - self.MIN_COMFORT_S)

        # 4. Apply the zone-based decision logic.
        if slack <= danger_threshold:
            # DANGER ZONE: Critically low slack. Must use On-Demand.
            return ClusterType.ON_DEMAND
        elif slack <= comfort_threshold:
            # CAUTION ZONE: Slack is okay. Prefer Spot, but use On-Demand if needed.
            return ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND
        else: # slack > comfort_threshold
            # COMFORT ZONE: Large slack buffer. Wait for Spot to save money.
            return ClusterType.SPOT if has_spot else ClusterType.NONE

    def _update_state(self, has_spot: bool) -> None:
        """Helper function to update the strategy's internal state."""
        # Update the moving average of spot availability.
        self.spot_history.append(1 if has_spot else 0)
        if len(self.spot_history) > 0:
            self.spot_availability_rate = sum(self.spot_history) / len(self.spot_history)

        # Efficiently update the total work done.
        if len(self.task_done_time) > self.last_task_done_time_len:
            # Sum only the new segments added since the last step.
            new_work_segments = self.task_done_time[self.last_task_done_time_len:]
            self.work_done_cache += sum(new_work_segments)
            self.last_task_done_time_len = len(self.task_done_time)

    @classmethod
    def _from_args(cls, parser):  # REQUIRED: For evaluator instantiation
        args, _ = parser.parse_known_args()
        return cls(args)