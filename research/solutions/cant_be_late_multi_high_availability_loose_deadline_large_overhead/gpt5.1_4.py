import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy using conservative Spot with On-Demand fallback."""

    NAME = "cant_be_late_threshold_v1"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config.

        The spec file contains:
        - deadline: deadline in hours
        - duration: task duration in hours
        - overhead: restart overhead in hours
        - trace_files: list of trace file paths (one per region)
        """
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        # Cached cumulative work done (seconds) to avoid O(N^2) summations.
        self._cached_task_done_total = 0.0
        self._cached_task_done_len = 0

        # Once set, we always run on On-Demand until completion.
        self._commit_to_on_demand = False

        return self

    def _update_task_progress_cache(self) -> None:
        """Incrementally update cached total of task_done_time."""
        segments = self.task_done_time
        cur_len = len(segments)
        if cur_len > self._cached_task_done_len:
            # Sum only newly added segments.
            self._cached_task_done_total += sum(
                segments[self._cached_task_done_len:cur_len]
            )
            self._cached_task_done_len = cur_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.

        Strategy:
        - Use Spot whenever it's available and we still have enough slack to
          safely tolerate one more potential preemption (including overhead).
        - If slack becomes tight, irrevocably switch to On-Demand to avoid
          missing the hard deadline.
        - If Spot is not available and we are still in the "safe" region,
          wait (NONE) instead of paying for On-Demand.
        """

        # Update cached work progress.
        self._update_task_progress_cache()
        remaining_work = self.task_duration - self._cached_task_done_total

        # If job is effectively done, don't run any more clusters.
        if remaining_work <= 1e-6:
            return ClusterType.NONE

        # If already committed to On-Demand, keep using it to avoid extra restarts.
        if self._commit_to_on_demand:
            return ClusterType.ON_DEMAND

        env = self.env
        t = getattr(env, "elapsed_seconds", 0.0)
        deadline = self.deadline
        gap = getattr(env, "gap_seconds", 0.0)
        restart_overhead = self.restart_overhead

        # Degenerate fallback if gap is missing or zero.
        if gap <= 0.0:
            time_left = deadline - t
            # If not enough time left to safely rely on Spot, use On-Demand.
            if time_left <= remaining_work + restart_overhead:
                self._commit_to_on_demand = True
                return ClusterType.ON_DEMAND
            # Otherwise, Spot when available, else wait.
            if has_spot:
                return ClusterType.SPOT
            return ClusterType.NONE

        # Conservative buffer accounting for:
        # - Up to one full step of lost compute time (gap),
        # - Existing restart overhead that might still be owed,
        # - One additional restart overhead in case of another preemption.
        safety_buffer = gap + 2.0 * restart_overhead

        # Latest time at which it's still safe to risk one more Spot step.
        t_safe_spot = deadline - (safety_buffer + remaining_work)

        # If current time has passed the safe point, switch to On-Demand.
        if t > t_safe_spot:
            self._commit_to_on_demand = True
            return ClusterType.ON_DEMAND

        # Before commit point: prefer cheap Spot when available; otherwise, wait.
        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE