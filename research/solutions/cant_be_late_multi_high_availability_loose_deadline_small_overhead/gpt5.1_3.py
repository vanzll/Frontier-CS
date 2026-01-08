import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy with hard-deadline safety."""

    NAME = "my_strategy"

    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        # Cache constants (in seconds)
        self._gap = float(getattr(self.env, "gap_seconds", 0.0))

        overhead = getattr(self, "restart_overhead", 0.0)
        if isinstance(overhead, (list, tuple)):
            overhead = overhead[0]
        self._overhead = float(overhead)

        task_duration = getattr(self, "task_duration", 0.0)
        if isinstance(task_duration, (list, tuple)):
            task_duration = task_duration[0]
        self._task_duration = float(task_duration)

        deadline = getattr(self, "deadline", 0.0)
        if isinstance(deadline, (list, tuple)):
            deadline = deadline[0]
        self._deadline = float(deadline)

        # Tracking accumulated work efficiently
        self._work_done = 0.0
        self._last_seen_segments_len = 0
        if hasattr(self, "task_done_time"):
            self._last_seen_segments_len = len(self.task_done_time)
            if self._last_seen_segments_len > 0:
                self._work_done = float(sum(self.task_done_time))

        # Once we commit to on-demand for deadline safety, never go back.
        self._locked_on_demand = False

        return self

    def _update_work_done(self) -> None:
        """Incrementally track total work done."""
        segments = self.task_done_time
        current_len = len(segments)
        if current_len > self._last_seen_segments_len:
            new_work = sum(segments[self._last_seen_segments_len:current_len])
            self._work_done += float(new_work)
            self._last_seen_segments_len = current_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update cached work done
        self._update_work_done()

        # If task already complete, do nothing.
        remaining_work = self._task_duration - self._work_done
        if remaining_work <= 0.0:
            return ClusterType.NONE

        elapsed = float(self.env.elapsed_seconds)
        time_left = self._deadline - elapsed
        if time_left <= 0.0:
            # Deadline already missed in simulator; keep running on-demand.
            self._locked_on_demand = True
            return ClusterType.ON_DEMAND

        gap = self._gap
        overhead = self._overhead

        # If we've already switched to on-demand for safety, keep it.
        if self._locked_on_demand:
            return ClusterType.ON_DEMAND

        # Safety thresholds (seconds):
        # - For SPOT: ensure that even with worst-case SPOT step (gap + overhead,
        #   no progress), we can still later finish purely on on-demand with
        #   overhead + gap discretization penalty.
        safe_spot_threshold = remaining_work + 2.0 * (gap + overhead)

        # - For NONE: worst-case degradation is just one gap of idle time.
        safe_none_threshold = remaining_work + (2.0 * gap + overhead)

        # Prefer SPOT whenever safely possible.
        if has_spot and time_left > safe_spot_threshold:
            return ClusterType.SPOT

        # If SPOT is unavailable, we can afford to wait (NONE) while still
        # preserving a guaranteed on-demand completion window.
        if (not has_spot) and time_left > safe_none_threshold:
            return ClusterType.NONE

        # Otherwise, we must commit to ON_DEMAND to guarantee deadline.
        self._locked_on_demand = True
        return ClusterType.ON_DEMAND