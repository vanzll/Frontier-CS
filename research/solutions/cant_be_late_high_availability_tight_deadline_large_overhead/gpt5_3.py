from typing import Any, Optional, Union, Iterable

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except Exception:
            # Base class may not require init args or may be initialized differently
            pass
        self._committed_to_od: bool = False
        self._last_choice: Optional[ClusterType] = None

    def solve(self, spec_path: str) -> "Solution":
        # No special initialization needed; return self for evaluator compatibility
        return self

    def _sum_done_seconds(self) -> float:
        # Robustly compute done seconds from self.task_done_time, which can be:
        # - a float/int
        # - a list of floats/ints
        # - a list of (start, end) tuples/lists
        # - objects with start/end attributes
        # Fallbacks to 0 if unknown format.
        x = getattr(self, "task_done_time", None)
        if x is None:
            x = getattr(self, "task_done", None)
        if x is None:
            x = getattr(self, "done_time", None)
        if x is None:
            return 0.0

        def seg_len(seg) -> float:
            try:
                if isinstance(seg, (int, float)):
                    return float(seg)
                if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                    a, b = seg[0], seg[1]
                    return float(b) - float(a)
                if hasattr(seg, "start") and hasattr(seg, "end"):
                    return float(seg.end) - float(seg.start)
                return float(seg)
            except Exception:
                return 0.0

        try:
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, Iterable):
                total = 0.0
                for seg in x:
                    total += seg_len(seg)
                return total
        except Exception:
            pass
        return 0.0

    def _remaining_compute_seconds(self) -> float:
        try:
            total = float(self.task_duration)
        except Exception:
            total = float(getattr(self, "task_duration", 0.0))
        done = self._sum_done_seconds()
        rem = total - done
        if rem < 0:
            return 0.0
        return rem

    def _get_gap_seconds(self) -> float:
        try:
            gap = float(self.env.gap_seconds)
            if gap > 0:
                return gap
        except Exception:
            pass
        # Fallback default step size if unavailable
        return 60.0

    def _get_overhead_seconds(self) -> float:
        try:
            oh = float(self.restart_overhead)
            if oh >= 0:
                return oh
        except Exception:
            pass
        # Fallback default overhead if unavailable (12 minutes)
        return 12 * 60.0

    def _get_elapsed_seconds(self) -> float:
        try:
            return float(self.env.elapsed_seconds)
        except Exception:
            return 0.0

    def _get_deadline_seconds(self) -> float:
        try:
            return float(self.deadline)
        except Exception:
            return self._get_elapsed_seconds()

    def _should_commit_to_od_now(
        self,
        last_cluster_type: ClusterType,
        remaining_compute: float,
        time_left: float,
        overhead: float,
        gap: float,
    ) -> bool:
        # If we are already on OD, switching has no restart overhead
        switch_overhead = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else overhead
        # Fudge to be slightly conservative due to discretization
        fudge = max(0.5 * gap, 1.0)
        required_time = remaining_compute + switch_overhead
        return time_left <= (required_time + fudge)

    def _safe_to_wait_one_step(
        self,
        remaining_compute: float,
        time_left: float,
        overhead: float,
        gap: float,
    ) -> bool:
        # If we wait one step doing NONE, the remaining compute is unchanged
        # and time_left reduces by gap. We must ensure that after waiting,
        # there is still time to finish on OD (including overhead).
        fudge = max(0.5 * gap, 1.0)
        return (time_left - gap) > (remaining_compute + overhead + fudge)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If we already committed to on-demand, keep it to avoid extra restarts.
        if self._committed_to_od:
            self._last_choice = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND

        gap = self._get_gap_seconds()
        overhead = self._get_overhead_seconds()
        elapsed = self._get_elapsed_seconds()
        deadline = self._get_deadline_seconds()
        time_left = max(0.0, deadline - elapsed)
        remaining = self._remaining_compute_seconds()

        if remaining <= 0.0:
            self._last_choice = ClusterType.NONE
            return ClusterType.NONE

        # Decide if we must commit to on-demand now to guarantee meeting the deadline
        if self._should_commit_to_od_now(last_cluster_type, remaining, time_left, overhead, gap):
            self._committed_to_od = True
            self._last_choice = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND

        # If spot is available, use it to reduce cost while we still have slack
        if has_spot:
            self._last_choice = ClusterType.SPOT
            return ClusterType.SPOT

        # Spot unavailable: decide to wait (NONE) or switch to on-demand
        if self._safe_to_wait_one_step(remaining, time_left, overhead, gap):
            self._last_choice = ClusterType.NONE
            return ClusterType.NONE

        # Not safe to wait; switch to on-demand and commit to finish
        self._committed_to_od = True
        self._last_choice = ClusterType.ON_DEMAND
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)