from typing import Any, List, Tuple, Union

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cbt_spot_robust_v1"

    def __init__(self, args: Any):
        super().__init__(args)
        self._force_on_demand: bool = False
        self._cached_done_time: float = 0.0
        self._cached_segments_count: int = 0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _estimate_remaining_work(self) -> float:
        total_duration = float(self.task_duration)

        segments: List[Union[float, Tuple[float, float], List[float]]] = getattr(
            self, "task_done_time", []
        )
        if segments is None:
            segments = []

        # Reset cache if the underlying list shrank (shouldn't normally happen,
        # but guard against any environment quirks).
        if len(segments) < self._cached_segments_count:
            self._cached_segments_count = 0
            self._cached_done_time = 0.0

        # Accumulate only new segments since last call.
        for i in range(self._cached_segments_count, len(segments)):
            seg = segments[i]
            if seg is None:
                continue
            try:
                if isinstance(seg, (list, tuple)):
                    if len(seg) >= 2:
                        start = float(seg[0])
                        end = float(seg[1])
                        if end > start:
                            self._cached_done_time += end - start
                    elif len(seg) == 1:
                        dur = float(seg[0])
                        if dur > 0.0:
                            self._cached_done_time += dur
                else:
                    dur = float(seg)
                    if dur > 0.0:
                        self._cached_done_time += dur
            except Exception:
                # Ignore malformed entries defensively.
                continue

        self._cached_segments_count = len(segments)

        remaining = total_duration - self._cached_done_time
        if remaining < 0.0:
            remaining = 0.0
        return remaining

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If we've already committed to on-demand, always continue with it.
        if self._force_on_demand:
            return ClusterType.ON_DEMAND

        remaining_work = self._estimate_remaining_work()

        # If task already completed, no need to run anything.
        if remaining_work <= 0.0:
            return ClusterType.NONE

        current_time = float(getattr(self.env, "elapsed_seconds", 0.0))
        gap = float(getattr(self.env, "gap_seconds", 0.0))
        deadline = float(self.deadline)
        restart_overhead = float(self.restart_overhead)

        time_left = deadline - current_time

        # If somehow at/after deadline, still attempt on-demand to salvage.
        if time_left <= 0.0:
            self._force_on_demand = True
            return ClusterType.ON_DEMAND

        if gap < 0.0:
            gap = 0.0

        # Safe to "gamble" one more gap interval (using SPOT or idling)
        # if, even after losing up to one full gap of time with no progress,
        # we can still finish on on-demand before the deadline:
        #   remaining_work + restart_overhead <= time_left - gap
        safe_to_gamble = (remaining_work + restart_overhead) <= (time_left - gap)

        if not safe_to_gamble:
            # Need to commit to on-demand now to guarantee finishing.
            self._force_on_demand = True
            return ClusterType.ON_DEMAND

        # Still safe: prefer cheap spot when available, else wait.
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)