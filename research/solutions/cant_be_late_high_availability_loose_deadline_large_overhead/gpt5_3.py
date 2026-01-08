import math
from typing import Any, Iterable, Tuple
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cblo_v1"

    def __init__(self, args=None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass
        self.args = args
        self.locked_to_on_demand = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _sum_done_time(self, segments: Any) -> float:
        if segments is None:
            return 0.0
        # If it's a scalar
        if isinstance(segments, (int, float)):
            return float(segments)
        total = 0.0
        try:
            for seg in segments:  # type: ignore
                if isinstance(seg, (int, float)):
                    total += float(seg)
                elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
                    a, b = seg[0], seg[1]
                    try:
                        total += max(0.0, float(b) - float(a))
                    except Exception:
                        # Fallback: try to cast seg directly
                        try:
                            total += float(seg)  # type: ignore
                        except Exception:
                            continue
                else:
                    try:
                        total += float(seg)  # type: ignore
                    except Exception:
                        continue
        except Exception:
            try:
                return float(segments)  # type: ignore
            except Exception:
                return 0.0
        return total

    def _remaining_work(self) -> float:
        done = self._sum_done_time(self.task_done_time)
        remaining = float(self.task_duration) - float(done)
        return max(0.0, remaining)

    def _safe_margin(self) -> float:
        # Add a small margin to account for restart overhead and discretization
        dt = float(getattr(self.env, "gap_seconds", 300.0))
        oh = float(getattr(self, "restart_overhead", 0.0))
        fudge_min = 120.0  # at least 2 minutes to guard discretization
        return max(0.0, oh) + max(2.0 * dt, fudge_min)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Ensure internal state
        if not hasattr(self, "locked_to_on_demand"):
            self.locked_to_on_demand = False

        remaining_work = self._remaining_work()
        if remaining_work <= 0.0:
            return ClusterType.NONE

        t_now = float(self.env.elapsed_seconds)
        deadline = float(self.deadline)
        time_left = deadline - t_now

        # If time already passed, force on-demand
        if time_left <= 0.0:
            self.locked_to_on_demand = True

        # Determine if we must commit to on-demand right now
        margin = self._safe_margin()
        # If we are already on-demand, switching continues without overhead
        # but for safety we keep using the same rule to maintain lock
        if time_left <= remaining_work + margin:
            self.locked_to_on_demand = True

        if self.locked_to_on_demand:
            return ClusterType.ON_DEMAND

        # If not locked to on-demand:
        if has_spot:
            return ClusterType.SPOT
        else:
            # Wait for spot if we still have slack above threshold
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)