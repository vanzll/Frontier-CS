from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "latest_safe_start_od_v1"

    def __init__(self, args=None):
        super().__init__(args)
        self._step_count = 0
        self._spot_avail_count = 0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _sum_completed(self):
        total = 0.0
        try:
            segments = getattr(self, "task_done_time", []) or []
            for seg in segments:
                if isinstance(seg, (int, float)):
                    total += float(seg)
                elif isinstance(seg, dict):
                    if "duration" in seg:
                        total += float(seg["duration"])
                    elif "start" in seg and "end" in seg:
                        total += float(seg["end"] - seg["start"])
                elif isinstance(seg, (list, tuple)):
                    if len(seg) >= 2 and all(isinstance(x, (int, float)) for x in seg[:2]):
                        total += float(seg[1] - seg[0])
                    elif len(seg) == 1 and isinstance(seg[0], (int, float)):
                        total += float(seg[0])
        except Exception:
            pass
        return max(total, 0.0)

    def _remaining_time(self):
        done = self._sum_completed()
        return max(float(self.task_duration) - done, 0.0)

    def _time_left(self):
        return float(self.deadline) - float(self.env.elapsed_seconds)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._step_count += 1
        if has_spot:
            self._spot_avail_count += 1

        gap = float(self.env.gap_seconds) if getattr(self.env, "gap_seconds", None) is not None else 0.0
        overhead = float(self.restart_overhead) if getattr(self, "restart_overhead", None) is not None else 0.0
        remaining = self._remaining_time()
        time_left = self._time_left()

        if remaining <= 0:
            return ClusterType.NONE

        # If already on on-demand, decide whether to keep or switch
        if last_cluster_type == ClusterType.ON_DEMAND:
            # Prefer to keep OD unless we have ample slack to risk a step on SPOT
            # Safe to risk one SPOT (or idle) step if even after losing next step,
            # we can start OD and finish: time_left - gap >= remaining + overhead
            # Add hysteresis: require 2*gap slack to reduce thrashing
            if has_spot:
                if time_left - 2.0 * gap >= remaining + overhead:
                    return ClusterType.SPOT
            # If spot unavailable, keep OD (avoid idle thrash)
            return ClusterType.ON_DEMAND

        # Not currently on OD: decide between SPOT / OD / NONE
        # Latest-safe-start rule: If waiting one more step could make deadline infeasible
        # (in worst case of zero progress), then start OD now.
        must_start_od_now = (time_left - gap) < (remaining + overhead)

        if must_start_od_now:
            return ClusterType.ON_DEMAND

        # Otherwise, we can afford to risk this step.
        if has_spot:
            return ClusterType.SPOT

        # Spot not available and safe to wait -> pause
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)