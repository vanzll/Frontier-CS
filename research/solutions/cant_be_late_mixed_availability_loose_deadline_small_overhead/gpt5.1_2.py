from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args):
        super().__init__(args)
        # Extra safety in units of gap_seconds to account for discretization/overheads.
        self._safety_extra_steps = 2

    def solve(self, spec_path: str) -> "Solution":
        # Optional initialization using spec_path if needed.
        return self

    def _total_task_done(self) -> float:
        tdt = self.task_done_time
        if tdt is None:
            return 0.0
        if isinstance(tdt, (int, float)):
            return float(tdt)
        total = 0.0
        for seg in tdt:
            if isinstance(seg, (list, tuple)):
                if len(seg) >= 2:
                    try:
                        total += float(seg[1]) - float(seg[0])
                    except Exception:
                        # Fallback: ignore malformed segment.
                        pass
                elif len(seg) == 1:
                    try:
                        total += float(seg[0])
                    except Exception:
                        pass
            else:
                try:
                    total += float(seg)
                except Exception:
                    pass
        return total

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        task_done = self._total_task_done()
        remaining = self.task_duration - task_done

        # If task already finished, no need to run more.
        if remaining <= 0:
            return ClusterType.NONE

        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        time_left = self.deadline - elapsed

        # If we're at or past the deadline, just run on-demand (will fail anyway).
        if time_left <= 0:
            return ClusterType.ON_DEMAND

        # Safety margin to ensure we can switch to on-demand and still finish.
        margin = self.restart_overhead + self._safety_extra_steps * gap

        # If we have plenty of slack beyond what pure on-demand would need (plus margin),
        # exploit spot when available, otherwise idle to save cost.
        if time_left > remaining + margin:
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.NONE

        # Not enough slack to risk more delay; use guaranteed on-demand capacity.
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)