import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "lods_safety_v1"

    def __init__(self, args=None):
        super().__init__(args)
        # Tunable parameters
        self.buffer_overhead_mult = getattr(args, "buffer_overhead_mult", 1.0) if args else 1.0
        self.buffer_gap_mult = getattr(args, "buffer_gap_mult", 2.0) if args else 2.0
        self.min_buffer_seconds = getattr(args, "min_buffer_seconds", 90.0) if args else 90.0

        self._last_choice = None
        self._spot_seen = 0
        self._steps_seen = 0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _safety_buffer(self) -> float:
        g = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        o = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        buffer_sec = self.buffer_overhead_mult * o + self.buffer_gap_mult * g
        return max(buffer_sec, self.min_buffer_seconds, o)

    def _remaining_work(self) -> float:
        try:
            if isinstance(self.task_done_time, (list, tuple)):
                done = float(sum(self.task_done_time))
            else:
                done = float(self.task_done_time)
        except Exception:
            done = 0.0
        total = float(self.task_duration or 0.0)
        return max(0.0, total - done)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._steps_seen += 1
        if has_spot:
            self._spot_seen += 1

        R = self._remaining_work()
        if R <= 0.0:
            self._last_choice = ClusterType.NONE
            return ClusterType.NONE

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        T = max(0.0, deadline - elapsed)

        buffer_sec = self._safety_buffer()
        slack = T - R

        if slack <= buffer_sec:
            choice = ClusterType.ON_DEMAND
        else:
            if has_spot:
                choice = ClusterType.SPOT
            else:
                choice = ClusterType.NONE

        self._last_choice = choice
        return choice

    @classmethod
    def _from_args(cls, parser):
        parser.add_argument("--buffer_overhead_mult", type=float, default=1.0)
        parser.add_argument("--buffer_gap_mult", type=float, default=2.0)
        parser.add_argument("--min_buffer_seconds", type=float, default=90.0)
        args, _ = parser.parse_known_args()
        return cls(args)