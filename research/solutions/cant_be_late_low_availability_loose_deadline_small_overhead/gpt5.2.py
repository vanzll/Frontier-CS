import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args=None):
        super().__init__(args)
        self._force_od = False

        self._tdt_id: Optional[int] = None
        self._tdt_len: int = 0
        self._tdt_sum: float = 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _segment_value(self, seg: Any) -> float:
        if seg is None:
            return 0.0
        if isinstance(seg, (int, float)):
            v = float(seg)
            return v if math.isfinite(v) else 0.0
        if isinstance(seg, dict):
            if "duration" in seg and isinstance(seg["duration"], (int, float)):
                v = float(seg["duration"])
                return v if math.isfinite(v) else 0.0
            if "start" in seg and "end" in seg and isinstance(seg["start"], (int, float)) and isinstance(seg["end"], (int, float)):
                v = float(seg["end"]) - float(seg["start"])
                return v if math.isfinite(v) else 0.0
            return 0.0
        if isinstance(seg, (list, tuple)) and len(seg) == 2 and isinstance(seg[0], (int, float)) and isinstance(seg[1], (int, float)):
            v = float(seg[1]) - float(seg[0])
            return v if math.isfinite(v) else 0.0
        return 0.0

    def _get_done_seconds(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if tdt is None:
            return 0.0

        if isinstance(tdt, (int, float)):
            v = float(tdt)
            return v if math.isfinite(v) else 0.0

        if not isinstance(tdt, (list, tuple)):
            try:
                v = float(tdt)
                return v if math.isfinite(v) else 0.0
            except Exception:
                return 0.0

        cur_id = id(tdt)
        cur_len = len(tdt)

        if self._tdt_id != cur_id:
            self._tdt_id = cur_id
            self._tdt_len = 0
            self._tdt_sum = 0.0

        if cur_len < self._tdt_len:
            self._tdt_len = 0
            self._tdt_sum = 0.0

        for i in range(self._tdt_len, cur_len):
            self._tdt_sum += self._segment_value(tdt[i])

        self._tdt_len = cur_len
        if not math.isfinite(self._tdt_sum):
            self._tdt_sum = 0.0
        return self._tdt_sum

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        done = self._get_done_seconds()
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)

        remaining_work = task_duration - done
        if remaining_work <= 1e-6:
            return ClusterType.NONE

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        restart_overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        safety = max(2.0 * restart_overhead + 2.0 * gap, 3.0 * gap) + 60.0
        latest_od_start = deadline - (remaining_work + safety)

        if self._force_od or elapsed >= latest_od_start:
            self._force_od = True
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)