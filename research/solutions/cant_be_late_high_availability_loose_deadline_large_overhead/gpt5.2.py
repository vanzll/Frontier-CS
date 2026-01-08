import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except Exception:
            try:
                super().__init__()
            except Exception:
                pass
        self.args = args
        self._committed_to_od = False
        self._last_elapsed = None

    def solve(self, spec_path: str) -> "Solution":
        self._committed_to_od = False
        self._last_elapsed = None
        return self

    def _reset_if_needed(self) -> None:
        t_now = float(getattr(getattr(self, "env", None), "elapsed_seconds", 0.0) or 0.0)
        if self._last_elapsed is None or t_now < self._last_elapsed or t_now <= 0.0:
            self._committed_to_od = False
        self._last_elapsed = t_now

    def _work_done_seconds(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            return 0.0

        if isinstance(tdt, (int, float)):
            return float(tdt)

        if not isinstance(tdt, (list, tuple)):
            return 0.0

        numeric = True
        numeric_vals = []
        for x in tdt:
            if isinstance(x, (int, float)):
                numeric_vals.append(float(x))
            else:
                numeric = False
                break

        if numeric:
            if len(numeric_vals) == 0:
                return 0.0
            s = sum(numeric_vals)
            if len(numeric_vals) >= 2:
                mono = True
                prev = numeric_vals[0]
                for v in numeric_vals[1:]:
                    if v + 1e-9 < prev:
                        mono = False
                        break
                    prev = v
                last = numeric_vals[-1]
                task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
                if mono and last <= task_duration * 1.10 and s > last * 1.20:
                    return last
            return s

        total = 0.0
        for x in tdt:
            if isinstance(x, (int, float)):
                total += float(x)
            elif isinstance(x, (tuple, list)):
                if len(x) >= 2 and isinstance(x[0], (int, float)) and isinstance(x[1], (int, float)):
                    total += max(0.0, float(x[1]) - float(x[0]))
                elif len(x) == 1 and isinstance(x[0], (int, float)):
                    total += float(x[0])
            elif isinstance(x, dict):
                if "duration" in x and isinstance(x["duration"], (int, float)):
                    total += float(x["duration"])
                elif "start" in x and "end" in x and isinstance(x["start"], (int, float)) and isinstance(x["end"], (int, float)):
                    total += max(0.0, float(x["end"]) - float(x["start"]))
        return total

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._reset_if_needed()

        env = getattr(self, "env", None)
        t_now = float(getattr(env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(env, "gap_seconds", 0.0) or 0.0)

        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        restart_overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        done = self._work_done_seconds()
        if done < 0.0:
            done = 0.0
        if done > task_duration:
            done = task_duration

        w_left = task_duration - done
        if w_left <= 1e-9:
            self._committed_to_od = False
            return ClusterType.NONE

        t_left = deadline - t_now

        if t_left <= 0.0:
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        slack = t_left - w_left

        if self._committed_to_od:
            return ClusterType.ON_DEMAND

        reserve_spot = max(1800.0, restart_overhead + 4.0 * gap)
        reserve_none = max(7200.0, restart_overhead + 6.0 * gap)

        hard_min = restart_overhead + 2.0 * gap
        if slack <= hard_min:
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        if has_spot:
            if slack <= reserve_spot:
                self._committed_to_od = True
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        if slack <= reserve_none:
            self._committed_to_od = True
            return ClusterType.ON_DEMAND
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)