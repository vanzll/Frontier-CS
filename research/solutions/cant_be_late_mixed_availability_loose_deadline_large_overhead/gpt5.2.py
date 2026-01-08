import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cbl_deadline_guard_v1"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass

        self.args = args
        self._committed_to_on_demand = False

        self._deadline_guard_steps = int(getattr(args, "deadline_guard_steps", 2) if args is not None else 2)
        self._deadline_guard_overhead_mult = float(
            getattr(args, "deadline_guard_overhead_mult", 2.0) if args is not None else 2.0
        )

        self._commit_spot_slack_mult = float(
            getattr(args, "commit_spot_slack_mult", 5.0) if args is not None else 5.0
        )

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _get_done_seconds(self) -> float:
        td = getattr(self, "task_done_time", None)
        if td is None:
            return 0.0
        if isinstance(td, (int, float)):
            return float(td)

        if isinstance(td, (list, tuple)):
            if not td:
                return 0.0

            try:
                s = 0.0
                for x in td:
                    if isinstance(x, (int, float)):
                        s += float(x)
                    elif isinstance(x, (list, tuple)) and len(x) >= 2:
                        try:
                            s += float(x[1]) - float(x[0])
                        except Exception:
                            pass
                if s > 0.0:
                    return s
            except Exception:
                pass

            last = td[-1]
            if isinstance(last, (int, float)):
                return float(last)
            if isinstance(last, (list, tuple)) and len(last) >= 2:
                try:
                    return float(last[1]) - float(last[0])
                except Exception:
                    return 0.0

        return 0.0

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not has_spot and self.env.cluster_type == ClusterType.SPOT:
            pass

        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        time_left = deadline - elapsed

        done = self._get_done_seconds()
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        done = max(0.0, min(task_duration, done))
        remaining = max(0.0, task_duration - done)

        if remaining <= 1e-9:
            return ClusterType.NONE

        restart_overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        deadline_guard = self._deadline_guard_steps * gap + self._deadline_guard_overhead_mult * restart_overhead
        if deadline_guard < 0.0:
            deadline_guard = 0.0

        if time_left <= 0.0:
            return ClusterType.ON_DEMAND

        od_overhead_if_start = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else restart_overhead

        must_use_on_demand_now = (od_overhead_if_start + remaining + deadline_guard) >= time_left
        if must_use_on_demand_now:
            self._committed_to_on_demand = True

        if self._committed_to_on_demand:
            slack = time_left - (remaining + od_overhead_if_start)
            commit_spot_slack = self._commit_spot_slack_mult * restart_overhead + 2.0 * gap

            if has_spot and slack > commit_spot_slack:
                if last_cluster_type == ClusterType.ON_DEMAND and slack <= (commit_spot_slack + restart_overhead):
                    return ClusterType.ON_DEMAND
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        parser.add_argument("--deadline_guard_steps", type=int, default=2)
        parser.add_argument("--deadline_guard_overhead_mult", type=float, default=2.0)
        parser.add_argument("--commit_spot_slack_mult", type=float, default=5.0)
        args, _ = parser.parse_known_args()
        return cls(args)