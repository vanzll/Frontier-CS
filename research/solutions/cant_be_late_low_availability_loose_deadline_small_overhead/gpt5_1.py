import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_latest_start"

    def __init__(self, args=None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass
        self._committed_od = False
        self._extra_safety_seconds = getattr(args, "extra_safety_seconds", 0.0) if args else 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _need_on_demand_now(self, last_cluster_type: ClusterType, remain: float, time_left: float) -> bool:
        # Overhead if we were to switch to OD now
        overhead_if_switch = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else float(self.restart_overhead)
        gap = float(self.env.gap_seconds)
        safety_margin = max(gap, 0.0) + float(self._extra_safety_seconds)
        return time_left <= (remain + overhead_if_switch + safety_margin)

    def _remaining_work_seconds(self) -> float:
        done = sum(self.task_done_time) if self.task_done_time else 0.0
        remain = max(float(self.task_duration) - float(done), 0.0)
        return remain

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If already committed to on-demand, stay on it until completion.
        if self._committed_od:
            return ClusterType.ON_DEMAND

        remain = self._remaining_work_seconds()
        if remain <= 0.0:
            return ClusterType.NONE

        time_left = float(self.deadline) - float(self.env.elapsed_seconds)

        # If we must switch to OD to guarantee completion before the deadline.
        if self._need_on_demand_now(last_cluster_type, remain, time_left):
            self._committed_od = True
            return ClusterType.ON_DEMAND

        # Prefer spot when available to minimize cost.
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable and not yet forced to OD: pause (NONE) to wait.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument("--extra_safety_seconds", type=float, default=0.0)
        args, _ = parser.parse_known_args()
        return cls(args)