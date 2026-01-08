import math
from typing import Any, List

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_heuristic"

    def __init__(self, args):
        super().__init__(args)
        self.commit_to_on_demand = False
        self._initialized_policy = False
        self.time_safety_margin = 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _initialize_policy(self):
        if self._initialized_policy:
            return

        try:
            deadline = float(self.deadline)
        except Exception:
            deadline = 0.0
        try:
            task_duration = float(self.task_duration)
        except Exception:
            task_duration = 0.0
        try:
            gap = float(self.env.gap_seconds)
        except Exception:
            gap = 0.0
        try:
            overhead = float(self.restart_overhead)
        except Exception:
            overhead = 0.0

        slack = max(deadline - task_duration, 0.0)
        # Safety margin: between one gap and 20% of slack, bounded using overhead.
        margin_from_slack = 0.2 * slack if slack > 0.0 else gap + overhead
        margin_from_overheads = 2.0 * (gap + overhead)
        margin = max(gap, min(margin_from_slack, margin_from_overheads))
        self.time_safety_margin = margin
        self._initialized_policy = True

    def _compute_completed_work(self) -> float:
        segments = getattr(self, "task_done_time", None)
        if not segments:
            return 0.0
        total = 0.0
        for seg in segments:
            if isinstance(seg, (list, tuple)):
                if len(seg) >= 2:
                    try:
                        total += float(seg[1]) - float(seg[0])
                    except Exception:
                        continue
                elif len(seg) == 1:
                    try:
                        total += float(seg[0])
                    except Exception:
                        continue
            elif isinstance(seg, dict):
                if "duration" in seg:
                    try:
                        total += float(seg["duration"])
                    except Exception:
                        continue
                elif "end" in seg and "start" in seg:
                    try:
                        total += float(seg["end"]) - float(seg["start"])
                    except Exception:
                        continue
            else:
                try:
                    total += float(seg)
                except Exception:
                    continue
        if total < 0.0:
            total = 0.0
        return total

    def _remaining_work(self) -> float:
        try:
            done = self._compute_completed_work()
            remaining = float(self.task_duration) - done
        except Exception:
            remaining = float(getattr(self, "task_duration", 0.0))
        if remaining < 0.0:
            remaining = 0.0
        return remaining

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._initialize_policy()

        remaining_work = self._remaining_work()
        if remaining_work <= 0.0:
            return ClusterType.NONE

        try:
            t = float(self.env.elapsed_seconds)
        except Exception:
            t = 0.0
        try:
            dt = float(self.env.gap_seconds)
        except Exception:
            dt = 0.0
        try:
            deadline = float(self.deadline)
        except Exception:
            deadline = 0.0
        try:
            overhead = float(self.restart_overhead)
        except Exception:
            overhead = 0.0

        time_left = deadline - t
        if time_left <= 0.0:
            return ClusterType.NONE

        if not self.commit_to_on_demand:
            # If we spend this step without progress (NONE or preempted SPOT),
            # will we still be able to finish on on-demand with one restart?
            worst_finish_time_next_step = (
                t + dt + overhead + remaining_work + self.time_safety_margin
            )
            if worst_finish_time_next_step > deadline:
                self.commit_to_on_demand = True

        if self.commit_to_on_demand:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)