import math
from typing import Any, Tuple

try:
    from sky_spot.strategies.strategy import Strategy
    from sky_spot.utils import ClusterType
except ImportError:  # Fallback stubs for local testing; real evaluator provides these.
    from enum import Enum

    class ClusterType(Enum):
        SPOT = 1
        ON_DEMAND = 2
        NONE = 3

    class Strategy:
        def __init__(self, args=None):
            self.env = type("Env", (), {})()
            self.env.elapsed_seconds = 0.0
            self.env.gap_seconds = 60.0
            self.env.cluster_type = ClusterType.NONE
            self.task_duration = 0.0
            self.task_done_time = []
            self.deadline = 0.0
            self.restart_overhead = 0.0


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args: Any = None):
        super().__init__(args)
        self._committed_to_on_demand = False

    def solve(self, spec_path: str) -> "Solution":
        # Optional: load config from spec_path. Not used in this heuristic.
        self._committed_to_on_demand = False
        return self

    def _compute_work_done(self) -> float:
        """Compute total completed work duration from task_done_time segments."""
        segments = getattr(self, "task_done_time", [])
        total = 0.0
        for seg in segments:
            start = None
            end = None
            if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                start, end = seg[0], seg[1]
            elif isinstance(seg, dict):
                if "start" in seg and "end" in seg:
                    start, end = seg["start"], seg["end"]
            else:
                # Fallback: try attributes
                if hasattr(seg, "start") and hasattr(seg, "end"):
                    start, end = getattr(seg, "start"), getattr(seg, "end")
            if start is None or end is None:
                continue
            try:
                s = float(start)
                e = float(end)
            except (TypeError, ValueError):
                continue
            if e > s:
                total += e - s
        return total

    def _should_commit_to_on_demand(
        self,
        elapsed: float,
        gap: float,
        remaining_work: float,
        restart_overhead: float,
        deadline: float,
    ) -> bool:
        """Decide if we must irrevocably switch to on-demand to meet the deadline."""
        # Time to finish if we commit to on-demand now (worst-case: full restart overhead).
        time_if_commit_now = elapsed + restart_overhead + remaining_work

        # If even committing now cannot meet the deadline, we are already too late.
        # Still commit to minimize lateness.
        if time_if_commit_now > deadline:
            return True

        # If waiting one more step (and possibly making zero progress) would
        # make it impossible to finish on time even with full on-demand,
        # we must commit now.
        time_if_delay_one_step = elapsed + gap + restart_overhead + remaining_work
        if time_if_delay_one_step > deadline:
            return True

        return False

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Compute how much work is left.
        work_done = self._compute_work_done()
        remaining_work = max(0.0, float(self.task_duration) - work_done)

        # If the task is already complete, do nothing.
        if remaining_work <= 0.0:
            return ClusterType.NONE

        elapsed = float(self.env.elapsed_seconds)
        gap = float(self.env.gap_seconds)
        restart_overhead = float(self.restart_overhead)
        deadline = float(self.deadline)

        # Decide whether to commit to on-demand based on current state.
        if not self._committed_to_on_demand:
            if self._should_commit_to_on_demand(
                elapsed=elapsed,
                gap=gap,
                remaining_work=remaining_work,
                restart_overhead=restart_overhead,
                deadline=deadline,
            ):
                self._committed_to_on_demand = True

        # If committed, always use on-demand until the job is finished.
        if self._committed_to_on_demand:
            return ClusterType.ON_DEMAND

        # Pre-commit phase: be aggressive with spot, conservative with on-demand.
        if has_spot:
            return ClusterType.SPOT

        # No spot and not yet forced into on-demand: wait (use slack).
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)