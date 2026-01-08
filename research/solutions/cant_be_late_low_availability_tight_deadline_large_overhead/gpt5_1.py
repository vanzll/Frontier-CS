from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "jit_deadline_guard_v3"

    def __init__(self, args=None):
        try:
            super().__init__(args)
        except TypeError:
            super().__init__()
        self._committed_to_od = False
        self._task_done_len = 0
        self._task_done_sum = 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _update_work_done_sum(self):
        # Efficiently update the cumulative work done from task_done_time list
        td = self.task_done_time
        if td is None:
            return
        n = len(td)
        if n > self._task_done_len:
            # Sum only the new entries since last step
            incremental = 0.0
            for i in range(self._task_done_len, n):
                try:
                    incremental += float(td[i])
                except Exception:
                    continue
            self._task_done_sum += incremental
            self._task_done_len = n

    def _remaining_work_seconds(self) -> float:
        self._update_work_done_sum()
        rem = float(self.task_duration) - float(self._task_done_sum)
        return max(0.0, rem)

    def _time_left_seconds(self) -> float:
        return max(0.0, float(self.deadline) - float(self.env.elapsed_seconds))

    def _should_commit_to_od(self) -> bool:
        # Decide whether it's time to irrevocably switch to on-demand
        rem_work = self._remaining_work_seconds()
        time_left = self._time_left_seconds()
        gap = float(getattr(self.env, "gap_seconds", 60.0))
        # Safety margin to account for discretization and control delay
        margin = max(1.0, gap)

        # If already on OD, the overhead to continue is zero.
        # Otherwise, assume worst-case overhead to start OD exactly when committing.
        overhead = 0.0 if self.env.cluster_type == ClusterType.ON_DEMAND else float(self.restart_overhead)

        # Conservative condition: if time left is less than or equal to the remaining work
        # plus one restart overhead and a small margin, we must commit to OD now.
        return time_left <= (rem_work + float(self.restart_overhead) + margin)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If we've already committed to on-demand, always continue using it.
        if self._committed_to_od:
            return ClusterType.ON_DEMAND

        # If job already done (shouldn't be called in that case), do nothing
        if self._remaining_work_seconds() <= 0.0:
            return ClusterType.NONE

        # Evaluate whether it's time to commit to on-demand to ensure deadline safety
        if self._should_commit_to_od():
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        # Otherwise, use spot if available (cheap progress), else wait to save cost
        if has_spot:
            return ClusterType.SPOT
        else:
            # Wait (NONE) while we still have enough slack; commit will trigger when needed
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)