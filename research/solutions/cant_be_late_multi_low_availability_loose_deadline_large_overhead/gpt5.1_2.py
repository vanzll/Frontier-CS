import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy."""

    NAME = "cant_be_late_static_commit_v1"

    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        # Internal state
        self._commit_time = None  # seconds; computed lazily when env is ready
        self._use_on_demand = False

        self._work_done = 0.0
        self._last_task_len = 0

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Lazily initialize commit time when env is available
        if self._commit_time is None:
            gap = float(getattr(self.env, "gap_seconds", 0.0))
            task_duration = float(self.task_duration)
            restart_overhead = float(self.restart_overhead)
            deadline = float(self.deadline)

            # Safe latest commit time: ensure that, even with worst-case remaining work
            # and one restart + discretization up to one additional step, an OD-only
            # schedule can finish before the deadline.
            buffer = 2.0 * gap
            latest_start = deadline - (task_duration + restart_overhead + buffer)
            if latest_start < 0.0:
                latest_start = 0.0
            elif latest_start > deadline:
                latest_start = deadline
            self._commit_time = latest_start

        # Update cached work done (incremental to avoid repeated full sums)
        td = self.task_done_time
        if td is not None:
            n = len(td)
            if n > self._last_task_len:
                new_work = 0.0
                for i in range(self._last_task_len, n):
                    new_work += td[i]
                self._work_done += new_work
                self._last_task_len = n

        # If task is already complete, no further computation is needed
        if self._work_done >= self.task_duration:
            return ClusterType.NONE

        # Decide if it's time to irrevocably switch to On-Demand
        if (not self._use_on_demand) and (self.env.elapsed_seconds >= self._commit_time):
            self._use_on_demand = True

        if self._use_on_demand:
            return ClusterType.ON_DEMAND

        # Before commit: opportunistically use Spot when available; otherwise, wait
        if has_spot:
            return ClusterType.SPOT

        return ClusterType.NONE