import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Deadline-safe, cost-aware multi-region scheduling strategy."""

    NAME = "my_strategy"

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

        # Internal state for efficient computation.
        self._params_cached = False
        self._lock_in_on_demand = False
        self._work_done_so_far = 0.0
        self._last_task_done_index = 0
        return self

    def _cache_params_if_needed(self) -> None:
        if self._params_cached:
            return

        # Task duration
        td = getattr(self, "task_duration", 0.0)
        if isinstance(td, (list, tuple)):
            td = td[0]
        self._task_duration = float(td)

        # Restart overhead
        ro = getattr(self, "restart_overhead", 0.0)
        if isinstance(ro, (list, tuple)):
            ro = ro[0]
        self._restart_overhead = float(ro)

        # Deadline
        dl = getattr(self, "deadline", 0.0)
        self._deadline = float(dl)

        # Time step size
        self._gap = float(getattr(self.env, "gap_seconds", 1.0))

        self._params_cached = True

    def _update_work_done_cache(self) -> None:
        """Incrementally track total work done to avoid O(n) sum each step."""
        segments = self.task_done_time
        if not segments:
            return
        n = len(segments)
        if n <= self._last_task_done_index:
            return
        # Sum only new segments since last step.
        total_new = 0.0
        for w in segments[self._last_task_done_index : n]:
            total_new += w
        self._work_done_so_far += total_new
        self._last_task_done_index = n

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._cache_params_if_needed()
        self._update_work_done_cache()

        remaining_work = self._task_duration - self._work_done_so_far

        # If task already complete, stop using resources.
        if remaining_work <= 0.0:
            self._lock_in_on_demand = True
            return ClusterType.NONE

        time_left = self._deadline - self.env.elapsed_seconds

        # If past deadline, just keep running on on-demand to minimize further loss.
        if time_left <= 0.0:
            self._lock_in_on_demand = True
            return ClusterType.ON_DEMAND

        # Minimum time needed to finish if we switch to on-demand now:
        # remaining_work + at most one restart_overhead.
        margin = time_left - (remaining_work + self._restart_overhead)

        # Once we commit to on-demand, we stay on it until completion.
        if self._lock_in_on_demand:
            return ClusterType.ON_DEMAND

        # If even immediate switch to on-demand barely suffices (or not at all),
        # switch and lock in now.
        if margin <= 0.0:
            self._lock_in_on_demand = True
            return ClusterType.ON_DEMAND

        # Safety rule:
        # Any action that may produce zero progress this step (SPOT or NONE)
        # is allowed only if margin >= gap. That guarantees we can still switch
        # to on-demand next step and meet the deadline even with no progress now.
        if has_spot:
            if margin >= self._gap:
                return ClusterType.SPOT
            else:
                self._lock_in_on_demand = True
                return ClusterType.ON_DEMAND
        else:
            if margin >= self._gap:
                # Wait for cheaper spot if we can afford to.
                return ClusterType.NONE
            else:
                self._lock_in_on_demand = True
                return ClusterType.ON_DEMAND