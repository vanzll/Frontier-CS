import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy."""

    NAME = "cant_be_late_mr_v1"

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
        self._total_done_time = 0.0
        self._last_task_done_len = 0
        # Commit margin: start on-demand somewhat before the strict latest time.
        self._commit_margin = 2.0 * self.restart_overhead
        self._must_run_on_demand = False

        return self

    def _update_progress_cache(self) -> None:
        """Efficiently maintain sum(task_done_time) without O(n) per step."""
        td = self.task_done_time
        last_len = self._last_task_done_len
        cur_len = len(td)
        if cur_len > last_len:
            total = self._total_done_time
            for i in range(last_len, cur_len):
                total += td[i]
            self._total_done_time = total
            self._last_task_done_len = cur_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update cached progress
        self._update_progress_cache()

        # Remaining work and time (seconds)
        remaining_work = self.task_duration - self._total_done_time
        if remaining_work <= 0.0:
            # Task already completed; no need to run more.
            return ClusterType.NONE

        remaining_time = self.deadline - self.env.elapsed_seconds
        if remaining_time <= 0.0:
            # Already at/after deadline; nothing smart to do.
            return ClusterType.ON_DEMAND

        # Decide whether to permanently switch to On-Demand.
        if not self._must_run_on_demand:
            # Delta measures slack if we were to switch to pure On-Demand now.
            # Required future time with On-Demand only: restart_overhead + remaining_work.
            delta = remaining_time - remaining_work - self.restart_overhead
            if delta <= self._commit_margin:
                self._must_run_on_demand = True

        if self._must_run_on_demand:
            return ClusterType.ON_DEMAND

        # Pre-commit phase: be aggressive on Spot, conservative on On-Demand.
        if has_spot:
            return ClusterType.SPOT

        # No Spot available and still in pre-commit phase: wait (no cost),
        # relying on future Spot or eventual switch to On-Demand near deadline.
        return ClusterType.NONE