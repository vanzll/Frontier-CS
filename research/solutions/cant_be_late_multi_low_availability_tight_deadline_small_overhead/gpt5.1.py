import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "cbm_schedule_v1"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config.

        The spec file contains:
        - deadline: deadline in hours
        - duration: task duration in hours
        - overhead: restart overhead in hours
        - trace_files: list of trace file paths (one per region)
        """
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        # Internal state for tracking progress efficiently
        self._work_done = 0.0
        self._last_task_done_len = 0
        self._force_on_demand = False

        # Cache scalar versions of key parameters (in seconds)
        # MultiRegionStrategy is expected to expose these in seconds.
        self._task_duration = float(getattr(self, "task_duration"))
        self._deadline = float(getattr(self, "deadline"))
        self._restart_overhead = float(getattr(self, "restart_overhead"))

        # Compute a safety margin on top of the pure on-demand time needed.
        # This margin allows for one final switch overhead and some slack.
        slack = max(self._deadline - self._task_duration, 0.0)
        if slack <= 0.0:
            safety_margin = 2.0 * self._restart_overhead
        else:
            # At least several overheads worth of slack, and some fraction of total slack
            safety_margin = max(4.0 * self._restart_overhead, 0.15 * slack)
            # Do not use more than half of total slack as a margin
            max_margin = 0.5 * slack
            if safety_margin > max_margin:
                safety_margin = max_margin
        self._safety_margin = safety_margin

        return self

    def _update_work_done(self):
        """Incrementally update total work done from task_done_time list."""
        td_list = getattr(self, "task_done_time", None)
        if td_list is None:
            return
        n = len(td_list)
        if n > self._last_task_done_len:
            total_new = 0.0
            for i in range(self._last_task_done_len, n):
                total_new += td_list[i]
            self._work_done += total_new
            self._last_task_done_len = n

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.

        Returns: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        # Update internal accounting of work done
        self._update_work_done()

        remaining_work = self._task_duration - self._work_done
        if remaining_work <= 0.0:
            # Task is complete; don't run any more clusters.
            return ClusterType.NONE

        time_left = self._deadline - self.env.elapsed_seconds

        if time_left <= 0.0:
            # Already past deadline; penalty is unavoidable. Just use cheapest if possible.
            return ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND

        # Slack is how much extra wall-clock time we have beyond pure on-demand time.
        slack = time_left - remaining_work

        # Once slack is low, permanently switch to on-demand to guarantee completion.
        if (not self._force_on_demand) and slack <= self._safety_margin:
            self._force_on_demand = True

        if self._force_on_demand:
            return ClusterType.ON_DEMAND

        # Opportunistic phase: prefer Spot when available; otherwise, wait (NONE).
        if has_spot:
            return ClusterType.SPOT

        # No Spot available and we still have ample slack: wait to save on-demand cost.
        return ClusterType.NONE