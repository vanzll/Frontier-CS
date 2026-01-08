import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy implementation."""

    NAME = "cant_be_late_multi_region_v1"

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

        # Normalize core parameters to scalars in seconds
        td = getattr(self, "task_duration", None)
        if isinstance(td, (list, tuple)):
            self._task_duration = float(td[0])
        else:
            self._task_duration = float(td)

        ro = getattr(self, "restart_overhead", None)
        if isinstance(ro, (list, tuple)):
            self._restart_overhead = float(ro[0])
        else:
            self._restart_overhead = float(ro)

        dl = getattr(self, "deadline", None)
        if isinstance(dl, (list, tuple)):
            self._deadline = float(dl[0])
        else:
            self._deadline = float(dl)

        # Environment time step
        self._gap = float(getattr(self.env, "gap_seconds", 1.0))

        # Internal tracking of completed work (seconds)
        self._work_done = 0.0
        self._last_len_task_done = 0

        # Control flags and thresholds (seconds)
        self._committed_to_on_demand = False

        # Commit to on-demand when slack drops below this.
        # Ensure it's comfortably larger than one step and several restart overheads.
        self._commit_slack = max(4.0 * self._restart_overhead, 3.0 * self._gap)

        # When spot is unavailable and slack is above this, we are willing to idle.
        # Below this but above commit_slack, we use on-demand during outages.
        self._idle_slack = self._commit_slack + 3.0 * self._gap

        return self

    def _update_work_done(self) -> None:
        """Incrementally accumulate completed work from task_done_time."""
        tlist = self.task_done_time
        last_len = self._last_len_task_done
        cur_len = len(tlist)
        if cur_len > last_len:
            acc = 0.0
            for i in range(last_len, cur_len):
                acc += tlist[i]
            self._work_done += acc
            self._last_len_task_done = cur_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update internal work counter
        self._update_work_done()

        # If job is already done, do not run anything further
        if self._work_done >= self._task_duration:
            return ClusterType.NONE

        # If we've already committed to on-demand, keep using it
        if self._committed_to_on_demand:
            return ClusterType.ON_DEMAND

        # Compute remaining work and slack
        remaining_work = self._task_duration - self._work_done
        time_elapsed = self.env.elapsed_seconds
        # Conservative slack: time left after paying one restart overhead and finishing remaining work
        slack_current = self._deadline - (time_elapsed + remaining_work + self._restart_overhead)

        # If slack is small, permanently commit to on-demand
        if slack_current <= self._commit_slack:
            self._committed_to_on_demand = True
            return ClusterType.ON_DEMAND

        # Still in spot-preferred phase
        if has_spot:
            # Prefer spot when available
            return ClusterType.SPOT

        # Spot not available; decide between idling and on-demand
        if slack_current >= self._idle_slack:
            # Plenty of slack: wait for cheap spot
            return ClusterType.NONE
        else:
            # Slack shrinking: use on-demand for progress
            return ClusterType.ON_DEMAND