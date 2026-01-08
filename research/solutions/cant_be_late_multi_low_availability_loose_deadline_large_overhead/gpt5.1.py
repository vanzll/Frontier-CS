import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy focusing on deadline safety and low cost."""

    NAME = "cbmrs_v1"

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

        # Internal tracking of work done to avoid expensive sum() every step.
        self._total_work_done = 0.0
        self._prev_task_len = 0

        # Whether we've committed to always using on-demand from now on.
        self._force_on_demand = False

        # Commit slack: how much extra slack (in seconds) we require before
        # switching permanently to on-demand. Chosen to be a small multiple
        # of the gap and overhead for safety while keeping cost low.
        gap = getattr(self.env, "gap_seconds", 3600.0)
        restart = getattr(self, "restart_overhead", 0.0)
        # At least 2 steps of slack and a few overheads worth.
        self._commit_slack = max(2.0 * gap, 4.0 * restart)

        return self

    def _update_work_done(self) -> None:
        """Incrementally update cached total work done."""
        td = self.task_done_time
        n = len(td)
        if n > self._prev_task_len:
            # Only process new segments since last step.
            total_new = 0.0
            for i in range(self._prev_task_len, n):
                total_new += td[i]
            self._total_work_done += total_new
            self._prev_task_len = n

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update our cached work-done total.
        self._update_work_done()

        # Compute remaining work (seconds).
        remaining_work = self.task_duration - self._total_work_done
        if remaining_work <= 0:
            # Task finished: no need to run anything.
            return ClusterType.NONE

        # If we've already committed to on-demand, keep using it.
        if self._force_on_demand:
            return ClusterType.ON_DEMAND

        # Time left until the deadline.
        time_left = self.deadline - self.env.elapsed_seconds

        if time_left <= 0:
            # Past the deadline; nothing we can really do, but run on-demand
            # to minimize any additional penalty/cost-related issues.
            self._force_on_demand = True
            return ClusterType.ON_DEMAND

        # Conservative estimate of how much time we need if we switch to
        # on-demand now and stay there: remaining work + one restart overhead.
        safe_od_needed = remaining_work + self.restart_overhead

        # Extra slack beyond what on-demand needs to finish.
        extra_slack = time_left - safe_od_needed

        # If extra slack has shrunk below our threshold, permanently switch to
        # on-demand to guarantee completion before the deadline.
        if extra_slack <= self._commit_slack:
            self._force_on_demand = True
            return ClusterType.ON_DEMAND

        # Otherwise, we're far from the deadline in terms of on-demand safety.
        # Use spot when available, otherwise pause to avoid unnecessary cost.
        if has_spot:
            return ClusterType.SPOT

        return ClusterType.NONE