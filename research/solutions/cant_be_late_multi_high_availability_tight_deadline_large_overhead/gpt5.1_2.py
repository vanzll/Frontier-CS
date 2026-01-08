import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy focused on cheap Spot usage with safe On-Demand fallback."""

    NAME = "cbm_multi_region_safewait_v1"

    def solve(self, spec_path: str) -> "Solution":
        """Initialize solution from spec_path config."""
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        # Cache total task duration and restart overhead (in seconds).
        td = getattr(self, "task_duration", 0.0)
        if isinstance(td, (list, tuple)):
            self._task_total_duration = float(sum(td))
        else:
            self._task_total_duration = float(td)

        ro = getattr(self, "restart_overhead", 0.0)
        if isinstance(ro, (list, tuple)):
            # There is only one task in this benchmark; be generic anyway.
            self._restart_overhead = float(ro[0] if ro else 0.0)
        else:
            self._restart_overhead = float(ro)

        # Incremental work tracking to avoid O(n^2) summations.
        self._work_done_cached = 0.0
        self._task_done_len = 0

        # Whether we've irrevocably switched to On-Demand.
        self.committed_to_on_demand = False

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """Decide next action based on current state."""
        # Ensure internal state exists (in case of alternative initialization paths).
        if not hasattr(self, "_work_done_cached"):
            self._work_done_cached = 0.0
            self._task_done_len = 0
        if not hasattr(self, "committed_to_on_demand"):
            self.committed_to_on_demand = False
        if not hasattr(self, "_task_total_duration"):
            td = getattr(self, "task_duration", 0.0)
            if isinstance(td, (list, tuple)):
                self._task_total_duration = float(sum(td))
            else:
                self._task_total_duration = float(td)
        if not hasattr(self, "_restart_overhead"):
            ro = getattr(self, "restart_overhead", 0.0)
            if isinstance(ro, (list, tuple)):
                self._restart_overhead = float(ro[0] if ro else 0.0)
            else:
                self._restart_overhead = float(ro)

        # Update cached work done using only new segments.
        tdt = self.task_done_time
        n = len(tdt)
        if n > self._task_done_len:
            self._work_done_cached += sum(tdt[self._task_done_len:n])
            self._task_done_len = n

        remaining_work = self._task_total_duration - self._work_done_cached
        if remaining_work <= 0.0:
            # Task is already complete.
            return ClusterType.NONE

        elapsed = self.env.elapsed_seconds
        time_left = self.deadline - elapsed

        # If we're already past the deadline, just run On-Demand (penalty already incurred).
        if time_left <= 0.0:
            self.committed_to_on_demand = True
            return ClusterType.ON_DEMAND

        # If we are already running On-Demand, stay on it to avoid extra restarts.
        if (
            self.committed_to_on_demand
            or last_cluster_type == ClusterType.ON_DEMAND
            or getattr(self.env, "cluster_type", None) == ClusterType.ON_DEMAND
        ):
            self.committed_to_on_demand = True
            return ClusterType.ON_DEMAND

        # Compute slack: time we can still "waste" via overhead or idling.
        slack = time_left - remaining_work

        # If we already have no slack (or negative), immediately fall back to On-Demand.
        if slack <= 0.0:
            self.committed_to_on_demand = True
            return ClusterType.ON_DEMAND

        gap = getattr(self.env, "gap_seconds", 1.0)
        restart_overhead = self._restart_overhead

        # Slack threshold: when slack drops below this, we irreversibly switch to On-Demand.
        # Designed to be small vs. total slack but large enough to cover restart overhead
        # and discretization effects.
        threshold_slack = max(3.0 * restart_overhead + 10.0 * gap, restart_overhead + 2.0 * gap)

        if slack <= threshold_slack:
            # Near the "last safe moment": commit to On-Demand to avoid deadline risk.
            self.committed_to_on_demand = True
            return ClusterType.ON_DEMAND

        # We have plenty of slack: aggressively favor Spot.
        if has_spot:
            return ClusterType.SPOT

        # Spot not available and we still have ample slack: wait (idle) for Spot to return.
        return ClusterType.NONE