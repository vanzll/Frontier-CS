import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "cant_wait_fallback_od"

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
        self._commit_to_od = False
        self._work_done_total = 0.0
        self._cached_done_idx = 0
        return self

    def _update_work_done_cache(self):
        if self._cached_done_idx < len(self.task_done_time):
            # Accumulate only new entries to avoid O(n) each step
            for i in range(self._cached_done_idx, len(self.task_done_time)):
                self._work_done_total += self.task_done_time[i]
            self._cached_done_idx = len(self.task_done_time)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update cached progress
        self._update_work_done_cache()

        # Remaining work and time
        t_remain = max(0.0, self.task_duration - self._work_done_total)
        if t_remain <= 1e-9:
            return ClusterType.NONE

        time_left = self.deadline - self.env.elapsed_seconds
        if time_left <= 0.0:
            # Too late; still attempt to run OD to salvage (env will penalize anyway)
            self._commit_to_od = True
            return ClusterType.ON_DEMAND

        # If already committed to on-demand, stick with it till the end
        if self._commit_to_od:
            return ClusterType.ON_DEMAND

        # Time needed if we start/continue OD now (guaranteed non-preemptible)
        # Overhead to switch to OD only applies if we are not already on OD
        overhead_to_od = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else self.restart_overhead

        # Decision buffer to handle discretization; commit if we're at/inside this boundary
        D = time_left - (t_remain + overhead_to_od)

        # If we don't have enough time to rely on spot anymore, commit to OD
        # Also, if spot is currently unavailable and we're within ~0.8 step margin, commit now
        if D <= 0.0 or (not has_spot and D <= self.env.gap_seconds * 0.8):
            self._commit_to_od = True
            return ClusterType.ON_DEMAND

        # Otherwise, use spot when available; else wait (NONE) to save cost
        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE