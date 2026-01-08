import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "cbm_slack_aware_v1"

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

        # Internal state for efficient tracking
        self._progress_done_seconds = 0.0
        self._segments_seen = 0
        self._commit_od = False
        return self

    def _update_progress(self):
        # Incrementally update progress to avoid O(n) summations every step
        n = len(self.task_done_time)
        if n > self._segments_seen:
            # Sum only new segments appended since last check
            new_work = 0.0
            for i in range(self._segments_seen, n):
                new_work += float(self.task_done_time[i])
            self._progress_done_seconds += new_work
            self._segments_seen = n

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_progress()

        # If task already done, no need to run further
        remaining = max(0.0, float(self.task_duration) - self._progress_done_seconds)
        if remaining <= 1e-9:
            return ClusterType.NONE

        now = float(self.env.elapsed_seconds)
        gap = float(self.env.gap_seconds)
        restart_overhead = float(self.restart_overhead)
        deadline = float(self.deadline)

        # Compute latest time we can start On-Demand with worst-case assumption:
        # - We might get zero useful work until switching to OD.
        # - Committing to OD incurs one restart overhead unless we're already on OD.
        # We use restart_overhead as the worst-case overhead at commit time.
        latest_commit_time = deadline - (remaining + restart_overhead)

        # If already committed to On-Demand, stay on it to avoid overheads and ensure completion.
        if self._commit_od or (now + gap) > latest_commit_time:
            self._commit_od = True
            return ClusterType.ON_DEMAND

        # We still have slack to try for cheap progress.
        # Prefer Spot if available; otherwise wait (NONE) to avoid cost.
        if has_spot:
            return ClusterType.SPOT

        return ClusterType.NONE