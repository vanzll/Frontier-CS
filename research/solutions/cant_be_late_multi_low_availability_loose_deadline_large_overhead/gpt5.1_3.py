import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy aiming to minimize cost while meeting deadline."""
    NAME = "cant_be_late_v1"

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

        # Internal state (initialized lazily when env is available)
        self._internal_initialized = False
        self._commit_to_od = False
        self._completed_work = 0.0
        self._last_task_segments_len = 0

        return self

    def _initialize_internal(self) -> None:
        """Lazy init that depends on env."""
        # Safety buffer to account for step granularity and restart overhead behavior.
        # Ensures we commit to on-demand early enough even if a long step elapses.
        self._buffer_seconds = self.env.gap_seconds + 3.0 * self.restart_overhead

        # Initialize completed work from any existing segments (usually zero at start).
        segs = getattr(self, "task_done_time", None)
        if segs:
            self._completed_work = float(sum(segs))
            self._last_task_segments_len = len(segs)
        else:
            self._completed_work = 0.0
            self._last_task_segments_len = 0

        self._commit_to_od = False
        self._internal_initialized = True

    def _update_completed_work(self) -> None:
        """Incrementally track total completed work without re-summing each step."""
        segs = self.task_done_time
        n_prev = self._last_task_segments_len
        n_now = len(segs)
        if n_now > n_prev:
            # Sum only new segments appended since last step.
            self._completed_work += float(sum(segs[n_prev:n_now]))
            self._last_task_segments_len = n_now

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._internal_initialized:
            self._initialize_internal()

        # Update our running total of completed work.
        self._update_completed_work()

        remaining_work = self.task_duration - self._completed_work
        if remaining_work <= 0.0:
            # Task is done; no need to run more.
            return ClusterType.NONE

        now = self.env.elapsed_seconds

        # Decide when to irrevocably commit to on-demand.
        if not self._commit_to_od:
            # Worst-case time to finish if we switch to on-demand now.
            time_needed_on_demand = self.restart_overhead + remaining_work
            # Commit when there's just enough time left (plus safety buffer).
            if now + time_needed_on_demand + self._buffer_seconds >= self.deadline:
                self._commit_to_od = True

        if self._commit_to_od:
            # After committing, always run on on-demand until finished.
            return ClusterType.ON_DEMAND

        # Pre-commit phase: purely opportunistic Spot usage.
        if has_spot:
            return ClusterType.SPOT

        # No Spot available and we still have ample slack: wait (no cost).
        return ClusterType.NONE