import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "my_strategy"  # REQUIRED: unique identifier

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

        # Caches for efficient tracking of completed work.
        self._cached_done_time = 0.0
        self._last_done_len = 0

        # Runtime parameters depending on the environment.
        self._runtime_initialized = False
        self._gap = None
        self._margin = None

        return self

    def _initialize_runtime(self):
        """Lazy initialization of parameters that depend on the env."""
        if self._runtime_initialized:
            return
        self._gap = self.env.gap_seconds
        # Safety margin to guarantee deadline when switching to on-demand:
        # restart_overhead + up to two time steps worth of discretization.
        self._margin = self.restart_overhead + 2.0 * self._gap
        self._runtime_initialized = True

    def _update_done_cache(self):
        """Incrementally update cached total done work."""
        # After the first environment step, env.elapsed_seconds > 0.
        current_len = len(self.task_done_time)
        if current_len > self._last_done_len:
            # Sum only new segments since last call.
            total_new = 0.0
            for i in range(self._last_done_len, current_len):
                total_new += self.task_done_time[i]
            self._cached_done_time += total_new
            self._last_done_len = current_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.

        Returns: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        if not self._runtime_initialized:
            self._initialize_runtime()

        # Update cached completed work.
        if self.env.elapsed_seconds > 0:
            self._update_done_cache()

        done_work = self._cached_done_time
        remaining_work = self.task_duration - done_work

        # If task is complete or deadline reached/passed, do nothing.
        if remaining_work <= 0:
            return ClusterType.NONE

        time_remaining = self.deadline - self.env.elapsed_seconds
        if time_remaining <= 0:
            return ClusterType.NONE

        # Slack = time remaining - work remaining.
        slack = time_remaining - remaining_work
        margin = self._margin

        # If slack is small, commit to on-demand to guarantee completion.
        if slack <= margin:
            return ClusterType.ON_DEMAND

        # Opportunistic phase: use spot when available, otherwise wait.
        if has_spot:
            return ClusterType.SPOT

        # No spot and plenty of slack: wait (no cost).
        return ClusterType.NONE