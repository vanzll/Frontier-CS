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

        # Custom initialization for our strategy
        self._initialized = False
        self._committed_to_ondemand = False
        self._task_done_cache_len = 0
        self._task_done_accumulated = 0.0
        self._slack_threshold = None
        self._gap = None
        return self

    def _init_if_needed(self) -> None:
        """Lazily initialize parameters that depend on the environment."""
        if self._initialized:
            return

        gap = getattr(self.env, "gap_seconds", None)
        if gap is None:
            gap = 60.0  # sensible fallback
        self._gap = float(gap)

        # Normalize restart_overhead to a scalar in seconds
        ro = getattr(self, "restart_overhead", 0.0)
        if isinstance(ro, (list, tuple)):
            ro = ro[0] if ro else 0.0
        try:
            ro = float(ro)
        except Exception:
            ro = 0.0
        self.restart_overhead = ro

        # Slack threshold (seconds) at which we permanently switch to on-demand.
        # Ensure it's comfortably larger than one gap plus one restart overhead.
        self._slack_threshold = max(2.0 * self._gap + 4.0 * self.restart_overhead,
                                    10.0 * self.restart_overhead)

        self._initialized = True

    def _update_task_progress(self) -> float:
        """Efficiently maintain cumulative completed work time."""
        segments = self.task_done_time
        l = len(segments)
        if l > self._task_done_cache_len:
            acc = self._task_done_accumulated
            for i in range(self._task_done_cache_len, l):
                acc += segments[i]
            self._task_done_accumulated = acc
            self._task_done_cache_len = l
        return self._task_done_accumulated

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.

        Available attributes:
        - self.env.get_current_region(): Get current region index
        - self.env.get_num_regions(): Get total number of regions
        - self.env.switch_region(idx): Switch to region by index
        - self.env.elapsed_seconds: Current time elapsed
        - self.task_duration: Total task duration needed (seconds)
        - self.deadline: Deadline time (seconds)
        - self.restart_overhead: Restart overhead (seconds)
        - self.task_done_time: List of completed work segments
        - self.remaining_restart_overhead: Current pending overhead

        Returns: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        self._init_if_needed()

        done_work = self._update_task_progress()
        remaining_work = self.task_duration - done_work

        # If task already completed, do nothing.
        if remaining_work <= 0.0:
            return ClusterType.NONE

        elapsed = float(self.env.elapsed_seconds)
        time_left = self.deadline - elapsed

        # If we're somehow past the deadline with remaining work, use on-demand.
        if time_left <= 0.0:
            self._committed_to_ondemand = True
            return ClusterType.ON_DEMAND

        # Slack if we were to run purely on on-demand from now on.
        slack = time_left - remaining_work

        # If we've already committed to on-demand, keep using it.
        if self._committed_to_ondemand:
            return ClusterType.ON_DEMAND

        # Not enough slack left: permanently switch to on-demand.
        if slack <= self._slack_threshold:
            self._committed_to_ondemand = True
            return ClusterType.ON_DEMAND

        # Plenty of slack: prefer Spot when available.
        if has_spot:
            return ClusterType.SPOT

        # No Spot available. Decide between waiting (NONE) and switching to on-demand.
        gap = self._gap
        projected_time_left = time_left - gap
        if projected_time_left <= 0.0:
            # Waiting would push us past the deadline.
            self._committed_to_ondemand = True
            return ClusterType.ON_DEMAND

        projected_slack = projected_time_left - remaining_work
        if projected_slack <= self._slack_threshold:
            # Can't safely afford to keep waiting; switch to on-demand now.
            self._committed_to_ondemand = True
            return ClusterType.ON_DEMAND

        # We have sufficient slack to wait for Spot to return.
        return ClusterType.NONE