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

        # Internal state for efficient progress tracking and policy control
        self._task_done_total = 0.0
        self._last_task_done_len = 0
        self._committed_to_on_demand = False

        # Margin of slack we require before we stop gambling on Spot.
        # Use one time-step as safety margin.
        self.commit_slack_margin = getattr(self.env, "gap_seconds", 1.0)

        return self

    def _update_task_done_total(self) -> None:
        """Incrementally maintain total completed work time."""
        lst = self.task_done_time
        n = len(lst)
        if n > self._last_task_done_len:
            # Only sum the newly appended segments
            new_segments = lst[self._last_task_done_len :]
            # Manual summation to avoid potential overhead of sum() on very small lists
            s = 0.0
            for v in new_segments:
                s += v
            self._task_done_total += s
            self._last_task_done_len = n

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
        # Efficiently track how much work has been done so far.
        self._update_task_done_total()

        elapsed = self.env.elapsed_seconds
        done = self._task_done_total
        remaining = self.task_duration - done
        if remaining <= 0.0:
            # Job already finished; no need to run further.
            return ClusterType.NONE

        time_left = self.deadline - elapsed
        if time_left <= 0.0:
            # Out of time; nothing sensible to do but stop.
            return ClusterType.NONE

        # If we've previously committed to on-demand, keep using it to avoid
        # further restarts and guarantee completion.
        if self._committed_to_on_demand:
            return ClusterType.ON_DEMAND

        gap = self.env.gap_seconds
        overhead = self.restart_overhead

        # Upper bound on time needed to finish if we switch to on-demand now
        # and then stay on it: one restart overhead plus remaining work.
        commit_needed_time = remaining + overhead

        # Slack: time we can still spend (even with no progress) and still
        # be able to finish purely with on-demand after that.
        slack = time_left - commit_needed_time

        # If slack is small (<= one time step), stop gambling on Spot and
        # switch permanently to on-demand to guarantee finishing before deadline.
        if slack <= self.commit_slack_margin:
            self._committed_to_on_demand = True
            return ClusterType.ON_DEMAND

        # Pre-commit phase: aggressively use Spot when available, otherwise pause.
        if has_spot:
            return ClusterType.SPOT

        # Spot not available and we still have comfortable slack: wait to save cost.
        return ClusterType.NONE