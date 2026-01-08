import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy with hard-deadline guarantee."""

    NAME = "cant_be_late_v1"

    def solve(self, spec_path: str) -> "Solution":
        """Initialize the solution from spec_path config."""
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        # Cache problem parameters in seconds using the spec directly.
        self._deadline_secs = float(config["deadline"]) * 3600.0
        self._task_duration_secs = float(config["duration"]) * 3600.0
        self._restart_overhead_secs = float(config["overhead"]) * 3600.0

        # Total slack = deadline - required work time.
        self._total_slack_secs = max(0.0, self._deadline_secs - self._task_duration_secs)

        # Safety overhead: account for both a possible pending overhead and a new one
        # when switching to on-demand. This is conservative but inexpensive.
        self._safety_overhead_secs = 2.0 * self._restart_overhead_secs

        # Maximum amount of "wasted" time we can afford before we must commit to
        # pure on-demand to still meet the deadline.
        self._postpone_threshold_secs = max(
            0.0, self._total_slack_secs - self._safety_overhead_secs
        )

        # Track cumulative useful work without re-summing the full list each step.
        self._cum_task_done = 0.0
        self._last_task_done_len = 0

        # Once set, we will always choose ON_DEMAND afterwards.
        self._force_on_demand = False

        return self

    def _update_cumulative_work_done(self) -> float:
        """Incrementally maintain sum(self.task_done_time) in O(1) amortized per step."""
        lst = getattr(self, "task_done_time", None)
        if not lst:
            return self._cum_task_done

        current_len = len(lst)
        if current_len > self._last_task_done_len:
            # Only add newly appended segments.
            for i in range(self._last_task_done_len, current_len):
                self._cum_task_done += lst[i]
            self._last_task_done_len = current_len

        return self._cum_task_done

    def _should_force_on_demand(self) -> None:
        """Decide whether we must switch to pure on-demand from now on."""
        if self._force_on_demand:
            return

        gap = self.env.gap_seconds
        work_done = self._update_cumulative_work_done()
        elapsed = self.env.elapsed_seconds

        # "Wasted" time so far: elapsed wall-clock minus useful work.
        wasted = elapsed - work_done

        # We may safely risk one more step that could yield zero progress
        # only if, after that step, a pure ON_DEMAND schedule could still
        # finish by the deadline. This gives:
        #   wasted_now + gap <= postpone_threshold  ==>  wasted_now <= postpone_threshold - gap
        if wasted > self._postpone_threshold_secs - gap:
            self._force_on_demand = True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.

        Returns: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        # Update internal state and possibly enter the "must use on-demand" phase.
        self._should_force_on_demand()

        if self._force_on_demand:
            # From this point on, always use on-demand to guarantee completion.
            return ClusterType.ON_DEMAND

        # Before the commit point: use Spot when available, otherwise wait.
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE