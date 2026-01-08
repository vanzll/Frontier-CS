import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy with deadline safety guarantees."""

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

        # Internal state initialization (deferred full init to first _step call).
        self._initialized = False
        self._lock_to_od = False
        self._cached_work_done = 0.0
        self._last_task_done_len = 0
        self._gap = None
        self._overhead = None
        self._margin = None
        return self

    def _initialize_runtime_state(self) -> None:
        """Initialize runtime-derived constants and caches."""
        self._gap = float(self.env.gap_seconds)
        self._overhead = float(self.restart_overhead)

        # Safety margin for switching permanently to on-demand.
        # Needs to be >= 2 * overhead + gap to guarantee deadline under arbitrary
        # preemption patterns (given on-demand is never preempted).
        raw_margin = 2.0 * self._overhead + 10.0 * self._gap

        # Cannot have a margin larger than initial slack.
        initial_slack = max(0.0, float(self.deadline) - float(self.task_duration))
        if initial_slack > 0.0:
            self._margin = min(raw_margin, initial_slack)
        else:
            self._margin = raw_margin

        # Initialize work-done cache from existing segments (if any).
        td = getattr(self, "task_done_time", None)
        if td is not None:
            self._cached_work_done = float(sum(td))
            self._last_task_done_len = len(td)
        else:
            self._cached_work_done = 0.0
            self._last_task_done_len = 0

        self._initialized = True

    def _update_work_done_cache(self) -> None:
        """Incrementally update cached total work done from task_done_time list."""
        td = self.task_done_time
        n = len(td)
        if n > self._last_task_done_len:
            added = 0.0
            for i in range(self._last_task_done_len, n):
                added += td[i]
            self._cached_work_done += added
            self._last_task_done_len = n

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.

        Returns: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        if not self._initialized:
            self._initialize_runtime_state()

        # Update cached completed work.
        self._update_work_done_cache()

        # If task is effectively complete, stop running to avoid extra cost.
        if self._cached_work_done >= float(self.task_duration) - 1e-6:
            return ClusterType.NONE

        # Once we've locked to on-demand, stay there to avoid further restarts.
        if self._lock_to_od:
            return ClusterType.ON_DEMAND

        elapsed = float(self.env.elapsed_seconds)
        time_left = float(self.deadline) - elapsed

        # Pathological case: no time left but still work to do -> choose on-demand.
        if time_left <= 0.0:
            self._lock_to_od = True
            return ClusterType.ON_DEMAND

        remaining_work = float(self.task_duration) - self._cached_work_done
        if remaining_work < 0.0:
            remaining_work = 0.0

        # If we're close enough to the deadline that we can't safely risk more
        # spot interruptions, permanently switch to on-demand.
        if time_left <= remaining_work + self._margin:
            self._lock_to_od = True
            return ClusterType.ON_DEMAND

        # Far from the deadline: prefer spot when available.
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable this step: fall back to on-demand to keep progress.
        return ClusterType.ON_DEMAND