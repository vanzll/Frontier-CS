import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


try:
    NONE_CLUSTER = ClusterType.NONE
except AttributeError:
    NONE_CLUSTER = ClusterType.None


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy with deadline guarantee."""

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
        return self

    def reset(self, *args, **kwargs):
        # Call parent reset if it exists
        parent_reset = getattr(super(), "reset", None)
        if callable(parent_reset):
            parent_reset(*args, **kwargs)

        # Initialize per-episode state
        self._work_done = 0.0
        self._last_task_done_len = 0
        self._committed_to_od = False

        # Cache scalar parameters (robust to list/tuple storage)
        td = getattr(self, "task_duration", 0.0)
        if isinstance(td, (list, tuple)):
            self._total_duration = float(td[0]) if td else 0.0
        else:
            self._total_duration = float(td)

        dl = getattr(self, "deadline", 0.0)
        if isinstance(dl, (list, tuple)):
            self._deadline = float(dl[0]) if dl else 0.0
        else:
            self._deadline = float(dl)

        ro = getattr(self, "restart_overhead", 0.0)
        if isinstance(ro, (list, tuple)):
            self._restart_overhead = float(ro[0]) if ro else 0.0
        else:
            self._restart_overhead = float(ro)

        return self

    def _ensure_init(self):
        """Lazy initialization in case reset() was not called."""
        if not hasattr(self, "_work_done"):
            self._work_done = 0.0
            self._last_task_done_len = 0
        if not hasattr(self, "_committed_to_od"):
            self._committed_to_od = False
        if not hasattr(self, "_total_duration"):
            td = getattr(self, "task_duration", 0.0)
            if isinstance(td, (list, tuple)):
                self._total_duration = float(td[0]) if td else 0.0
            else:
                self._total_duration = float(td)
        if not hasattr(self, "_deadline"):
            dl = getattr(self, "deadline", 0.0)
            if isinstance(dl, (list, tuple)):
                self._deadline = float(dl[0]) if dl else 0.0
            else:
                self._deadline = float(dl)
        if not hasattr(self, "_restart_overhead"):
            ro = getattr(self, "restart_overhead", 0.0)
            if isinstance(ro, (list, tuple)):
                self._restart_overhead = float(ro[0]) if ro else 0.0
            else:
                self._restart_overhead = float(ro)

    def _update_work_done(self):
        """Incrementally update cached total work done."""
        tasks = getattr(self, "task_done_time", None)
        if tasks is None:
            return
        length = len(tasks)
        last_len = self._last_task_done_len
        if length > last_len:
            total = self._work_done
            for i in range(last_len, length):
                total += tasks[i]
            self._work_done = total
            self._last_task_done_len = length

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Lazy-init state if needed
        self._ensure_init()

        # Once we ever run on-demand, stick to it for safety
        if last_cluster_type == ClusterType.ON_DEMAND:
            self._committed_to_od = True

        # Update accumulated work
        self._update_work_done()

        # If task is already completed or time exceeded, do nothing
        current_time = self.env.elapsed_seconds
        if self._work_done >= self._total_duration or current_time >= self._deadline:
            return NONE_CLUSTER

        # If we've already committed to on-demand, always use it
        if self._committed_to_od:
            return ClusterType.ON_DEMAND

        # Remaining work
        remaining_work = self._total_duration - self._work_done
        if remaining_work < 0.0:
            remaining_work = 0.0

        gap = self.env.gap_seconds
        overhead = self._restart_overhead
        deadline = self._deadline
        t = current_time

        # Deadline-safe commit rule:
        # If skipping this step could make it impossible to finish on time
        # using only on-demand from some future point, commit now.
        if t + overhead + remaining_work + 2.0 * gap > deadline:
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        # Otherwise, opportunistically use spot if available; else pause
        if has_spot:
            return ClusterType.SPOT
        return NONE_CLUSTER