import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Cant-Be-Late Multi-Region Scheduling Strategy."""

    NAME = "cbmrs_heuristic_v1"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config.
        """
        with open(spec_path) as f:
            config = json.load(f)

        # Store config values (hours) for fallback use
        self._duration_hours = float(config["duration"])
        self._deadline_hours = float(config["deadline"])
        self._overhead_hours = float(config["overhead"])

        args = Namespace(
            deadline_hours=self._deadline_hours,
            task_duration_hours=[self._duration_hours],
            restart_overhead_hours=[self._overhead_hours],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        # Runtime state
        self._initialized_runtime_state = False
        self._committed_to_on_demand = False

        # Efficient progress tracking
        self._progress_cache = 0.0
        self._task_done_len = 0

        return self

    def _initialize_runtime_state(self):
        """Lazily initialize derived runtime parameters."""
        if self._initialized_runtime_state:
            return

        # Task duration in seconds
        if hasattr(self, "task_duration"):
            td = getattr(self, "task_duration")
            if isinstance(td, (list, tuple)):
                self._task_duration_val = float(td[0])
            else:
                self._task_duration_val = float(td)
        else:
            self._task_duration_val = self._duration_hours * 3600.0

        # Restart overhead in seconds
        if hasattr(self, "restart_overhead"):
            ro = getattr(self, "restart_overhead")
            if isinstance(ro, (list, tuple)):
                self._restart_overhead_val = float(ro[0])
            else:
                self._restart_overhead_val = float(ro)
        else:
            self._restart_overhead_val = self._overhead_hours * 3600.0

        # Deadline in seconds
        if hasattr(self, "deadline"):
            dl = getattr(self, "deadline")
            if isinstance(dl, (list, tuple)):
                self._deadline_val = float(dl[0])
            else:
                self._deadline_val = float(dl)
        else:
            self._deadline_val = self._deadline_hours * 3600.0

        self._initialized_runtime_state = True

    def _get_progress(self) -> float:
        """Return cached total progress (seconds), updating incrementally."""
        segments = getattr(self, "task_done_time", []) or []

        # Ensure cache initialized
        if not hasattr(self, "_progress_cache"):
            self._progress_cache = 0.0
            self._task_done_len = 0

        n = len(segments)
        if n > self._task_done_len:
            new_sum = 0.0
            for v in segments[self._task_done_len : n]:
                new_sum += float(v)
            self._progress_cache += new_sum
            self._task_done_len = n

        return self._progress_cache

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        self._initialize_runtime_state()

        # If we've committed to on-demand, stay on on-demand forever.
        if self._committed_to_on_demand:
            return ClusterType.ON_DEMAND

        elapsed = float(self.env.elapsed_seconds)
        gap = float(self.env.gap_seconds)

        # Current net progress and remaining work (seconds)
        progress = self._get_progress()
        remaining_work = max(0.0, self._task_duration_val - progress)

        # Conservative estimate of time needed if we switch to on-demand now:
        # remaining work plus a full restart overhead.
        on_demand_time_needed = remaining_work + self._restart_overhead_val

        # Slack time if we switch to on-demand immediately.
        slack = self._deadline_val - (elapsed + on_demand_time_needed)

        # Safety margin: at most one timestep of slack loss between decisions.
        safety_margin = gap

        # If slack is small enough that waiting one more step could risk
        # missing the deadline, commit to on-demand now.
        if slack <= safety_margin:
            self._committed_to_on_demand = True
            return ClusterType.ON_DEMAND

        # Before commitment: use Spot whenever available, otherwise wait (NONE)
        # to avoid unnecessary on-demand cost.
        if has_spot:
            return ClusterType.SPOT

        # Spot not available and we still have ample slack: pause to save cost.
        return ClusterType.NONE