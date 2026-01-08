import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy implementation."""

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

        # Defer full initialization until first _step call (env is ready).
        self._initialized = False
        return self

    def _initialize_internal_state(self):
        if self._initialized:
            return
        self._initialized = True

        # Cache scalar parameters (in seconds).
        td = getattr(self, "task_duration", None)
        if isinstance(td, (list, tuple)):
            self._task_total = float(td[0]) if td else 0.0
        else:
            self._task_total = float(td if td is not None else 0.0)

        dl = getattr(self, "deadline", 0.0)
        self._deadline = float(dl)

        ro = getattr(self, "restart_overhead", 0.0)
        self._restart_overhead = float(ro)

        env = getattr(self, "env", None)
        if env is not None:
            self._gap_seconds = float(getattr(env, "gap_seconds", 1.0))
        else:
            self._gap_seconds = 1.0

        # Progress tracking
        self._completed_work = 0.0
        self._last_segments_len = 0
        if hasattr(self, "task_done_time"):
            tdt = self.task_done_time
            if tdt:
                self._completed_work = float(sum(tdt))
                self._last_segments_len = len(tdt)

        # Control flags and parameters
        self._force_on_demand = False
        self._step_counter = 0

        # Heuristic thresholds for urgency (dimensionless ratios).
        self._critical_ratio = 0.92      # switch permanently to ON_DEMAND when above this
        self._non_spot_idle_ratio = 0.80  # when below this and no spot, we can idle

    def _update_completed_work(self):
        """Incrementally track total completed work to avoid O(n) summations each step."""
        tdt = self.task_done_time
        l = len(tdt)
        if l > self._last_segments_len:
            for i in range(self._last_segments_len, l):
                self._completed_work += tdt[i]
            self._last_segments_len = l

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Lazy initialization
        if not getattr(self, "_initialized", False):
            self._initialize_internal_state()

        self._step_counter += 1

        # Update progress
        self._update_completed_work()
        remaining = self._task_total - self._completed_work
        if remaining <= 0.0:
            # Task already complete; don't run more.
            return ClusterType.NONE

        elapsed = float(self.env.elapsed_seconds)
        T_left = self._deadline - elapsed
        if T_left <= 0.0:
            # Past deadline; still run on-demand to minimize overrun.
            self._force_on_demand = True
            return ClusterType.ON_DEMAND

        # Compute urgency ratio (required average speed as fraction of max).
        ratio = remaining / T_left

        # Decide if we must permanently commit to ON_DEMAND.
        if not self._force_on_demand:
            # Last-chance check: if even switching to ON_DEMAND now leaves no slack.
            overhead_if_switch = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else self._restart_overhead
            if remaining + overhead_if_switch >= T_left:
                self._force_on_demand = True
            elif ratio >= self._critical_ratio:
                self._force_on_demand = True

        if self._force_on_demand:
            return ClusterType.ON_DEMAND

        # Flexible phase: try to exploit Spot while we have slack.
        if has_spot:
            return ClusterType.SPOT

        # No Spot available in current region for this step.
        # Decide between idling (NONE) and using expensive ON_DEMAND.
        if ratio < self._non_spot_idle_ratio:
            # Still plenty of slack; it's safe to wait for cheaper Spot.
            return ClusterType.NONE

        # Getting tighter on time; use ON_DEMAND to maintain progress.
        return ClusterType.ON_DEMAND