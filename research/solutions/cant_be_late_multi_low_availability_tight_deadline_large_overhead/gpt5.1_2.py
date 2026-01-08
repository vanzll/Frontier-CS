import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy with conservative deadline guarantees."""

    NAME = "cant_be_late_threshold"

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

        # Internal strategy state
        self._strategy_initialized = False
        self._gap_seconds = None

        self._work_done_cache = 0.0
        self._last_td_len = 0

        self._committed_to_on_demand = False

        return self

    # --- Internal helpers -------------------------------------------------

    def _initialize_strategy_state(self):
        """Lazy initialization of derived parameters from the environment."""
        if self._strategy_initialized:
            return

        # Time step size
        self._gap_seconds = getattr(self.env, "gap_seconds", 3600.0)

        # Ensure second-based fields exist
        if not hasattr(self, "task_duration"):
            td_hours = getattr(self, "task_duration_hours", [0.0])[0]
            self.task_duration = float(td_hours) * 3600.0

        if not hasattr(self, "restart_overhead"):
            oh_hours = getattr(self, "restart_overhead_hours", [0.0])[0]
            self.restart_overhead = float(oh_hours) * 3600.0

        if not hasattr(self, "deadline"):
            dl_hours = getattr(self, "deadline_hours", 0.0)
            self.deadline = float(dl_hours) * 3600.0

        self._strategy_initialized = True

    def _update_work_done_cache(self):
        """Incrementally maintain total completed work to avoid O(n) summations."""
        td = getattr(self, "task_done_time", None)
        if td is None:
            return
        n = len(td)
        if n > self._last_td_len:
            # Sum only new segments since last call.
            self._work_done_cache += sum(td[self._last_td_len:n])
            self._last_td_len = n

    # --- Core decision logic ----------------------------------------------

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.

        Returns: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        # Lazy init parameters derived from env
        self._initialize_strategy_state()

        # Update cached progress
        self._update_work_done_cache()

        # Remaining work (seconds)
        work_left = max(self.task_duration - self._work_done_cache, 0.0)

        # If task already completed, do nothing
        if work_left <= 0.0:
            self._committed_to_on_demand = True
            return ClusterType.NONE

        # Remaining time to deadline (seconds)
        time_left = self.deadline - self.env.elapsed_seconds

        # If we're already past the deadline, just use on-demand as best effort
        if time_left <= 0.0:
            self._committed_to_on_demand = True
            return ClusterType.ON_DEMAND

        # Once we commit to on-demand, we never switch back to avoid extra overhead
        if self._committed_to_on_demand:
            return ClusterType.ON_DEMAND

        overhead = self.restart_overhead
        gap = self._gap_seconds

        # Conservative threshold for continuing to gamble on spot or waiting.
        # Derivation (worst-case):
        # - This step on spot could yield zero progress and incur up to `gap + overhead`
        #   additional time before we can react.
        # - We still need one more `overhead` to start a final on-demand run.
        # To ensure that, even in this worst-case, we can switch to on-demand and
        # finish before the deadline, require:
        #   time_left >= work_left + 2 * overhead + gap
        safe_spot_threshold = work_left + 2.0 * overhead + gap

        # If we don't have enough slack to safely use spot/wait, commit to on-demand.
        if time_left <= safe_spot_threshold:
            self._committed_to_on_demand = True
            return ClusterType.ON_DEMAND

        # Plenty of slack: prefer cheap spot when available, else wait (NONE) to save cost.
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable and still lots of slack: pause instead of paying for on-demand.
        return ClusterType.NONE