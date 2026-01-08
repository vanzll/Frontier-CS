import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy with deadline-aware spot usage."""

    NAME = "cant_be_late_heuristic_v1"

    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path) as f:
            config = json.load(f)

        deadline_hours = float(config["deadline"])
        duration_hours = float(config["duration"])
        overhead_hours = float(config["overhead"])

        args = Namespace(
            deadline_hours=deadline_hours,
            task_duration_hours=[duration_hours],
            restart_overhead_hours=[overhead_hours],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        # Internal parameters in seconds
        self._task_duration_s = duration_hours * 3600.0
        self._deadline_s = deadline_hours * 3600.0
        self._restart_overhead_s = overhead_hours * 3600.0

        # State for incremental work tracking
        self._work_done_cache = 0.0
        self._last_task_done_len = 0

        # Policy state
        self._policy_initialized = False
        self._force_on_demand = False

        return self

    def _initialize_policy(self) -> None:
        if self._policy_initialized:
            return
        self._policy_initialized = True

        # Time step size
        gap = getattr(self.env, "gap_seconds", 0.0)
        self._gap = float(gap)

        # Commit margin: ensure we switch to on-demand early enough that
        # a full step (including a possible restart overhead) still fits.
        dt_max = self._gap + self._restart_overhead_s
        if dt_max <= 0.0:
            if self._deadline_s > 0.0:
                dt_max = max(self._deadline_s * 0.01, 1.0)
            else:
                dt_max = 1.0
        # Add small safety buffer (60s) to handle rounding / modeling quirks.
        self._commit_margin_s = dt_max + 60.0

    def _update_work_done(self) -> None:
        td_list = getattr(self, "task_done_time", None)
        if td_list is None:
            return
        last_len = self._last_task_done_len
        current_len = len(td_list)
        if current_len > last_len:
            s = 0.0
            for i in range(last_len, current_len):
                s += float(td_list[i])
            self._work_done_cache += s
            self._last_task_done_len = current_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._policy_initialized:
            self._initialize_policy()

        # Update cached work done
        self._update_work_done()
        work_done = self._work_done_cache
        work_remaining = self._task_duration_s - work_done
        if work_remaining <= 0.0:
            # Task completed
            return ClusterType.NONE

        time_now = float(self.env.elapsed_seconds)
        time_left = self._deadline_s - time_now

        if time_left <= 0.0:
            # Already at or beyond deadline; best effort: run on-demand.
            self._force_on_demand = True
            return ClusterType.ON_DEMAND

        # Minimal time to finish if we switch to on-demand now and stay there.
        required_for_od = work_remaining + self._restart_overhead_s
        if required_for_od < 0.0:
            required_for_od = 0.0

        # Decide when to irrevocably switch to on-demand.
        if (not self._force_on_demand) and (time_left <= required_for_od + self._commit_margin_s):
            self._force_on_demand = True

        if self._force_on_demand:
            return ClusterType.ON_DEMAND

        # Pre-commit phase: use spot whenever available, otherwise wait.
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE