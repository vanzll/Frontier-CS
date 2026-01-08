import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "cb_late_multi_region_v1"

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
        self._init_internal_state()
        return self

    def _init_internal_state(self):
        gap = getattr(self.env, "gap_seconds", 1.0)
        overhead = getattr(self, "restart_overhead", 0.0)
        # Commit when slack becomes smaller than one step plus overhead.
        self._commit_slack_threshold = gap + overhead
        self._force_on_demand = False
        self._work_done = 0.0
        self._last_task_done_len = 0

    def _update_work_done(self):
        if not hasattr(self, "_commit_slack_threshold"):
            self._init_internal_state()

        td_list = getattr(self, "task_done_time", None)
        if td_list is None:
            return

        current_len = len(td_list)

        # Detect environment resets (new episode)
        if current_len < getattr(self, "_last_task_done_len", 0):
            self._init_internal_state()
            current_len = len(td_list)

        if not hasattr(self, "_last_task_done_len"):
            self._last_task_done_len = 0
            self._work_done = 0.0

        if current_len > self._last_task_done_len:
            added = 0.0
            for i in range(self._last_task_done_len, current_len):
                added += td_list[i]
            self._work_done += added
            self._last_task_done_len = current_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_work_done()

        task_duration = getattr(self, "task_duration", None)
        if task_duration is None:
            # Fallback: prefer cheaper but reliable if no config
            return ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND

        remaining_work = task_duration - getattr(self, "_work_done", 0.0)
        if remaining_work <= 0:
            # Task finished; avoid additional cost.
            return ClusterType.NONE

        deadline = getattr(self, "deadline", None)
        elapsed = getattr(self.env, "elapsed_seconds", 0.0)
        if deadline is None:
            # No deadline info; basic cost-aware behavior.
            return ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND

        remaining_time = deadline - elapsed
        if remaining_time <= 0:
            # Past deadline; just run on-demand to minimize extra delay.
            return ClusterType.ON_DEMAND

        restart_overhead = getattr(self, "restart_overhead", 0.0)

        # Overhead if we switch to on-demand now.
        overhead_if_switch = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else restart_overhead

        slack = remaining_time - (remaining_work + overhead_if_switch)

        if not hasattr(self, "_commit_slack_threshold"):
            self._init_internal_state()
        commit_slack_threshold = self._commit_slack_threshold

        if not getattr(self, "_force_on_demand", False):
            if slack < 0.0:
                # Behind schedule; commit to on-demand.
                self._force_on_demand = True
            elif slack <= commit_slack_threshold:
                # Close to boundary where only on-demand can guarantee completion.
                self._force_on_demand = True

        if getattr(self, "_force_on_demand", False):
            return ClusterType.ON_DEMAND

        # Before committing: use spot whenever available; otherwise pause.
        if has_spot:
            return ClusterType.SPOT

        return ClusterType.NONE