import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
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

        # Internal lazy-init flag; real init happens on first _step call
        self._internal_inited = False
        return self

    def _initialize_internal_state(self) -> None:
        self._internal_inited = True

        # Normalize task_duration, restart_overhead, deadline into scalars (seconds)
        td = getattr(self, "task_duration", 0.0)
        if isinstance(td, (list, tuple)):
            self._task_duration = float(td[0])
        else:
            self._task_duration = float(td)

        ro = getattr(self, "restart_overhead", 0.0)
        if isinstance(ro, (list, tuple)):
            self._restart_overhead = float(ro[0])
        else:
            self._restart_overhead = float(ro)

        dl = getattr(self, "deadline", 0.0)
        if isinstance(dl, (list, tuple)):
            self._deadline = float(dl[0])
        else:
            self._deadline = float(dl)

        gap = float(getattr(self.env, "gap_seconds", 0.0))
        # Safety margin ensures we can always finish with On-Demand if we commit
        self._safety_margin = self._restart_overhead + gap

        # Track cumulative work done without O(N^2) summations
        td_list = getattr(self, "task_done_time", None)
        if td_list:
            self._total_done = float(sum(td_list))
            self._last_done_len = len(td_list)
        else:
            self._total_done = 0.0
            self._last_done_len = 0

        # Once committed, we use On-Demand only
        self._committed_to_on_demand = False

    def _update_progress_cache(self) -> None:
        td_list = self.task_done_time
        last_len = self._last_done_len
        cur_len = len(td_list)
        if cur_len > last_len:
            # Sum only newly added segments
            self._total_done += sum(td_list[last_len:cur_len])
            self._last_done_len = cur_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not getattr(self, "_internal_inited", False):
            self._initialize_internal_state()

        # Update cached work done
        self._update_progress_cache()

        remaining_work = self._task_duration - self._total_done
        if remaining_work <= 0.0:
            # Task is already complete; no need to run anything
            return ClusterType.NONE

        time_elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        time_left = self._deadline - time_elapsed

        if time_left <= 0.0:
            # Already at/after deadline; keep running On-Demand to minimize further risk
            self._committed_to_on_demand = True
            return ClusterType.ON_DEMAND

        # Decide whether we must commit to On-Demand to guarantee meeting the deadline
        if not self._committed_to_on_demand:
            # If the remaining time is not more than remaining work plus margin, commit
            if time_left <= remaining_work + self._safety_margin:
                self._committed_to_on_demand = True

        if self._committed_to_on_demand:
            return ClusterType.ON_DEMAND

        # Not yet committed: prefer Spot; if unavailable, wait (NONE) while slack is ample
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE