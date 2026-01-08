import json
import math
from argparse import Namespace
from typing import Optional

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_multiregion_v1"

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
        self._reset_internal()
        return self

    def _reset_internal(self) -> None:
        self._initialized = False
        self._t = 0
        self._last_elapsed = -1.0
        self._commit_on_demand = False

        self._work_done = 0.0
        self._tdt_len = 0

        self._search_streak = 0
        self._search_dwell_steps = 1

        self._num_regions = 0
        self._total_obs = []
        self._spot_obs = []
        self._last_spot_step = []
        self._last_visit_step = []

    def _ensure_initialized(self) -> None:
        if getattr(self, "_initialized", False):
            return
        try:
            n = int(self.env.get_num_regions())
        except Exception:
            n = 1
        self._num_regions = max(1, n)
        self._total_obs = [0] * self._num_regions
        self._spot_obs = [0] * self._num_regions
        self._last_spot_step = [-10**18] * self._num_regions
        self._last_visit_step = [-10**18] * self._num_regions
        gap = float(getattr(self.env, "gap_seconds", 3600.0))
        self._search_dwell_steps = max(1, int(1800.0 / max(1.0, gap)))
        self._initialized = True

    @staticmethod
    def _as_scalar(x) -> float:
        if isinstance(x, (list, tuple)):
            return float(x[0]) if x else 0.0
        return float(x)

    def _get_task_done_list(self):
        tdt = self.task_done_time
        if isinstance(tdt, list) and tdt and isinstance(tdt[0], list):
            return tdt[0]
        return tdt

    def _update_work_done(self) -> None:
        tdt = self._get_task_done_list()
        if not isinstance(tdt, list):
            self._work_done = 0.0
            self._tdt_len = 0
            return
        L = len(tdt)
        if L <= self._tdt_len:
            return
        if L == self._tdt_len + 1:
            self._work_done += float(tdt[-1])
        else:
            self._work_done += float(sum(tdt[self._tdt_len : L]))
        self._tdt_len = L

    def _best_region_to_probe(self, current_region: int) -> Optional[int]:
        n = self._num_regions
        if n <= 1:
            return None
        t = self._t
        best_idx = None
        best_score = -1e30
        logt = math.log(t + 2.0)
        for i in range(n):
            if i == current_region:
                continue
            total = self._total_obs[i]
            spot = self._spot_obs[i]
            p = (spot + 1.0) / (total + 2.0)
            bonus = math.sqrt(2.0 * logt / (total + 1.0))
            rec = 1.0 / (1.0 + (t - self._last_spot_step[i])) if self._last_spot_step[i] > -10**17 else 0.0
            visit_rec = 1.0 / (1.0 + (t - self._last_visit_step[i])) if self._last_visit_step[i] > -10**17 else 0.0
            score = p + 0.35 * bonus + 0.20 * rec - 0.05 * visit_rec
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._ensure_initialized()

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        if self._last_elapsed >= 0.0 and elapsed < self._last_elapsed:
            self._reset_internal()
            self._ensure_initialized()
        self._last_elapsed = elapsed

        self._t += 1
        self._update_work_done()

        gap = float(getattr(self.env, "gap_seconds", 3600.0))
        task_duration = self._as_scalar(self.task_duration)
        deadline = self._as_scalar(self.deadline)
        restart_overhead = self._as_scalar(self.restart_overhead)
        remaining_restart_overhead = self._as_scalar(getattr(self, "remaining_restart_overhead", 0.0))

        try:
            current_region = int(self.env.get_current_region())
        except Exception:
            current_region = 0
        if current_region < 0 or current_region >= self._num_regions:
            current_region = 0

        self._total_obs[current_region] += 1
        self._last_visit_step[current_region] = self._t
        if has_spot:
            self._spot_obs[current_region] += 1
            self._last_spot_step[current_region] = self._t

        remaining_work = task_duration - self._work_done
        if remaining_work <= 1e-9:
            return ClusterType.NONE

        remaining_time = deadline - elapsed
        if remaining_time <= 0.0:
            return ClusterType.ON_DEMAND

        pending_overhead_od = remaining_restart_overhead if last_cluster_type == ClusterType.ON_DEMAND else restart_overhead
        t_finish_od = remaining_work + pending_overhead_od

        commit_margin = max(3.0 * gap, 6.0 * restart_overhead)
        if not self._commit_on_demand and remaining_time <= t_finish_od + commit_margin:
            self._commit_on_demand = True

        if self._commit_on_demand:
            self._search_streak = 0
            return ClusterType.ON_DEMAND

        if has_spot:
            self._search_streak = 0
            if last_cluster_type == ClusterType.ON_DEMAND and remaining_work <= 3.0 * gap:
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        t_finish_if_wait_then_od = gap + remaining_work + restart_overhead
        if remaining_time <= t_finish_if_wait_then_od + commit_margin:
            self._search_streak = 0
            return ClusterType.ON_DEMAND

        if last_cluster_type != ClusterType.NONE:
            self._search_streak = 0
        self._search_streak += 1

        if self._search_streak >= self._search_dwell_steps:
            nxt = self._best_region_to_probe(current_region)
            if nxt is not None and nxt != current_region:
                try:
                    self.env.switch_region(nxt)
                    current_region = nxt
                except Exception:
                    pass
            self._search_streak = 0

        return ClusterType.NONE