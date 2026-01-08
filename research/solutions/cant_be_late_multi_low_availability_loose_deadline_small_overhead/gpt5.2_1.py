import json
import math
from argparse import Namespace
from typing import Optional

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


_CT_SPOT = getattr(ClusterType, "SPOT")
_CT_OD = getattr(ClusterType, "ON_DEMAND")
_CT_NONE = getattr(ClusterType, "NONE", getattr(ClusterType, "None", None))
if _CT_NONE is None:
    _CT_NONE = ClusterType(0)  # Fallback; should never happen in the eval environment.


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_multi_region_v1"

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
        self._init_done = False
        return self

    def _lazy_init(self) -> None:
        if self._init_done:
            return
        num_regions = int(self.env.get_num_regions())
        self._num_regions = num_regions
        self._region_total = [0] * num_regions
        self._region_avail = [0] * num_regions
        self._t = 0

        self._committed_on_demand = False

        self._tdt_len = 0
        self._work_done = 0.0

        # Switching heuristics
        self._last_switch_t = -10**9
        self._switch_cooldown_steps = 2
        self._switch_margin = 0.02

        # Exploration strength for region selection when idling (spot unavailable).
        self._ucb_c = 0.8

        # Deadline safety buffer to avoid missing the deadline due to additional restarts.
        self._safety_buffer = max(1.0, 8.0 * float(self.restart_overhead))

        self._init_done = True

    def _update_work_done(self) -> None:
        tdt = self.task_done_time
        n = len(tdt)
        if n > self._tdt_len:
            self._work_done += sum(tdt[self._tdt_len : n])
            self._tdt_len = n

    def _select_best_region(self, current_region: int) -> Optional[int]:
        if self._num_regions <= 1:
            return None

        t = max(1, self._t)
        logt = math.log(t + 1.0)

        best_idx = current_region
        best_score = -1e30

        cur_tot = self._region_total[current_region]
        cur_av = self._region_avail[current_region]
        cur_mean = (cur_av + 1.0) / (cur_tot + 2.0)
        cur_ucb = cur_mean + self._ucb_c * math.sqrt(logt / (cur_tot + 1.0))

        for i in range(self._num_regions):
            tot = self._region_total[i]
            av = self._region_avail[i]
            mean = (av + 1.0) / (tot + 2.0)
            score = mean + self._ucb_c * math.sqrt(logt / (tot + 1.0))
            if score > best_score or (score == best_score and i < best_idx):
                best_score = score
                best_idx = i

        if best_idx == current_region:
            return None
        if best_score <= cur_ucb + self._switch_margin:
            return None
        if (self._t - self._last_switch_t) < self._switch_cooldown_steps:
            return None
        return best_idx

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._lazy_init()
        self._t += 1

        self._update_work_done()
        remaining_work = self.task_duration - self._work_done
        if remaining_work <= 0.0:
            return _CT_NONE

        time_left = self.deadline - float(self.env.elapsed_seconds)
        if time_left <= 0.0:
            return _CT_OD

        current_region = int(self.env.get_current_region())
        self._region_total[current_region] += 1
        if has_spot:
            self._region_avail[current_region] += 1

        if self._committed_on_demand:
            return _CT_OD

        if last_cluster_type == _CT_OD:
            od_initial_overhead = float(self.remaining_restart_overhead)
        else:
            od_initial_overhead = float(self.restart_overhead)

        required_time_od = remaining_work + od_initial_overhead
        if time_left <= required_time_od + self._safety_buffer:
            self._committed_on_demand = True
            return _CT_OD

        if float(self.remaining_restart_overhead) > 0.0:
            if last_cluster_type == _CT_SPOT and has_spot:
                return _CT_SPOT
            if last_cluster_type == _CT_OD:
                return _CT_OD
            return _CT_NONE

        if has_spot:
            return _CT_SPOT

        target = self._select_best_region(current_region)
        if target is not None:
            self.env.switch_region(int(target))
            self._last_switch_t = self._t

        return _CT_NONE