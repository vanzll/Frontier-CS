import json
import math
from argparse import Namespace
from typing import Optional, Sequence, Any

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


def _scalar_seconds(x: Any) -> float:
    if isinstance(x, (list, tuple)):
        if not x:
            return 0.0
        return float(x[0])
    return float(x)


def _ct_member(name: str) -> ClusterType:
    v = getattr(ClusterType, name, None)
    if v is not None:
        return v
    if name == "NONE":
        v = getattr(ClusterType, "None", None)
        if v is not None:
            return v
    raise AttributeError(f"ClusterType missing member {name}")


_CT_SPOT = _ct_member("SPOT")
_CT_ON_DEMAND = _ct_member("ON_DEMAND")
_CT_NONE = _ct_member("NONE")


class Solution(MultiRegionStrategy):
    NAME = "adaptive_slack_ucb_v1"

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

        self._done_sum = 0.0
        self._done_len = 0

        self._inited = False
        self._commit_on_demand = False

        self._reg_success = None
        self._reg_trials = None
        self._reg_total_obs = 0

        return self

    def _lazy_init(self) -> None:
        if self._inited:
            return
        n = int(self.env.get_num_regions())
        self._reg_success = [1.0] * n
        self._reg_trials = [2.0] * n
        self._reg_total_obs = 0
        self._inited = True

    def _update_done_sum(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            self._done_sum = 0.0
            self._done_len = 0
            return 0.0
        L = len(tdt)
        while self._done_len < L:
            self._done_sum += float(tdt[self._done_len])
            self._done_len += 1
        return self._done_sum

    def _choose_region_ucb(self, exclude_region: Optional[int]) -> int:
        n = int(self.env.get_num_regions())
        cur = int(self.env.get_current_region())
        total = float(self._reg_total_obs + 1)

        best = cur
        best_score = -1e30

        for i in range(n):
            if exclude_region is not None and i == exclude_region:
                continue
            trials = float(self._reg_trials[i])
            mean = float(self._reg_success[i]) / trials
            bonus = math.sqrt(2.0 * math.log(total + 1.0) / trials)
            score = mean + 0.35 * bonus
            if score > best_score:
                best_score = score
                best = i

        if exclude_region is not None and best == exclude_region:
            best = cur
        return best

    def _maybe_switch_region_for_probe(self) -> None:
        cur = int(self.env.get_current_region())
        target = self._choose_region_ucb(exclude_region=cur)
        if target != cur:
            self.env.switch_region(int(target))

    def _task_duration_seconds(self) -> float:
        return _scalar_seconds(getattr(self, "task_duration", 0.0))

    def _deadline_seconds(self) -> float:
        return _scalar_seconds(getattr(self, "deadline", 0.0))

    def _restart_overhead_seconds(self) -> float:
        return _scalar_seconds(getattr(self, "restart_overhead", 0.0))

    def _remaining_overhead_seconds(self) -> float:
        return _scalar_seconds(getattr(self, "remaining_restart_overhead", 0.0) or 0.0)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._lazy_init()

        cur_region = int(self.env.get_current_region())
        self._reg_total_obs += 1
        self._reg_trials[cur_region] += 1.0
        if has_spot:
            self._reg_success[cur_region] += 1.0

        done = self._update_done_sum()
        duration = self._task_duration_seconds()
        rem_work = duration - done
        if rem_work <= 1e-9:
            return _CT_NONE

        deadline = self._deadline_seconds()
        now = float(getattr(self.env, "elapsed_seconds", 0.0))
        time_rem = deadline - now
        if time_rem <= 0.0:
            return _CT_ON_DEMAND

        gap = float(getattr(self.env, "gap_seconds", 1.0))
        ro = self._restart_overhead_seconds()
        pending = self._remaining_overhead_seconds()

        buffer_commit = max(3.0 * gap, 10.0 * ro, gap + 4.0 * ro)
        if (not self._commit_on_demand) and (time_rem <= rem_work + pending + buffer_commit):
            self._commit_on_demand = True

        if self._commit_on_demand:
            return _CT_ON_DEMAND

        slack = time_rem - rem_work - pending

        if has_spot:
            if last_cluster_type == _CT_ON_DEMAND:
                if slack >= max(2.0 * gap, 6.0 * ro, gap + 2.0 * ro):
                    return _CT_SPOT
                return _CT_ON_DEMAND
            return _CT_SPOT

        wait_thresh = max(2.0 * gap + 3.0 * ro, 12.0 * ro + gap)
        if slack >= wait_thresh:
            self._maybe_switch_region_for_probe()
            return _CT_NONE

        if last_cluster_type != _CT_ON_DEMAND:
            self._maybe_switch_region_for_probe()
        return _CT_ON_DEMAND