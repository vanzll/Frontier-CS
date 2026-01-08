import json
import math
from argparse import Namespace
from typing import Optional

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_bandit_v1"

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

        self._initialized = False
        self._commit_ondemand = False

        self._done_sum = 0.0
        self._done_len = 0

        self._obs = None
        self._avail = None

        self._gap = None
        self._restart = self._as_scalar(getattr(self, "restart_overhead", 0.0))
        self._deadline = self._as_scalar(getattr(self, "deadline", 0.0))
        self._task_duration = self._as_scalar(getattr(self, "task_duration", 0.0))
        self._safety_margin = None
        return self

    @staticmethod
    def _as_scalar(x) -> float:
        if isinstance(x, (list, tuple)):
            if not x:
                return 0.0
            return float(x[0])
        return float(x)

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        n = int(self.env.get_num_regions())
        self._obs = [0] * n
        self._avail = [0] * n
        self._gap = float(self.env.gap_seconds)
        # Conservative but not too costly: provides buffer for step granularity and brief spot droughts.
        self._safety_margin = max(2.0 * self._gap, 4.0 * float(self._restart), 1800.0)
        self._initialized = True

    def _update_done_sum(self) -> None:
        tdt = self.task_done_time
        if tdt is None:
            return
        cur_len = len(tdt)
        if cur_len > self._done_len:
            self._done_sum += float(sum(tdt[self._done_len:cur_len]))
            self._done_len = cur_len

    def _select_next_region(self, current: int) -> Optional[int]:
        n = len(self._obs)
        if n <= 1:
            return None

        total_obs = sum(self._obs) + 1
        log_term = math.log(total_obs + 1.0)

        best_idx = None
        best_score = -1e100

        for i in range(n):
            obs = self._obs[i]
            avail = self._avail[i]
            p = (avail + 1.0) / (obs + 2.0)  # Laplace smoothing
            bonus = math.sqrt(2.0 * log_term / (obs + 1.0))
            score = p + bonus
            if i == current:
                score -= 1e-6  # tiny nudge to prefer switching when pausing
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None or best_idx == current:
            if n == 2:
                return 1 - current
            return (current + 1) % n
        return best_idx

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._lazy_init()
        self._update_done_sum()

        cur_region = int(self.env.get_current_region())
        if 0 <= cur_region < len(self._obs):
            self._obs[cur_region] += 1
            if has_spot:
                self._avail[cur_region] += 1

        remaining_work = self._task_duration - self._done_sum
        if remaining_work <= 1e-9:
            return ClusterType.NONE

        time_left = self._deadline - float(self.env.elapsed_seconds)
        if time_left <= 0.0:
            self._commit_ondemand = True
            return ClusterType.ON_DEMAND

        if self._commit_ondemand:
            return ClusterType.ON_DEMAND

        if last_cluster_type == ClusterType.ON_DEMAND:
            overhead_if_start_ondemand_now = float(getattr(self, "remaining_restart_overhead", 0.0))
        else:
            overhead_if_start_ondemand_now = float(self._restart)

        required_if_ondemand_now = remaining_work + overhead_if_start_ondemand_now
        if time_left <= required_if_ondemand_now + float(self._safety_margin):
            self._commit_ondemand = True
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        # Pause and search for a region likely to have spot next step.
        nxt = self._select_next_region(cur_region)
        if nxt is not None and nxt != cur_region:
            self.env.switch_region(int(nxt))
        return ClusterType.NONE