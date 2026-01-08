import json
from argparse import Namespace
from typing import Optional, List

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


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

        self._inited = False
        self._work_done = 0.0
        self._task_done_len = 0

        self._spot_avail: List[int] = []
        self._spot_total: List[int] = []

        self._no_spot_streak = 0.0
        self._last_switch_elapsed = -1e30
        self._last_switched_from: Optional[int] = None

        self._forced_on_demand = False
        self._critical_margin = None
        self._switch_threshold = None
        self._switch_cooldown = None
        return self

    def _ensure_init(self) -> None:
        if self._inited:
            return
        n = int(self.env.get_num_regions())
        self._spot_avail = [0] * n
        self._spot_total = [0] * n
        self._task_done_len = 0
        self._work_done = 0.0
        self._no_spot_streak = 0.0
        self._last_switch_elapsed = -1e30
        self._last_switched_from = None
        self._forced_on_demand = False

        gap = float(self.env.gap_seconds)
        ro = float(self.restart_overhead)
        self._critical_margin = 3.0 * ro + 5.0 * gap
        self._switch_threshold = max(ro, 10.0 * gap)
        self._switch_cooldown = 2.0 * self._switch_threshold

        self._inited = True

    def _update_work_done(self) -> None:
        td = self.task_done_time
        new_len = len(td)
        old_len = self._task_done_len
        if new_len <= old_len:
            return
        s = 0.0
        for i in range(old_len, new_len):
            s += float(td[i])
        self._work_done += s
        self._task_done_len = new_len

    def _best_region_to_try(self, current: int) -> Optional[int]:
        n = len(self._spot_total)
        if n <= 1:
            return None

        best_idx = None
        best_score = -1.0
        for r in range(n):
            if r == current:
                continue
            if self._last_switched_from is not None and r == self._last_switched_from and n > 2:
                continue
            total = self._spot_total[r]
            avail = self._spot_avail[r]
            score = (avail + 1.0) / (total + 2.0)  # Laplace smoothing
            if score > best_score + 1e-15 or (best_idx is None or (abs(score - best_score) <= 1e-15 and r < best_idx)):
                best_score = score
                best_idx = r

        if best_idx is None:
            best_idx = (current + 1) % n
        return best_idx

    def _should_switch_region(self, last_cluster_type: ClusterType, slack: float, remaining_time: float) -> bool:
        n = len(self._spot_total)
        if n <= 1:
            return False
        if self._forced_on_demand:
            return False
        if slack <= 6.0 * float(self.restart_overhead):
            return False
        if remaining_time <= 0:
            return False

        elapsed = float(self.env.elapsed_seconds)
        if elapsed - self._last_switch_elapsed < float(self._switch_cooldown):
            return False
        if float(self._no_spot_streak) < float(self._switch_threshold):
            return False

        if last_cluster_type == ClusterType.NONE and float(self._no_spot_streak) < 2.0 * float(self._switch_threshold):
            return False

        return True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._ensure_init()
        self._update_work_done()

        elapsed = float(self.env.elapsed_seconds)
        gap = float(self.env.gap_seconds)
        ro = float(self.restart_overhead)

        current_region = int(self.env.get_current_region())
        self._spot_total[current_region] += 1
        if has_spot:
            self._spot_avail[current_region] += 1
            self._no_spot_streak = 0.0
        else:
            self._no_spot_streak += gap

        remaining_work = float(self.task_duration) - float(self._work_done)
        if remaining_work <= 0.0:
            return ClusterType.NONE

        remaining_time = float(self.deadline) - elapsed
        pending = float(self.remaining_restart_overhead)
        needed_time = remaining_work + pending
        slack = remaining_time - needed_time

        if self._forced_on_demand or remaining_time <= needed_time + float(self._critical_margin):
            self._forced_on_demand = True
            return ClusterType.ON_DEMAND

        if pending > 0.0:
            if has_spot:
                return ClusterType.SPOT
            return ClusterType.NONE

        if has_spot:
            return ClusterType.SPOT

        if self._should_switch_region(last_cluster_type, slack, remaining_time):
            target = self._best_region_to_try(current_region)
            if target is not None and target != current_region:
                self.env.switch_region(target)
                self._last_switch_elapsed = elapsed
                self._last_switched_from = current_region
                self._no_spot_streak = 0.0

        return ClusterType.NONE