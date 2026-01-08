import json
import math
from argparse import Namespace
from typing import Optional

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_ucb_v1"

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
        self._committed = False

        self._task_duration_seconds = None
        self._deadline_seconds = None
        self._restart_overhead_seconds = None

        self._task_done_len = 0
        self._work_done_total = 0.0

        self._t = 0

        self._region_total = None
        self._region_spot = None
        self._ucb_c = 1.1

        self._commit_buffer = None

        self._no_spot_streak = 0
        self._rr_base_region = None

        return self

    def _lazy_init(self) -> None:
        if self._initialized:
            return

        td = getattr(self, "task_duration", 0.0)
        if isinstance(td, (list, tuple)):
            td = td[0] if td else 0.0
        self._task_duration_seconds = float(td)

        dl = getattr(self, "deadline", 0.0)
        if isinstance(dl, (list, tuple)):
            dl = dl[0] if dl else 0.0
        self._deadline_seconds = float(dl)

        ro = getattr(self, "restart_overhead", 0.0)
        if isinstance(ro, (list, tuple)):
            ro = ro[0] if ro else 0.0
        self._restart_overhead_seconds = float(ro)

        n = 1
        try:
            n = int(self.env.get_num_regions())
        except Exception:
            n = 1

        self._region_total = [0] * n
        self._region_spot = [0] * n

        self._initialized = True

    def _update_work_done(self) -> None:
        td = self.task_done_time
        if td is None:
            return
        ln = len(td)
        if ln < self._task_done_len:
            self._work_done_total = float(sum(td))
            self._task_done_len = ln
            return
        if ln == self._task_done_len:
            return
        self._work_done_total += float(sum(td[self._task_done_len:ln]))
        self._task_done_len = ln

    def _pick_region_ucb_excluding_current(self, current: int) -> Optional[int]:
        n = len(self._region_total)
        if n <= 1:
            return None

        t = self._t + 1
        best_i = None
        best_score = -1e30
        logt = math.log(t + 1.0)

        for i in range(n):
            if i == current:
                continue
            tot = self._region_total[i]
            spot = self._region_spot[i]
            mean = (spot + 1.0) / (tot + 2.0)
            bonus = self._ucb_c * math.sqrt(logt / (tot + 1.0))
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_i = i

        return best_i

    def _remaining_work(self) -> float:
        rem = self._task_duration_seconds - self._work_done_total
        return rem if rem > 0.0 else 0.0

    def _ensure_commit_buffer(self) -> None:
        if self._commit_buffer is not None:
            return
        gap = getattr(self.env, "gap_seconds", 1.0)
        try:
            gap = float(gap)
        except Exception:
            gap = 1.0
        self._commit_buffer = max(5.0, 0.02 * gap)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._lazy_init()
        self._ensure_commit_buffer()

        self._t += 1
        env = self.env

        try:
            cur_region = int(env.get_current_region())
        except Exception:
            cur_region = 0

        if 0 <= cur_region < len(self._region_total):
            self._region_total[cur_region] += 1
            if has_spot:
                self._region_spot[cur_region] += 1

        self._update_work_done()
        remaining = self._remaining_work()
        if remaining <= 0.0:
            return ClusterType.NONE

        elapsed = float(getattr(env, "elapsed_seconds", 0.0))
        time_left = self._deadline_seconds - elapsed
        if time_left <= 0.0:
            return ClusterType.NONE

        if not self._committed:
            if last_cluster_type == ClusterType.ON_DEMAND:
                overhead = float(getattr(self, "remaining_restart_overhead", 0.0) or 0.0)
            else:
                overhead = self._restart_overhead_seconds
            if time_left <= remaining + overhead + self._commit_buffer:
                self._committed = True

        if self._committed:
            self._no_spot_streak = 0
            self._rr_base_region = None
            return ClusterType.ON_DEMAND

        if has_spot:
            self._no_spot_streak = 0
            self._rr_base_region = None
            return ClusterType.SPOT

        n = len(self._region_total)
        if n > 1:
            if self._no_spot_streak == 0:
                self._rr_base_region = cur_region
            self._no_spot_streak += 1

            next_region = None
            if self._rr_base_region is not None and self._no_spot_streak <= (n - 1):
                candidate = (self._rr_base_region + self._no_spot_streak) % n
                if candidate != cur_region:
                    next_region = candidate
            if next_region is None:
                next_region = self._pick_region_ucb_excluding_current(cur_region)

            if next_region is not None and next_region != cur_region:
                try:
                    env.switch_region(int(next_region))
                except Exception:
                    pass

        return ClusterType.NONE