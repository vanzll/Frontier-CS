import json
import math
from argparse import Namespace
from typing import Optional, List

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


def _scalar(x) -> float:
    if isinstance(x, (list, tuple)):
        return float(x[0]) if x else 0.0
    return float(x)


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

        self._inited = False
        self._force_ondemand = False

        self._num_regions = 0
        self._visits: List[int] = []
        self._hits: List[int] = []
        self._ema: List[float] = []

        self._step_count = 0
        self._total_obs = 0

        self._last_region: Optional[int] = None
        self._last_switch_step = -10**18

        self._spot_streak = 0
        self._no_spot_streak = 0

        self._done_sum = 0.0
        self._done_len = 0

        self._task_duration_s = 0.0
        self._deadline_s = 0.0
        self._overhead_s = 0.0

        self._overhead_steps = 1
        self._min_switch_interval = 1
        self._spot_confirm_steps = 1

        self._ucb_c = 0.25
        self._ema_alpha = 0.03

        return self

    def _ensure_init(self) -> None:
        if self._inited:
            return
        self._task_duration_s = _scalar(getattr(self, "task_duration", 0.0))
        self._deadline_s = _scalar(getattr(self, "deadline", 0.0))
        self._overhead_s = _scalar(getattr(self, "restart_overhead", 0.0))

        self._num_regions = int(self.env.get_num_regions())
        self._visits = [0] * self._num_regions
        self._hits = [0] * self._num_regions
        self._ema = [0.5] * self._num_regions

        gap = float(getattr(self.env, "gap_seconds", 1.0)) or 1.0
        self._overhead_steps = max(1, int(math.ceil(self._overhead_s / gap)))
        self._min_switch_interval = self._overhead_steps
        self._spot_confirm_steps = max(1, min(3, max(1, self._overhead_steps // 3)))

        self._inited = True

    def _update_done(self) -> float:
        td = getattr(self, "task_done_time", None)
        if not td:
            self._done_sum = 0.0
            self._done_len = 0
            return 0.0
        if self._done_len > len(td):
            self._done_sum = 0.0
            self._done_len = 0
        n = len(td)
        while self._done_len < n:
            self._done_sum += float(td[self._done_len])
            self._done_len += 1
        return self._done_sum

    def _pick_best_region(self) -> int:
        total = max(1, self._total_obs + 1)
        logt = math.log(total + 1.0)

        best_i = 0
        best_score = -1e18
        for i in range(self._num_regions):
            v = self._visits[i]
            h = self._hits[i]
            mean = (h + 1.0) / (v + 2.0)
            bonus = self._ucb_c * math.sqrt(logt / (v + 1.0))
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_i = i
        return best_i

    def _maybe_switch_region(self, target: int) -> None:
        cur = int(self.env.get_current_region())
        if target == cur:
            return
        if (self._step_count - self._last_switch_step) < self._min_switch_interval:
            return
        self.env.switch_region(int(target))
        self._last_switch_step = self._step_count
        self._spot_streak = 0
        self._no_spot_streak = 0
        self._last_region = int(target)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._ensure_init()
        self._step_count += 1

        gap = float(getattr(self.env, "gap_seconds", 1.0)) or 1.0
        now = float(getattr(self.env, "elapsed_seconds", 0.0))

        cur_region = int(self.env.get_current_region())
        if self._last_region is None:
            self._last_region = cur_region
        elif self._last_region != cur_region:
            self._spot_streak = 0
            self._no_spot_streak = 0
            self._last_region = cur_region

        self._visits[cur_region] += 1
        self._total_obs += 1
        if has_spot:
            self._hits[cur_region] += 1
            self._ema[cur_region] = (1.0 - self._ema_alpha) * self._ema[cur_region] + self._ema_alpha * 1.0
            self._spot_streak += 1
            self._no_spot_streak = 0
        else:
            self._ema[cur_region] = (1.0 - self._ema_alpha) * self._ema[cur_region] + self._ema_alpha * 0.0
            self._no_spot_streak += 1
            self._spot_streak = 0

        done = self._update_done()
        remaining_work = max(0.0, self._task_duration_s - done)
        if remaining_work <= 1e-9:
            return ClusterType.NONE

        remaining_time = max(0.0, self._deadline_s - now)
        slack = remaining_time - remaining_work

        critical_slack = max(2.0 * self._overhead_s, gap)
        if slack <= critical_slack:
            self._force_ondemand = True

        if self._force_ondemand:
            return ClusterType.ON_DEMAND

        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND:
                if slack > 2.0 * self._overhead_s and self._spot_streak >= self._spot_confirm_steps:
                    return ClusterType.SPOT
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        can_idle = (remaining_work + self._overhead_s) <= (remaining_time - gap)
        if can_idle:
            target = self._pick_best_region()
            self._maybe_switch_region(target)
            return ClusterType.NONE

        if last_cluster_type != ClusterType.ON_DEMAND:
            target = self._pick_best_region()
            self._maybe_switch_region(target)
            return ClusterType.ON_DEMAND

        if self._no_spot_streak >= max(2, self._overhead_steps) and slack > 6.0 * self._overhead_s:
            target = self._pick_best_region()
            self._maybe_switch_region(target)

        return ClusterType.ON_DEMAND