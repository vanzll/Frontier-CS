import json
import math
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_ucb_pause"

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

        self._mr_inited = False
        return self

    @staticmethod
    def _scalar(x: object) -> float:
        if isinstance(x, (list, tuple)):
            return float(x[0])
        return float(x)

    def _lazy_init(self) -> None:
        if self._mr_inited:
            return
        self._mr_inited = True

        try:
            self._num_regions = int(self.env.get_num_regions())
        except Exception:
            self._num_regions = 1

        self._timestep = 0
        self._work_done = 0.0
        self._last_task_done_len = 0

        self._ema_alpha = 0.08
        self._ucb_c = 0.9
        self._ema_spot = [0.5] * self._num_regions
        self._obs = [0] * self._num_regions
        self._rr = 0

        self._locked_on_demand = False

    def _update_work_done(self) -> None:
        td = self.task_done_time
        n = len(td)
        if n > self._last_task_done_len:
            self._work_done += float(sum(td[self._last_task_done_len:n]))
            self._last_task_done_len = n

    def _select_next_region(self, exclude: int) -> int:
        rcount = self._num_regions
        if rcount <= 1:
            return exclude

        logt = math.log(self._timestep + 2.0)
        start = self._rr % rcount
        self._rr += 1

        best_r = exclude
        best_score = -1e30

        for i in range(rcount):
            r = (start + i) % rcount
            if r == exclude:
                continue
            n = self._obs[r]
            score = self._ema_spot[r] + self._ucb_c * math.sqrt(logt / (n + 1.0))
            if score > best_score + 1e-12:
                best_score = score
                best_r = r
        return best_r

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._lazy_init()
        self._timestep += 1

        region = int(self.env.get_current_region())
        if 0 <= region < self._num_regions:
            x = 1.0 if has_spot else 0.0
            a = self._ema_alpha
            self._ema_spot[region] = (1.0 - a) * self._ema_spot[region] + a * x
            self._obs[region] += 1

        self._update_work_done()

        task_duration = self._scalar(getattr(self, "task_duration", 0.0))
        deadline = self._scalar(getattr(self, "deadline", 0.0))
        restart_overhead = self._scalar(getattr(self, "restart_overhead", 0.0))

        remaining_work = task_duration - self._work_done
        if remaining_work <= 1e-9:
            return ClusterType.NONE

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        remaining_time = deadline - elapsed
        if remaining_time <= 0:
            return ClusterType.ON_DEMAND

        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        rem_oh = float(getattr(self, "remaining_restart_overhead", 0.0) or 0.0)

        overhead_for_safety = max(restart_overhead, rem_oh)

        margin = gap + 2.0 * restart_overhead + 1e-9

        if self._locked_on_demand:
            return ClusterType.ON_DEMAND

        if remaining_time <= remaining_work + overhead_for_safety + margin:
            self._locked_on_demand = True
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        can_pause_one_step = (remaining_time - gap) >= (remaining_work + restart_overhead + margin)

        if can_pause_one_step:
            if self._num_regions > 1:
                nxt = self._select_next_region(exclude=region)
                if nxt != region:
                    self.env.switch_region(int(nxt))
            return ClusterType.NONE

        return ClusterType.ON_DEMAND