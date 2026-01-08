import json
import math
from argparse import Namespace
from typing import Optional

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "my_strategy"

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
        self._work_done = 0.0
        self._task_done_len = 0

        self._succ = None
        self._tot = None
        self._t = 0

        self._ucb_c = 0.7
        self._cooldown_steps = 3
        self._guarded_slack = 4 * 3600.0
        self._panic_extra = 0.0

        self._prev_choice = ClusterType.NONE
        self._on_demand_steps = 0
        return self

    def _lazy_init(self) -> None:
        r = int(self.env.get_num_regions())
        self._succ = [0] * r
        self._tot = [0] * r
        self._t = 0

        gap = float(self.env.gap_seconds)
        ro = float(self.restart_overhead)
        self._cooldown_steps = max(2, int(math.ceil(ro / max(gap, 1e-9))) + 1)

        guarded_hours = 4.5 / max(1.0, math.sqrt(float(r)))
        guarded_hours = max(1.5, min(5.0, guarded_hours))
        self._guarded_slack = guarded_hours * 3600.0

        self._panic_extra = 2.0 * gap + 2.0 * ro

        self._initialized = True

    def _update_work_done(self) -> None:
        td = self.task_done_time
        n = len(td)
        i = self._task_done_len
        if i < n:
            s = 0.0
            for j in range(i, n):
                s += float(td[j])
            self._work_done += s
            self._task_done_len = n

    def _select_next_region(self, cur: int) -> Optional[int]:
        r = len(self._tot)
        if r <= 1:
            return None

        best = None
        best_score = -1e100
        logt = math.log(max(2, self._t + 1))

        for i in range(r):
            if i == cur:
                continue
            n = self._tot[i]
            if n <= 0:
                score = 1e18
            else:
                mean = self._succ[i] / n
                score = mean + self._ucb_c * math.sqrt(logt / n)
            if score > best_score:
                best_score = score
                best = i
        return best

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._initialized:
            self._lazy_init()

        self._update_work_done()
        if self._work_done >= float(self.task_duration) - 1e-9:
            self._prev_choice = ClusterType.NONE
            self._on_demand_steps = 0
            return ClusterType.NONE

        cur = int(self.env.get_current_region())

        self._t += 1
        self._tot[cur] += 1
        if has_spot:
            self._succ[cur] += 1

        elapsed = float(self.env.elapsed_seconds)
        remaining_work = float(self.task_duration) - self._work_done
        time_left = float(self.deadline) - elapsed

        if last_cluster_type == ClusterType.ON_DEMAND:
            min_od_finish = remaining_work + float(self.remaining_restart_overhead)
        else:
            min_od_finish = remaining_work + float(self.restart_overhead)

        if time_left <= min_od_finish + self._panic_extra:
            action = ClusterType.ON_DEMAND
        else:
            slack = time_left - remaining_work
            if slack <= self._guarded_slack:
                if has_spot:
                    if self._prev_choice == ClusterType.ON_DEMAND and self._on_demand_steps < self._cooldown_steps:
                        action = ClusterType.ON_DEMAND
                    else:
                        action = ClusterType.SPOT
                else:
                    action = ClusterType.ON_DEMAND
            else:
                if has_spot:
                    action = ClusterType.SPOT
                else:
                    if self.env.get_num_regions() > 1:
                        nxt = self._select_next_region(cur)
                        if nxt is not None and nxt != cur:
                            self.env.switch_region(int(nxt))
                    action = ClusterType.NONE

        if action == ClusterType.ON_DEMAND:
            if self._prev_choice == ClusterType.ON_DEMAND:
                self._on_demand_steps += 1
            else:
                self._on_demand_steps = 1
        else:
            self._on_demand_steps = 0

        self._prev_choice = action
        return action