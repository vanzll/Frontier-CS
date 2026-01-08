import json
import math
from argparse import Namespace

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

        self._inited = False
        self._seen_done_len = 0
        self._work_done = 0.0
        self._global_step = 0

        # Heuristic parameters
        self._ucb_c = 0.25
        self._recent_bonus = 0.12

        # Cost parameters (used only for switch heuristics)
        self._od_rate = 3.06
        self._spot_rate = 0.9701

        return self

    def _lazy_init(self) -> None:
        if self._inited:
            return
        n = int(self.env.get_num_regions())
        self._num_regions = n
        self._avail = [0] * n
        self._total = [0] * n
        self._last_seen_step = [-10**12] * n
        self._last_has_spot = [0] * n
        self._inited = True

    def _update_work_done(self) -> None:
        td = self.task_done_time
        i = self._seen_done_len
        L = len(td)
        if i < L:
            s = 0.0
            for j in range(i, L):
                s += float(td[j])
            self._work_done += s
            self._seen_done_len = L

    def _steps_left(self) -> int:
        gap = float(self.env.gap_seconds)
        tl = float(self.deadline - self.env.elapsed_seconds)
        if tl <= 0:
            return 0
        # tl should be multiple of gap; use floor
        return int((tl + 1e-9) // gap)

    def _steps_needed(self, work_left: float, start_overhead: float) -> int:
        if work_left <= 1e-9:
            return 0
        gap = float(self.env.gap_seconds)
        if gap <= 0:
            return 10**18
        h = max(0.0, float(start_overhead))
        return int(math.ceil((work_left + h) / gap - 1e-12))

    def _step_work_if_run(self, restart_overhead: float) -> float:
        gap = float(self.env.gap_seconds)
        h = max(0.0, float(restart_overhead))
        if h >= gap:
            return 0.0
        return gap - h

    def _region_score(self, idx: int) -> float:
        t = self._total[idx]
        a = self._avail[idx]
        p = (a + 1.0) / (t + 2.0)
        bonus = 0.0
        if self._last_has_spot[idx]:
            age = max(0, self._global_step - self._last_seen_step[idx])
            bonus = self._recent_bonus / (1.0 + age / 4.0)
        exploration = self._ucb_c * math.sqrt(math.log(self._global_step + 2.0) / (t + 1.0))
        return p + bonus + exploration

    def _pick_best_region(self, exclude: int | None = None) -> int:
        n = self._num_regions
        if n <= 1:
            return 0
        best_idx = None
        best_score = -1e30
        for i in range(n):
            if exclude is not None and i == exclude:
                continue
            sc = self._region_score(i)
            if sc > best_score:
                best_score = sc
                best_idx = i
        if best_idx is None:
            return int(exclude) if exclude is not None else 0
        return int(best_idx)

    def _safe_to_pause_one_step(self, work_left: float, steps_left: int) -> bool:
        if steps_left <= 1:
            return False
        # After pausing 1 step, worst-case we must start ON_DEMAND with restart overhead
        needed = self._steps_needed(work_left, self.restart_overhead)
        return needed <= (steps_left - 1)

    def _safe_to_switch_od_to_spot(self, work_left: float, steps_left: int) -> bool:
        if steps_left <= 0:
            return False
        # If we switch to spot now, we pay restart overhead now and might need to switch
        # back to on-demand next step (restart overhead again).
        spot_work = self._step_work_if_run(self.restart_overhead)
        work_after = work_left - spot_work
        if work_after <= 1e-9:
            return True
        if steps_left <= 1:
            return False
        needed_after = self._steps_needed(work_after, self.restart_overhead)
        return needed_after <= (steps_left - 1)

    def _economic_to_switch_od_to_spot(self) -> bool:
        gap = float(self.env.gap_seconds)
        if gap <= 0:
            return False
        gap_hours = gap / 3600.0
        overhead_hours = float(self.restart_overhead) / 3600.0
        savings = (self._od_rate - self._spot_rate) * gap_hours
        worst_case_extra = self._od_rate * overhead_hours
        return savings > worst_case_extra + 1e-12

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._lazy_init()
        self._update_work_done()
        self._global_step += 1

        work_left = float(self.task_duration - self._work_done)
        if work_left <= 1e-6:
            return ClusterType.NONE

        steps_left = self._steps_left()
        if steps_left <= 0:
            return ClusterType.NONE

        cur_region = int(self.env.get_current_region())
        self._total[cur_region] += 1
        if has_spot:
            self._avail[cur_region] += 1
            self._last_has_spot[cur_region] = 1
        else:
            self._last_has_spot[cur_region] = 0
        self._last_seen_step[cur_region] = self._global_step

        # If even continuous ON_DEMAND from now can't finish, run ON_DEMAND and hope.
        od_start_overhead_now = float(self.remaining_restart_overhead) if last_cluster_type == ClusterType.ON_DEMAND else float(self.restart_overhead)
        if self._steps_needed(work_left, od_start_overhead_now) > steps_left:
            return ClusterType.ON_DEMAND

        # Primary policy: use SPOT whenever available (unless switching from ON_DEMAND is unsafe/uneconomic)
        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND:
                if (not self._safe_to_switch_od_to_spot(work_left, steps_left)) or (not self._economic_to_switch_od_to_spot()):
                    return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        # No spot available: prefer to pause if it is safe; switch region while paused to hunt for spot
        if self._safe_to_pause_one_step(work_left, steps_left):
            if self._num_regions > 1:
                nxt = self._pick_best_region(exclude=cur_region)
                if nxt != cur_region:
                    self.env.switch_region(nxt)
            return ClusterType.NONE

        # Must make progress now: run ON_DEMAND.
        # If we're restarting anyway (not currently ON_DEMAND), we can also switch regions "for free".
        if last_cluster_type != ClusterType.ON_DEMAND and self._num_regions > 1:
            nxt = self._pick_best_region(exclude=None)
            if nxt != cur_region:
                self.env.switch_region(nxt)
        return ClusterType.ON_DEMAND