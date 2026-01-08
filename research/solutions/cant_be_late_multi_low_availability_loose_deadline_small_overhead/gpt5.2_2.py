import json
import math
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


def _cluster_type(name: str):
    if hasattr(ClusterType, name):
        return getattr(ClusterType, name)
    # Fallback for potential different enum naming
    for attr in ("NONE", "None", "none"):
        if hasattr(ClusterType, attr) and attr.lower() == name.lower():
            return getattr(ClusterType, attr)
    raise AttributeError(f"ClusterType missing {name}")


CT_SPOT = _cluster_type("SPOT")
CT_OD = _cluster_type("ON_DEMAND")
CT_NONE = _cluster_type("NONE")


def _scalar(x):
    if isinstance(x, (list, tuple)):
        return float(x[0])
    return float(x)


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_mr_v1"

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

        self._done_total = 0.0
        self._done_len = 0

        self._mr_inited = False
        self._alpha = 0.08
        self._ucb_c = 0.25
        self._total_obs = 0

        self._od_committed = False

        self._eps = 1.0  # seconds margin
        return self

    def _ensure_mr(self):
        n = int(self.env.get_num_regions())
        if (not self._mr_inited) or (len(self._score) != n):
            self._score = [0.5] * n
            self._obs = [0] * n
            self._total_obs = 0
            self._mr_inited = True

    def _update_done_cache(self) -> float:
        tdt = self.task_done_time
        l = len(tdt)
        if l == self._done_len:
            return self._done_total
        if l < self._done_len:
            self._done_total = float(sum(tdt))
            self._done_len = l
            return self._done_total
        self._done_total += float(sum(tdt[self._done_len:l]))
        self._done_len = l
        return self._done_total

    def _update_region_stats(self, region: int, has_spot: bool):
        x = 1.0 if has_spot else 0.0
        s = self._score[region]
        self._score[region] = s + self._alpha * (x - s)
        self._obs[region] += 1
        self._total_obs += 1

    def _pick_region_ucb(self, exclude: int = -1) -> int:
        n = len(self._score)
        if n <= 1:
            return 0
        t = self._total_obs + n + 1
        logt = math.log(float(t))

        best_i = 0 if exclude != 0 else 1
        best_val = -1e30
        for i in range(n):
            if i == exclude:
                continue
            obs = self._obs[i]
            ucb = self._score[i] + self._ucb_c * math.sqrt(logt / float(obs + 1))
            if ucb > best_val:
                best_val = ucb
                best_i = i
        return best_i

    def _work_gain_if_run(self, dt: float, last_cluster_type: ClusterType, action: ClusterType) -> float:
        if action == CT_NONE:
            return 0.0
        ro = _scalar(self.restart_overhead)
        pending = float(getattr(self, "remaining_restart_overhead", 0.0) or 0.0)
        restart_now = (action != last_cluster_type)
        overhead = ro if restart_now else pending
        if overhead < 0.0:
            overhead = 0.0
        if overhead > dt:
            overhead = dt
        gain = dt - overhead
        if gain < 0.0:
            gain = 0.0
        return gain

    def _safe_after_action_worstcase_od(self, remaining_time: float, remaining_work: float, dt: float,
                                        last_cluster_type: ClusterType, action: ClusterType) -> bool:
        gain = self._work_gain_if_run(dt, last_cluster_type, action)
        rt2 = remaining_time - dt
        rw2 = remaining_work - gain
        if rw2 <= self._eps:
            return True
        if rt2 <= 0.0:
            return False
        ro = _scalar(self.restart_overhead)
        # Worst case: need to run on-demand from next step onward.
        future_overhead = 0.0 if action == CT_OD else ro
        return rt2 + 1e-9 >= rw2 + future_overhead + self._eps

    def _can_wait_one_step(self, remaining_time: float, remaining_work: float, dt: float) -> bool:
        rt2 = remaining_time - dt
        if rt2 <= 0.0:
            return False
        ro = _scalar(self.restart_overhead)
        return rt2 + 1e-9 >= remaining_work + ro + self._eps

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._ensure_mr()
        cur_region = int(self.env.get_current_region())

        # Update region stats for the region whose availability we were told.
        self._update_region_stats(cur_region, has_spot)

        done = self._update_done_cache()
        task_dur = _scalar(self.task_duration)
        deadline = _scalar(self.deadline)
        remaining_work = task_dur - done
        if remaining_work <= self._eps:
            return CT_NONE

        remaining_time = deadline - float(self.env.elapsed_seconds)
        if remaining_time <= 0.0:
            return CT_NONE

        gap = float(self.env.gap_seconds)
        dt = gap if remaining_time >= gap else remaining_time

        # If we've committed to on-demand for safety, keep it simple and reliable.
        if self._od_committed:
            return CT_OD

        # If spot is available in the current region and using it keeps a safe fallback path, use it.
        if has_spot:
            if self._safe_after_action_worstcase_od(remaining_time, remaining_work, dt, last_cluster_type, CT_SPOT):
                return CT_SPOT
            self._od_committed = True
            return CT_OD

        # No spot available in current region.
        # Prefer waiting (NONE) if we can still finish by switching to on-demand afterwards.
        if self._can_wait_one_step(remaining_time, remaining_work, dt):
            if self.env.get_num_regions() > 1:
                target = self._pick_region_ucb(exclude=cur_region)
                if target != cur_region:
                    self.env.switch_region(int(target))
            return CT_NONE

        # Must make progress: switch to on-demand and commit for reliability.
        self._od_committed = True
        return CT_OD