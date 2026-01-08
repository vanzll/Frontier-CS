import json
import math
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_multiregion_v1"

    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path, "r") as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        self._cached_done_sum = 0.0
        self._cached_done_len = 0

        self._num_regions = None
        self._region_obs = None
        self._region_avail = None

        self._od_lock = False

        self._alpha = 1.0
        self._ucb_c = 0.35

        self._task_duration_s = None
        self._deadline_s = None
        self._restart_overhead_s = None

        return self

    def _as_scalar(self, x, idx: int = 0) -> float:
        if isinstance(x, (list, tuple)):
            if not x:
                return 0.0
            return float(x[idx])
        return float(x)

    def _ensure_params_cached(self) -> None:
        if self._task_duration_s is None:
            self._task_duration_s = self._as_scalar(getattr(self, "task_duration", 0.0))
        if self._deadline_s is None:
            self._deadline_s = self._as_scalar(getattr(self, "deadline", 0.0))
        if self._restart_overhead_s is None:
            self._restart_overhead_s = self._as_scalar(getattr(self, "restart_overhead", 0.0))

    def _ensure_region_stats(self) -> None:
        if self._num_regions is None:
            try:
                self._num_regions = int(self.env.get_num_regions())
            except Exception:
                self._num_regions = 1
            if self._num_regions <= 0:
                self._num_regions = 1
            self._region_obs = [0] * self._num_regions
            self._region_avail = [0] * self._num_regions

    def _update_work_done_cache(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if not isinstance(tdt, list):
            self._cached_done_sum = float(tdt or 0.0)
            self._cached_done_len = 0
            return self._cached_done_sum

        n = len(tdt)
        if n < self._cached_done_len:
            self._cached_done_sum = 0.0
            self._cached_done_len = 0
        if n > self._cached_done_len:
            self._cached_done_sum += sum(tdt[self._cached_done_len : n])
            self._cached_done_len = n
        return self._cached_done_sum

    def _time_needed_if_run_od_from_now(self, remaining_work: float, last_cluster_type: ClusterType) -> float:
        if last_cluster_type == ClusterType.ON_DEMAND:
            overhead = float(getattr(self, "remaining_restart_overhead", 0.0) or 0.0)
        else:
            overhead = self._restart_overhead_s
        return remaining_work + max(0.0, overhead)

    def _safe_to_wait_one_step_then_od(self, remaining_work: float, time_left: float, gap: float) -> bool:
        if time_left <= gap:
            return False
        # After waiting one step, we will need to start ON_DEMAND from NONE => pay full restart overhead.
        return (time_left - gap) + 1e-9 >= remaining_work + self._restart_overhead_s

    def _should_avoid_starting_spot_due_to_double_overhead(
        self, remaining_work: float, time_left: float, gap: float
    ) -> bool:
        # If we are not currently on spot and start spot now, we may pay overhead now,
        # and might need to pay another overhead switching to on-demand next step.
        # Worst-case: spot disappears immediately next step.
        work_gain_now = max(0.0, gap - self._restart_overhead_s)
        remaining_after_now = max(0.0, remaining_work - work_gain_now)
        need_after_now_if_switch_to_od_next = remaining_after_now + self._restart_overhead_s
        return need_after_now_if_switch_to_od_next > (time_left - gap) + 1e-9

    def _pick_best_region(self, current_region: int) -> int:
        nr = self._num_regions
        if nr <= 1:
            return current_region

        total_obs = 0
        obs = self._region_obs
        avail = self._region_avail
        for i in range(nr):
            total_obs += obs[i]
        log_term = math.log(total_obs + 2.0)

        best_r = current_region
        best_score = -1e30

        alpha = self._alpha
        c = self._ucb_c
        for r in range(nr):
            o = obs[r]
            a = avail[r]
            p = (a + alpha) / (o + 2.0 * alpha)
            bonus = c * math.sqrt(log_term / (o + 1.0))
            score = p + bonus
            if score > best_score:
                best_score = score
                best_r = r
        return best_r

    def _maybe_switch_region_while_waiting(self) -> None:
        try:
            cur = int(self.env.get_current_region())
        except Exception:
            return
        best = self._pick_best_region(cur)
        if best != cur:
            try:
                self.env.switch_region(best)
            except Exception:
                pass

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._ensure_params_cached()
        self._ensure_region_stats()

        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        time_left = self._deadline_s - elapsed

        work_done = self._update_work_done_cache()
        remaining_work = self._task_duration_s - work_done
        if remaining_work <= 1e-9:
            return ClusterType.NONE
        if time_left <= 1e-9:
            return ClusterType.NONE

        try:
            cur_region = int(self.env.get_current_region())
        except Exception:
            cur_region = 0
        if 0 <= cur_region < self._num_regions:
            self._region_obs[cur_region] += 1
            if has_spot:
                self._region_avail[cur_region] += 1

        if self._od_lock:
            return ClusterType.ON_DEMAND

        if has_spot:
            if last_cluster_type == ClusterType.SPOT:
                return ClusterType.SPOT

            # Starting spot from non-spot can create a "double overhead" scenario near the end.
            if gap > 0.0 and time_left > gap and self._should_avoid_starting_spot_due_to_double_overhead(
                remaining_work, time_left, gap
            ):
                self._od_lock = True
                return ClusterType.ON_DEMAND

            return ClusterType.SPOT

        # No spot available.
        if gap <= 0.0:
            self._od_lock = True
            return ClusterType.ON_DEMAND

        if self._safe_to_wait_one_step_then_od(remaining_work, time_left, gap):
            self._maybe_switch_region_while_waiting()
            return ClusterType.NONE

        self._od_lock = True
        return ClusterType.ON_DEMAND