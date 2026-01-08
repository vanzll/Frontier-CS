import json
from argparse import Namespace
from typing import Callable, List, Optional, Sequence, Any

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

        self._initialized = False
        self._num_regions = 0
        self._gap = 0.0

        self._work_done_total = 0.0
        self._last_done_len = 0

        self._od_mode = False

        self._any_spot_ema = 0.5
        self._alpha_any = 0.02

        self._spot_query_all: Optional[Callable[[], List[bool]]] = None

        self._ema_avail: Optional[List[float]] = None
        self._streak: Optional[List[int]] = None
        self._ema_run: Optional[List[float]] = None
        self._prev_spot: Optional[List[bool]] = None
        self._alpha_avail = 0.02
        self._beta_run = 0.08

        self._safety_buffer = 0.0
        self._switch_penalty = 0.03

        self._prev_overhead: Optional[float] = None
        self._last_action: Optional[ClusterType] = None

        return self

    def _as_scalar_seconds(self, v: Any) -> float:
        if isinstance(v, (list, tuple)) and v:
            return float(v[0])
        return float(v)

    def _try_make_query_all(self) -> Optional[Callable[[], List[bool]]]:
        env = self.env
        n = self._num_regions

        def _is_bool_seq(x: Any) -> bool:
            if not isinstance(x, (list, tuple)):
                return False
            if len(x) != n:
                return False
            for y in x:
                if not isinstance(y, (bool, int)):
                    return False
            return True

        # 1) Callable returning list of bools
        for name in (
            "get_spot_availabilities",
            "get_spot_availability",
            "get_all_spot_availabilities",
            "spot_availabilities",
            "get_has_spot_all",
            "get_has_spot_list",
            "get_has_spot_by_region",
        ):
            if not hasattr(env, name):
                continue
            attr = getattr(env, name)
            if callable(attr):
                try:
                    res = attr()
                    if _is_bool_seq(res):
                        def q(_attr=attr):
                            r = _attr()
                            return [bool(x) for x in r]
                        return q
                except Exception:
                    pass
            else:
                try:
                    if _is_bool_seq(attr):
                        def q(_attr=attr):
                            return [bool(x) for x in _attr]
                        return q
                except Exception:
                    pass

        # 2) Callable taking region idx
        for name in (
            "get_has_spot",
            "has_spot_in_region",
            "spot_available",
            "get_spot",
            "get_spot_available",
            "get_spot_status",
            "has_spot",
        ):
            if not hasattr(env, name):
                continue
            attr = getattr(env, name)
            if not callable(attr):
                continue
            try:
                test = attr(0)
                if isinstance(test, (bool, int)):
                    def q(_attr=attr):
                        out = [False] * n
                        for i in range(n):
                            out[i] = bool(_attr(i))
                        return out
                    return q
            except Exception:
                continue

        # 3) Look for common attribute holding a sequence
        for name in (
            "_has_spot",
            "_spot_availabilities",
            "current_spot_availabilities",
            "spot_availability",
            "has_spot_by_region",
        ):
            if not hasattr(env, name):
                continue
            try:
                attr = getattr(env, name)
                if _is_bool_seq(attr):
                    def q(_attr=attr):
                        return [bool(x) for x in _attr]
                    return q
            except Exception:
                pass

        return None

    def _init_once(self) -> None:
        self._gap = float(getattr(self.env, "gap_seconds"))
        self._num_regions = int(self.env.get_num_regions())

        self._spot_query_all = self._try_make_query_all()

        self._ema_avail = [0.5] * self._num_regions
        self._streak = [0] * self._num_regions
        self._ema_run = [2.0] * self._num_regions
        self._prev_spot = [False] * self._num_regions

        restart_overhead = self._as_scalar_seconds(getattr(self, "restart_overhead"))
        gap = self._gap

        base = 2.0 * restart_overhead + 2.0 * gap
        base = max(base, 0.5 * 3600.0)
        base = min(base, 6.0 * 3600.0)
        self._safety_buffer = base

        self._initialized = True

    def _update_work_done_total(self) -> None:
        tdt = self.task_done_time
        ln = len(tdt)
        if ln > self._last_done_len:
            for i in range(self._last_done_len, ln):
                self._work_done_total += float(tdt[i])
            self._last_done_len = ln

    def _update_spot_stats(self, avail: Optional[List[bool]], current_region: int, has_spot: bool) -> bool:
        any_spot = False
        if avail is None:
            any_spot = bool(has_spot)
            # update only current region stats
            i = current_region
            cur = bool(has_spot)
            prev = self._prev_spot[i]
            self._ema_avail[i] = (1.0 - self._alpha_avail) * self._ema_avail[i] + self._alpha_avail * (1.0 if cur else 0.0)
            if cur:
                self._streak[i] += 1
            else:
                if prev and self._streak[i] > 0:
                    self._ema_run[i] = (1.0 - self._beta_run) * self._ema_run[i] + self._beta_run * float(self._streak[i])
                self._streak[i] = 0
            self._prev_spot[i] = cur
            return any_spot

        any_spot = any(avail)
        for i, cur in enumerate(avail):
            cur = bool(cur)
            prev = self._prev_spot[i]
            self._ema_avail[i] = (1.0 - self._alpha_avail) * self._ema_avail[i] + self._alpha_avail * (1.0 if cur else 0.0)
            if cur:
                self._streak[i] += 1
            else:
                if prev and self._streak[i] > 0:
                    self._ema_run[i] = (1.0 - self._beta_run) * self._ema_run[i] + self._beta_run * float(self._streak[i])
                self._streak[i] = 0
            self._prev_spot[i] = cur
        return any_spot

    def _pick_best_spot_region(self, avail: List[bool], current_region: int) -> int:
        best = current_region
        best_score = -1e30
        for i, ok in enumerate(avail):
            if not ok:
                continue
            run_norm = self._ema_run[i] / (self._ema_run[i] + 8.0)
            score = 2.2 * self._ema_avail[i] + 0.8 * run_norm + 0.02 * min(self._streak[i], 50)
            if i == current_region:
                score += 0.12
            else:
                score -= self._switch_penalty
            if score > best_score:
                best_score = score
                best = i
        return best

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._initialized:
            self._init_once()

        self._update_work_done_total()

        task_duration = self._as_scalar_seconds(getattr(self, "task_duration"))
        deadline = self._as_scalar_seconds(getattr(self, "deadline"))
        restart_overhead = self._as_scalar_seconds(getattr(self, "restart_overhead"))
        remaining_overhead = float(getattr(self, "remaining_restart_overhead"))

        remaining_work = task_duration - self._work_done_total
        if remaining_work <= 1e-9:
            self._last_action = ClusterType.NONE
            self._prev_overhead = remaining_overhead
            return ClusterType.NONE

        now = float(getattr(self.env, "elapsed_seconds"))
        time_left = deadline - now
        if time_left <= 1e-9:
            self._last_action = ClusterType.ON_DEMAND
            self._prev_overhead = remaining_overhead
            return ClusterType.ON_DEMAND

        current_region = int(self.env.get_current_region())

        avail = None
        if self._spot_query_all is not None:
            try:
                avail = self._spot_query_all()
                if not (isinstance(avail, list) and len(avail) == self._num_regions):
                    avail = None
            except Exception:
                avail = None

        any_spot = self._update_spot_stats(avail, current_region, has_spot)
        self._any_spot_ema = (1.0 - self._alpha_any) * self._any_spot_ema + self._alpha_any * (1.0 if any_spot else 0.0)

        # Conservative time needed if we commit to on-demand now and never switch again
        need_commit_od = remaining_work + remaining_overhead
        if last_cluster_type != ClusterType.ON_DEMAND:
            need_commit_od += restart_overhead

        # Enter on-demand mode near the end to guarantee completion; never exit.
        if (not self._od_mode) and (time_left <= need_commit_od + self._safety_buffer):
            self._od_mode = True

        if self._od_mode:
            self._last_action = ClusterType.ON_DEMAND
            self._prev_overhead = remaining_overhead
            return ClusterType.ON_DEMAND

        # If restart overhead is pending, avoid switching/launching to prevent resetting overhead.
        # Prefer waiting it out unless it's not decreasing on NONE.
        eps = 1e-9
        if remaining_overhead > eps and last_cluster_type != ClusterType.ON_DEMAND:
            if self._last_action == ClusterType.NONE and self._prev_overhead is not None:
                if remaining_overhead >= self._prev_overhead - eps:
                    # overhead didn't decrease while waiting; proceed to run if possible
                    pass
                else:
                    self._last_action = ClusterType.NONE
                    self._prev_overhead = remaining_overhead
                    return ClusterType.NONE
            else:
                self._last_action = ClusterType.NONE
                self._prev_overhead = remaining_overhead
                return ClusterType.NONE

        # Spot mode: use spot if available in some region, else pause.
        if avail is None:
            if has_spot:
                self._last_action = ClusterType.SPOT
                self._prev_overhead = remaining_overhead
                return ClusterType.SPOT
            self._last_action = ClusterType.NONE
            self._prev_overhead = remaining_overhead
            return ClusterType.NONE

        if any_spot:
            target = self._pick_best_spot_region(avail, current_region)
            if target != current_region:
                try:
                    self.env.switch_region(target)
                    current_region = target
                except Exception:
                    pass

            # Safety: only return SPOT if we are confident the chosen region has spot.
            # If switch failed or availability unknown, fall back to NONE.
            if 0 <= current_region < self._num_regions and bool(avail[current_region]):
                self._last_action = ClusterType.SPOT
                self._prev_overhead = remaining_overhead
                return ClusterType.SPOT

            self._last_action = ClusterType.NONE
            self._prev_overhead = remaining_overhead
            return ClusterType.NONE

        self._last_action = ClusterType.NONE
        self._prev_overhead = remaining_overhead
        return ClusterType.NONE