import json
from argparse import Namespace
from typing import Callable, Optional, Sequence, Any, List

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_v1"

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

        self._num_regions = 1
        self._obs = []
        self._avail = []

        self._done_cache = 0.0
        self._done_len = 0

        self._switch_cooldown = 0
        self._switch_cooldown_reset = 3

        self._step_count = 0

        self._od_buffer = 0.0
        self._peek_has_spot: Optional[Callable[[int], Optional[bool]]] = None
        self._peek_calibrated = False

        return self

    def _as_float(self, x: Any) -> float:
        if isinstance(x, (list, tuple)):
            return float(x[0]) if x else 0.0
        return float(x)

    def _get_task_done_list(self) -> List[float]:
        tdt = getattr(self, "task_done_time", None)
        if tdt is None:
            return []
        if isinstance(tdt, (list, tuple)) and tdt and isinstance(tdt[0], (list, tuple)):
            return list(tdt[0])
        return list(tdt)

    def _update_done_cache(self) -> float:
        tdt = self._get_task_done_list()
        n = len(tdt)
        if n < self._done_len:
            self._done_cache = 0.0
            self._done_len = 0
        if n > self._done_len:
            s = self._done_cache
            for i in range(self._done_len, n):
                try:
                    s += float(tdt[i])
                except Exception:
                    pass
            self._done_cache = s
            self._done_len = n
        return self._done_cache

    def _ensure_init(self, has_spot: bool) -> None:
        if self._inited:
            if not self._peek_calibrated:
                self._calibrate_peek(has_spot)
            return

        try:
            self._num_regions = int(self.env.get_num_regions())
        except Exception:
            self._num_regions = 1

        n = max(1, self._num_regions)
        self._obs = [2] * n
        self._avail = [1] * n

        gap = float(getattr(self.env, "gap_seconds", 1.0))
        ro = self._as_float(getattr(self, "restart_overhead", 0.0))
        self._od_buffer = max(3.0 * 3600.0, 6.0 * gap) + 2.0 * ro

        self._inited = True
        self._calibrate_peek(has_spot)

    def _calibrate_peek(self, has_spot: bool) -> None:
        self._peek_calibrated = True
        if self._peek_has_spot is not None:
            return

        env = self.env
        try:
            cur = int(env.get_current_region())
        except Exception:
            cur = 0

        def mk_method(name: str) -> Optional[Callable[[int], Optional[bool]]]:
            fn = getattr(env, name, None)
            if not callable(fn):
                return None

            def _f(r: int) -> Optional[bool]:
                try:
                    v = fn(r)
                    if isinstance(v, (bool, int)):
                        return bool(v)
                except TypeError:
                    try:
                        v = fn()
                        if isinstance(v, Sequence) and 0 <= r < len(v):
                            return bool(v[r])
                    except Exception:
                        return None
                except Exception:
                    return None
                return None

            return _f

        for nm in ("get_has_spot", "has_spot", "is_spot_available", "get_spot", "spot_available"):
            f = mk_method(nm)
            if f is None:
                continue
            v = f(cur)
            if v is None:
                continue
            if bool(v) == bool(has_spot):
                self._peek_has_spot = f
                return

        gap = float(getattr(env, "gap_seconds", 1.0))
        step_idx = 0
        try:
            if gap > 0:
                step_idx = int(float(getattr(env, "elapsed_seconds", 0.0)) // gap)
        except Exception:
            step_idx = 0

        def mk_trace(attr_name: str) -> Optional[Callable[[int], Optional[bool]]]:
            arr = getattr(env, attr_name, None)
            if arr is None:
                return None
            if not isinstance(arr, (list, tuple)):
                return None
            if len(arr) < self._num_regions:
                return None

            def _f(r: int) -> Optional[bool]:
                try:
                    tr = arr[r]
                    if tr is None:
                        return None
                    if hasattr(tr, "__len__") and step_idx < len(tr):
                        v = tr[step_idx]
                        if isinstance(v, (bool, int)):
                            return bool(v)
                        try:
                            return bool(int(v))
                        except Exception:
                            return None
                except Exception:
                    return None
                return None

            return _f

        for an in ("spot_traces", "_spot_traces", "spot_trace", "spot_availability", "availability", "_traces", "traces"):
            f = mk_trace(an)
            if f is None:
                continue
            v = f(cur)
            if v is None:
                continue
            if bool(v) == bool(has_spot):
                self._peek_has_spot = f
                return

    def _choose_region(self, cur: int) -> int:
        n = self._num_regions
        if n <= 1:
            return cur

        t = max(1, self._step_count)
        import math

        best_r = cur
        best_score = -1e18

        logt = math.log(t + 1.0)
        c = 0.55

        for r in range(n):
            obs = self._obs[r]
            av = self._avail[r]
            mean = (av + 1.0) / (obs + 2.0)
            bonus = c * math.sqrt(logt / (obs + 1.0))
            score = mean + bonus
            if r == cur:
                score -= 0.02
            if score > best_score:
                best_score = score
                best_r = r

        if best_r == cur:
            second_r = cur
            second_score = -1e18
            for r in range(n):
                if r == cur:
                    continue
                obs = self._obs[r]
                av = self._avail[r]
                mean = (av + 1.0) / (obs + 2.0)
                score = mean
                if score > second_score:
                    second_score = score
                    second_r = r
            if second_r != cur:
                best_r = second_r

        return best_r

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._step_count += 1
        self._ensure_init(has_spot)

        try:
            cur = int(self.env.get_current_region())
        except Exception:
            cur = 0

        if 0 <= cur < self._num_regions:
            self._obs[cur] += 1
            if has_spot:
                self._avail[cur] += 1

        done = self._update_done_cache()

        task_duration = self._as_float(getattr(self, "task_duration", 0.0))
        deadline = self._as_float(getattr(self, "deadline", 0.0))
        restart_overhead = self._as_float(getattr(self, "restart_overhead", 0.0))

        remaining_work = task_duration - done
        if remaining_work <= 0.0:
            return ClusterType.NONE

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        time_remaining = deadline - elapsed
        if time_remaining <= 0.0:
            return ClusterType.ON_DEMAND

        remaining_restart_overhead = self._as_float(getattr(self, "remaining_restart_overhead", 0.0))

        if last_cluster_type == ClusterType.ON_DEMAND:
            overhead_for_commit = remaining_restart_overhead
        else:
            overhead_for_commit = restart_overhead

        req_if_commit_od = remaining_work + max(0.0, overhead_for_commit)
        slack_od = time_remaining - req_if_commit_od

        if self._force_ondemand:
            return ClusterType.ON_DEMAND

        if slack_od <= max(float(getattr(self.env, "gap_seconds", 1.0)), 0.0):
            self._force_ondemand = True
            return ClusterType.ON_DEMAND

        if slack_od <= self._od_buffer:
            self._force_ondemand = True
            return ClusterType.ON_DEMAND

        if has_spot:
            self._switch_cooldown = max(0, self._switch_cooldown - 1)
            return ClusterType.SPOT

        if self._num_regions > 1:
            if self._switch_cooldown > 0:
                self._switch_cooldown -= 1
            else:
                target = cur
                if self._peek_has_spot is not None:
                    try:
                        candidates = []
                        for r in range(self._num_regions):
                            if r == cur:
                                continue
                            v = self._peek_has_spot(r)
                            if v is True:
                                candidates.append(r)
                        if candidates:
                            best = candidates[0]
                            best_mean = -1.0
                            for r in candidates:
                                mean = (self._avail[r] + 1.0) / (self._obs[r] + 2.0)
                                if mean > best_mean:
                                    best_mean = mean
                                    best = r
                            target = best
                        else:
                            target = self._choose_region(cur)
                    except Exception:
                        target = self._choose_region(cur)
                else:
                    target = self._choose_region(cur)

                if target != cur:
                    try:
                        self.env.switch_region(int(target))
                    except Exception:
                        pass
                    self._switch_cooldown = self._switch_cooldown_reset

        return ClusterType.NONE