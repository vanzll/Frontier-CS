import json
import inspect
from argparse import Namespace
from typing import Any, Callable, Optional

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
        self._num_regions = 1
        self._rr_region = 0

        self._td_len = 0
        self._done_sum = 0.0

        self._spot_cnt = []
        self._spot_sum = []
        self._no_spot_streak = 0

        self._spot_query: Optional[Callable[[int], bool]] = None
        self._spot_query_ready = False

        self._safety = 0.0
        self._switch_streak = 2
        self._min_obs_exploit = 10

        return self

    def _init_once(self, has_spot: bool) -> None:
        if self._inited:
            return
        self._inited = True

        try:
            self._num_regions = int(self.env.get_num_regions())
        except Exception:
            self._num_regions = 1

        self._rr_region = 0
        self._spot_cnt = [0] * self._num_regions
        self._spot_sum = [0] * self._num_regions

        self._td_len = 0
        self._done_sum = 0.0

        gap = float(getattr(self.env, "gap_seconds", 1.0) or 1.0)
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        self._safety = max(2.0 * ro, min(gap * 0.05, 600.0))

        self._ensure_spot_query(has_spot)

    def _update_done_sum(self) -> None:
        td = self.task_done_time
        n = len(td)
        if n <= self._td_len:
            return
        self._done_sum += float(sum(td[self._td_len : n]))
        self._td_len = n

    def _try_make_query_from_method(self, name: str, has_spot: bool) -> Optional[Callable[[int], bool]]:
        env = self.env
        fn = getattr(env, name, None)
        if not callable(fn):
            return None

        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            arity = sum(
                1
                for p in params
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                and p.default is inspect._empty
            )
        except Exception:
            arity = None

        def _t_idx() -> int:
            gap = float(getattr(env, "gap_seconds", 1.0) or 1.0)
            return int(float(getattr(env, "elapsed_seconds", 0.0) or 0.0) / gap)

        cur_r = int(env.get_current_region())

        candidates = []
        if arity is None:
            candidates = [
                lambda idx: bool(fn(idx)),
                lambda idx: bool(fn(idx, _t_idx())),
                lambda idx: bool(fn(_t_idx(), idx)),
            ]
        elif arity == 1:
            candidates = [lambda idx: bool(fn(idx))]
        elif arity == 2:
            candidates = [lambda idx: bool(fn(idx, _t_idx())), lambda idx: bool(fn(_t_idx(), idx))]
        else:
            return None

        for q in candidates:
            try:
                v = q(cur_r)
                if bool(v) == bool(has_spot):
                    return q
            except Exception:
                continue
        return None

    def _try_make_query_from_traces(self, has_spot: bool) -> Optional[Callable[[int], bool]]:
        env = self.env
        num_regions = self._num_regions
        cur_r = int(env.get_current_region())
        gap = float(getattr(env, "gap_seconds", 1.0) or 1.0)
        t_idx = int(float(getattr(env, "elapsed_seconds", 0.0) or 0.0) / gap)

        def _coerce_bool(x: Any) -> bool:
            try:
                if isinstance(x, (bool, int)):
                    return bool(x)
                if hasattr(x, "item"):
                    return bool(x.item())
                return bool(x)
            except Exception:
                return False

        try:
            items = list(getattr(env, "__dict__", {}).items())
        except Exception:
            return None

        for _, v in items:
            try:
                if not isinstance(v, (list, tuple)) or len(v) != num_regions:
                    continue
                seq = v[cur_r]
                if seq is None:
                    continue
                if not hasattr(seq, "__len__") or len(seq) <= t_idx:
                    continue
                if _coerce_bool(seq[t_idx]) != bool(has_spot):
                    continue

                def q(idx: int, traces=v) -> bool:
                    env2 = self.env
                    gap2 = float(getattr(env2, "gap_seconds", 1.0) or 1.0)
                    t2 = int(float(getattr(env2, "elapsed_seconds", 0.0) or 0.0) / gap2)
                    return _coerce_bool(traces[idx][t2])

                return q
            except Exception:
                continue
        return None

    def _ensure_spot_query(self, has_spot: bool) -> None:
        if self._spot_query_ready:
            return
        self._spot_query_ready = True

        for name in (
            "is_spot_available",
            "get_spot_availability",
            "get_spot_available",
            "spot_available",
            "has_spot",
            "get_has_spot",
            "spot",
            "get_spot",
        ):
            q = self._try_make_query_from_method(name, has_spot)
            if q is not None:
                self._spot_query = q
                return

        q = self._try_make_query_from_traces(has_spot)
        if q is not None:
            self._spot_query = q
            return

        self._spot_query = None

    def _pick_best_region_by_estimate(self) -> int:
        best_r = int(self.env.get_current_region())
        best_rate = -1.0

        for r in range(self._num_regions):
            c = self._spot_cnt[r]
            if c <= 0:
                rate = 0.5
            else:
                rate = float(self._spot_sum[r]) / float(c)
            if rate > best_rate:
                best_rate = rate
                best_r = r

        return best_r

    def _pick_explore_region(self) -> int:
        self._rr_region = (self._rr_region + 1) % self._num_regions
        return self._rr_region

    def _region_with_spot_now(self) -> Optional[int]:
        if self._spot_query is None:
            return None
        cur = int(self.env.get_current_region())
        try:
            if self._spot_query(cur):
                return cur
        except Exception:
            return None
        for r in range(self._num_regions):
            if r == cur:
                continue
            try:
                if self._spot_query(r):
                    return r
            except Exception:
                continue
        return None

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._init_once(has_spot)
        self._update_done_sum()

        cur_r = int(self.env.get_current_region())
        if 0 <= cur_r < self._num_regions:
            self._spot_cnt[cur_r] += 1
            if has_spot:
                self._spot_sum[cur_r] += 1

        if has_spot:
            self._no_spot_streak = 0
        else:
            self._no_spot_streak += 1

        remaining_work = float(self.task_duration) - float(self._done_sum)
        if remaining_work <= 0.0:
            return ClusterType.NONE

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        time_left = float(self.deadline) - elapsed
        if time_left <= 0.0:
            return ClusterType.NONE

        pending_overhead = float(getattr(self, "remaining_restart_overhead", 0.0) or 0.0)
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 1.0) or 1.0)

        must_commit = time_left <= (remaining_work + pending_overhead + ro + self._safety)
        if must_commit:
            return ClusterType.ON_DEMAND

        if not has_spot:
            alt = self._region_with_spot_now()
            if alt is not None and alt != cur_r:
                try:
                    self.env.switch_region(int(alt))
                    cur_r = int(alt)
                except Exception:
                    pass
                return ClusterType.SPOT
            elif alt == cur_r:
                return ClusterType.SPOT

        if has_spot:
            slack = time_left - (remaining_work + pending_overhead + self._safety)
            if last_cluster_type == ClusterType.ON_DEMAND and slack <= (gap + ro + self._safety):
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        can_wait_one = (time_left - gap) > (remaining_work + pending_overhead + ro + self._safety)
        action = ClusterType.NONE if can_wait_one else ClusterType.ON_DEMAND

        if action == ClusterType.NONE and self._num_regions > 1:
            should_switch = self._no_spot_streak >= self._switch_streak
            if should_switch:
                best_r = self._pick_best_region_by_estimate()
                if self._spot_cnt[best_r] < self._min_obs_exploit:
                    target = self._pick_explore_region()
                else:
                    target = best_r
                if target != cur_r:
                    try:
                        self.env.switch_region(int(target))
                    except Exception:
                        pass

        return action