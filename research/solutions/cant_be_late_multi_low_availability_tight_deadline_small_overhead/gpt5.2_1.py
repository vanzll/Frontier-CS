import json
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "deadline_aware_mr_spot"

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
        return self

    def _lazy_init(self) -> None:
        if getattr(self, "_inited", False):
            return
        self._inited = True

        self._done_total = 0.0
        self._done_len = 0

        self._committed_on_demand = False
        self._safety_buffer_seconds = None  # type: Optional[float]

        self._probe_mode = 0  # 0: none, 1: direct trace accessor, 2: switch+read
        self._spot_accessor = None  # type: Optional[Callable[[int, int], Optional[bool]]]
        self._read_current_spot = None  # type: Optional[Callable[[], Optional[bool]]]

        self._ema = None  # type: Optional[List[float]]
        self._ema_alpha = 0.06

    def _update_done_cache(self) -> None:
        tdt = self.task_done_time
        n = len(tdt)
        if n > self._done_len:
            self._done_total += sum(tdt[self._done_len : n])
            self._done_len = n

    def _step_index(self) -> int:
        gap = float(getattr(self.env, "gap_seconds", 1.0))
        if gap <= 0:
            return 0
        return int(float(getattr(self.env, "elapsed_seconds", 0.0)) // gap)

    def _build_read_current_spot(self) -> Optional[Callable[[], Optional[bool]]]:
        env = self.env

        attr_names = [
            "has_spot",
            "spot_available",
            "spot_availability",
            "current_has_spot",
            "_has_spot",
            "_spot_available",
            "spot",
        ]
        method_names = [
            "get_has_spot",
            "get_current_has_spot",
            "has_current_spot",
            "is_spot_available",
            "spot_is_available",
            "get_spot_available",
            "get_spot",
        ]

        def reader() -> Optional[bool]:
            for nm in attr_names:
                try:
                    v = getattr(env, nm)
                except Exception:
                    continue
                try:
                    if isinstance(v, bool):
                        return v
                    if callable(v):
                        r = v()
                        if isinstance(r, bool):
                            return r
                except Exception:
                    pass
            for nm in method_names:
                m = getattr(env, nm, None)
                if callable(m):
                    try:
                        r = m()
                        if isinstance(r, bool):
                            return r
                    except Exception:
                        continue
            return None

        if reader() is None:
            return None
        return reader

    def _build_spot_accessor(self) -> Optional[Callable[[int, int], Optional[bool]]]:
        env = self.env
        num_regions = int(env.get_num_regions())

        method_candidates = [
            "get_spot_availability",
            "get_has_spot_in_region",
            "has_spot_in_region",
            "is_spot_available_in_region",
            "spot_available_in_region",
            "get_spot_in_region",
            "peek_spot",
        ]
        for nm in method_candidates:
            m = getattr(env, nm, None)
            if not callable(m):
                continue

            def make_method_accessor(func: Callable[..., Any]) -> Callable[[int, int], Optional[bool]]:
                def acc(ridx: int, tidx: int) -> Optional[bool]:
                    try:
                        v = func(ridx, tidx)
                        if isinstance(v, bool):
                            return v
                        if isinstance(v, (int, float)):
                            return bool(v)
                    except Exception:
                        pass
                    try:
                        v = func(ridx)
                        if isinstance(v, bool):
                            return v
                        if isinstance(v, (int, float)):
                            return bool(v)
                    except Exception:
                        pass
                    return None

                return acc

            acc = make_method_accessor(m)
            try:
                test = acc(0, self._step_index())
                if isinstance(test, bool):
                    return acc
            except Exception:
                pass

        attr_candidates = [
            "spot_traces",
            "_spot_traces",
            "spot_trace",
            "_spot_trace",
            "traces",
            "_traces",
            "availability_traces",
            "_availability_traces",
            "spot_availabilities",
            "_spot_availabilities",
            "region_traces",
            "_region_traces",
        ]

        def seq_get(seq: Any, idx: int) -> Optional[bool]:
            try:
                v = seq[idx]
            except Exception:
                try:
                    v = seq.get(idx)
                except Exception:
                    return None
            try:
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int, float)):
                    return bool(v)
                if hasattr(v, "item"):
                    vv = v.item()
                    if isinstance(vv, bool):
                        return vv
                    if isinstance(vv, (int, float)):
                        return bool(vv)
            except Exception:
                pass
            return None

        for nm in attr_candidates:
            traces = getattr(env, nm, None)
            if traces is None:
                continue

            if isinstance(traces, dict):
                if len(traces) < num_regions:
                    continue

                def make_dict_accessor(d: Dict[Any, Any]) -> Callable[[int, int], Optional[bool]]:
                    def acc(ridx: int, tidx: int) -> Optional[bool]:
                        tr = d.get(ridx)
                        if tr is None:
                            tr = d.get(str(ridx))
                        if tr is None:
                            return None
                        return seq_get(tr, tidx)

                    return acc

                acc = make_dict_accessor(traces)
                test = acc(0, self._step_index())
                if isinstance(test, bool):
                    return acc

            if isinstance(traces, (list, tuple)):
                if len(traces) < num_regions:
                    continue
                if not traces:
                    continue
                test = None
                try:
                    test = seq_get(traces[0], self._step_index())
                except Exception:
                    test = None
                if not isinstance(test, bool):
                    continue

                def make_list_accessor(lst: Any) -> Callable[[int, int], Optional[bool]]:
                    def acc(ridx: int, tidx: int) -> Optional[bool]:
                        if ridx < 0 or ridx >= len(lst):
                            return None
                        return seq_get(lst[ridx], tidx)

                    return acc

                return make_list_accessor(traces)

        return None

    def _ensure_env_helpers(self, has_spot_param: bool) -> None:
        if self._safety_buffer_seconds is None:
            gap = float(getattr(self.env, "gap_seconds", 3600.0))
            ro = float(self.restart_overhead)
            self._safety_buffer_seconds = max(2.0 * 3600.0, 2.0 * gap, 8.0 * ro)

        if self._ema is None:
            n = int(self.env.get_num_regions())
            self._ema = [0.5] * n

        if self._probe_mode != 0:
            return

        self._spot_accessor = self._build_spot_accessor()
        if self._spot_accessor is not None:
            self._probe_mode = 1
            return

        self._read_current_spot = self._build_read_current_spot()
        if self._read_current_spot is not None and int(self.env.get_num_regions()) > 1:
            self._probe_mode = 2
            return

        self._probe_mode = 0

    def _spot_now_all_regions(self, has_spot_param: bool) -> Optional[List[bool]]:
        env = self.env
        n = int(env.get_num_regions())
        cur = int(env.get_current_region())
        t = self._step_index()

        if self._probe_mode == 1 and self._spot_accessor is not None:
            out = [False] * n
            for i in range(n):
                v = self._spot_accessor(i, t)
                if v is None:
                    if i == cur:
                        out[i] = bool(has_spot_param)
                    else:
                        out[i] = False
                else:
                    out[i] = bool(v)
            return out

        if self._probe_mode == 2 and self._read_current_spot is not None:
            orig = cur
            out = [False] * n
            for i in range(n):
                try:
                    env.switch_region(i)
                except Exception:
                    continue
                v = self._read_current_spot()
                if v is None:
                    out[i] = bool(has_spot_param) if i == orig else False
                else:
                    out[i] = bool(v)
            try:
                env.switch_region(orig)
            except Exception:
                pass
            return out

        return None

    def _choose_spot_region(self, has_spot_param: bool) -> Optional[int]:
        env = self.env
        n = int(env.get_num_regions())
        cur = int(env.get_current_region())
        if n <= 1:
            return cur if has_spot_param else None

        spots = self._spot_now_all_regions(has_spot_param)
        if spots is None:
            return cur if has_spot_param else None

        if spots[cur]:
            return cur

        best = None
        best_score = -1e18
        ema = self._ema if self._ema is not None else [0.0] * n
        for i in range(n):
            if not spots[i]:
                continue
            score = ema[i]
            if score > best_score:
                best_score = score
                best = i
        return best

    def _update_ema(self, has_spot_param: bool) -> None:
        if self._ema is None:
            return
        spots = self._spot_now_all_regions(has_spot_param)
        if spots is None:
            cur = int(self.env.get_current_region())
            a = self._ema_alpha
            self._ema[cur] = (1.0 - a) * self._ema[cur] + a * (1.0 if has_spot_param else 0.0)
            return
        a = self._ema_alpha
        for i, v in enumerate(spots):
            self._ema[i] = (1.0 - a) * self._ema[i] + a * (1.0 if v else 0.0)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._lazy_init()
        self._update_done_cache()
        self._ensure_env_helpers(has_spot)
        self._update_ema(has_spot)

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        time_left = float(self.deadline) - elapsed
        if time_left <= 0:
            return ClusterType.NONE

        work_left = float(self.task_duration) - float(self._done_total)
        if work_left <= 0:
            return ClusterType.NONE

        remaining_oh = float(getattr(self, "remaining_restart_overhead", 0.0) or 0.0)
        safety = float(self._safety_buffer_seconds or 0.0)

        if (not self._committed_on_demand) and (time_left <= work_left + remaining_oh + float(self.restart_overhead) + safety):
            self._committed_on_demand = True

        if self._committed_on_demand:
            return ClusterType.ON_DEMAND

        spot_region = self._choose_spot_region(has_spot)
        if spot_region is not None:
            cur = int(self.env.get_current_region())
            if spot_region != cur:
                try:
                    self.env.switch_region(spot_region)
                except Exception:
                    pass
            return ClusterType.SPOT

        return ClusterType.NONE