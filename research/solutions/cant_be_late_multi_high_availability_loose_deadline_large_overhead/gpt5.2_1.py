import json
import inspect
from argparse import Namespace
from typing import Any, Callable, Optional, Tuple

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


_CT_SPOT = getattr(ClusterType, "SPOT")
_CT_OD = getattr(ClusterType, "ON_DEMAND")
_CT_NONE = getattr(ClusterType, "NONE", getattr(ClusterType, "None"))


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

        # Optional (not required by the base class)
        self._trace_files = config.get("trace_files", None)

        # Runtime-initialized
        self._rt_inited = False
        self._done_work = 0.0
        self._td_len = 0
        self._num_regions = 1
        self._gap = 0.0
        self._rr_start = 0

        # Spot query support:
        # mode: "arg" -> query(region) bool
        #       "current" -> query() bool for current region (use switch_region to probe)
        #       "none" -> no query available
        self._spot_query_mode = "none"
        self._spot_query: Optional[Callable[..., bool]] = None

        return self

    def _get_task_done_list(self):
        td = self.task_done_time
        if isinstance(td, (list, tuple)) and td and isinstance(td[0], (list, tuple)):
            return td[0]
        return td

    def _get_task_duration(self) -> float:
        td = self.task_duration
        if isinstance(td, (list, tuple)):
            return float(td[0])
        return float(td)

    def _get_deadline(self) -> float:
        d = self.deadline
        if isinstance(d, (list, tuple)):
            return float(d[0])
        return float(d)

    def _get_restart_overhead(self) -> float:
        ro = self.restart_overhead
        if isinstance(ro, (list, tuple)):
            return float(ro[0])
        return float(ro)

    def _init_runtime(self):
        self._gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        if self._gap <= 0:
            self._gap = 1.0
        self._num_regions = int(self.env.get_num_regions())
        self._rr_start = int(self.env.get_current_region())

        self._detect_spot_query()
        self._rt_inited = True

    def _try_make_query_callable(self, obj: Any) -> Optional[Tuple[str, Callable[..., bool]]]:
        if not callable(obj):
            return None
        try:
            sig = inspect.signature(obj)
            params = sig.parameters
            n_required = 0
            for p in params.values():
                if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                if p.default is inspect._empty:
                    n_required += 1
            if n_required == 0:
                def q0() -> bool:
                    return bool(obj())
                _ = q0()
                return "current", q0
            if n_required == 1:
                def q1(region: int) -> bool:
                    return bool(obj(region))
                _ = q1(self.env.get_current_region())
                return "arg", q1
        except Exception:
            pass

        # Fallback tries
        try:
            _ = obj()
            return "current", (lambda: bool(obj()))
        except TypeError:
            try:
                _ = obj(self.env.get_current_region())
                return "arg", (lambda region: bool(obj(region)))
            except Exception:
                return None
        except Exception:
            return None

    def _detect_spot_query(self):
        env = self.env

        # Common candidates
        candidates = [
            "has_spot",
            "get_has_spot",
            "spot_available",
            "is_spot_available",
            "get_spot_available",
            "get_spot_availability",
            "get_spot",
        ]
        for name in candidates:
            if hasattr(env, name):
                obj = getattr(env, name)
                if callable(obj):
                    res = self._try_make_query_callable(obj)
                    if res is not None:
                        self._spot_query_mode, self._spot_query = res
                        return
                else:
                    # Boolean attribute for current region
                    if isinstance(obj, bool):
                        self._spot_query_mode = "current"
                        self._spot_query = lambda: bool(getattr(env, name))
                        return

        # If passed has_spot is redundant, there may be env attributes containing spot per region
        # Attempt to find a per-region boolean array for current timestep.
        for name in dir(env):
            if "spot" not in name.lower():
                continue
            try:
                val = getattr(env, name)
            except Exception:
                continue
            if isinstance(val, (list, tuple)) and len(val) == self._num_regions:
                # If it looks like per-region availability booleans at current time
                if all(isinstance(x, bool) for x in val):
                    self._spot_query_mode = "arg"
                    self._spot_query = lambda region, _val=val: bool(_val[region])
                    return

        self._spot_query_mode = "none"
        self._spot_query = None

    def _update_done_work(self):
        td = self._get_task_done_list()
        if not isinstance(td, (list, tuple)):
            return
        n = len(td)
        if n > self._td_len:
            self._done_work += sum(float(x) for x in td[self._td_len:n])
            self._td_len = n

    def _spot_available_in_region(self, region: int, current_has_spot: Optional[bool]) -> Optional[bool]:
        if self._spot_query_mode == "arg" and self._spot_query is not None:
            try:
                return bool(self._spot_query(region))
            except Exception:
                return None
        if self._spot_query_mode == "current" and self._spot_query is not None:
            try:
                cur = int(self.env.get_current_region())
                if region != cur:
                    self.env.switch_region(region)
                return bool(self._spot_query())
            except Exception:
                return None
        # No query support: only know current region's has_spot from argument
        cur = int(self.env.get_current_region())
        if current_has_spot is not None and region == cur:
            return bool(current_has_spot)
        return None

    def _find_spot_region_now(self, current_has_spot: bool) -> Optional[int]:
        cur = int(self.env.get_current_region())

        if current_has_spot:
            return cur

        if self._spot_query_mode == "none":
            return None

        num = self._num_regions
        start = (self._rr_start + 1) % num if num > 0 else 0
        chosen = None

        orig_region = cur
        try:
            for k in range(num):
                r = (start + k) % num
                avail = self._spot_available_in_region(r, current_has_spot if r == cur else None)
                if avail:
                    chosen = r
                    break
        finally:
            if self._spot_query_mode == "current":
                # Restore to original region to avoid leaving in a probed state.
                try:
                    self.env.switch_region(orig_region)
                except Exception:
                    pass

        if chosen is not None:
            self._rr_start = chosen
        return chosen

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._rt_inited:
            self._init_runtime()

        self._update_done_work()

        task_duration = self._get_task_duration()
        deadline = self._get_deadline()
        restart_overhead = self._get_restart_overhead()

        remaining_work = task_duration - self._done_work
        if remaining_work <= 1e-9:
            return _CT_NONE

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        remaining_time = deadline - elapsed

        gap = self._gap
        slack = remaining_time - remaining_work

        # Conservative urgency: if slack small, switch to on-demand to guarantee completion.
        urgent_slack = 2.0 * gap + 2.0 * restart_overhead
        must_finish_slack = restart_overhead + 0.5 * gap

        if slack <= must_finish_slack or remaining_time <= 0:
            return _CT_OD
        urgent = slack <= urgent_slack

        # If we already committed to on-demand and it's getting close, avoid switching back.
        if last_cluster_type == _CT_OD:
            if urgent:
                return _CT_OD
            # Switch back to spot only if we have ample slack.
            if slack < (6.0 * restart_overhead + 3.0 * gap):
                return _CT_OD
            # else, allow switching to spot if spot is available (here or elsewhere)

        # Prefer spot if possible (current or another region), unless urgent.
        spot_region = None
        if not urgent:
            spot_region = self._find_spot_region_now(bool(has_spot))

        if not urgent and spot_region is not None:
            cur = int(self.env.get_current_region())
            if spot_region != cur:
                self.env.switch_region(spot_region)
                # If we can't query spot after switching, do not risk returning SPOT.
                if self._spot_query_mode == "none":
                    return _CT_NONE
                # Validate spot availability if possible
                if self._spot_query_mode != "none":
                    avail = self._spot_available_in_region(spot_region, None)
                    if avail is not True:
                        return _CT_NONE
            return _CT_SPOT

        # No spot found now (or urgent): choose between pausing and on-demand.
        # Pause when there's ample slack left, otherwise on-demand.
        pause_slack = 6.0 * gap + 3.0 * restart_overhead
        if not urgent and slack >= pause_slack:
            # If we can (cheaply) move to another region to try our luck next step even without query, do it.
            if self._spot_query_mode == "none" and self._num_regions > 1:
                nxt = (int(self.env.get_current_region()) + 1) % self._num_regions
                self.env.switch_region(nxt)
            return _CT_NONE

        return _CT_OD