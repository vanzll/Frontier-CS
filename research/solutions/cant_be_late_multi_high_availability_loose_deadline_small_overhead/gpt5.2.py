import json
import math
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Tuple

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_multi_region_v1"

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

        self._trace_files = list(config.get("trace_files", []))
        self._use_traces = False
        self._trace_mismatch = 0
        self._trace_mismatch_disable_threshold = 6

        self._avail: List[List[bool]] = []
        self._any_spot: List[bool] = []
        self._best_run_any: List[int] = []
        self._best_region_at: List[int] = []
        self._next_any_spot: List[int] = []
        self._next_good_spot: List[int] = []

        self._done_sum = 0.0
        self._done_len = 0

        self._lock_on_demand = False

        try:
            self._init_traces()
        except Exception:
            self._use_traces = False

        return self

    def _task_duration_seconds(self) -> float:
        td = getattr(self, "task_duration", 0.0)
        if isinstance(td, (list, tuple)):
            return float(td[0]) if td else 0.0
        return float(td)

    def _deadline_seconds(self) -> float:
        dl = getattr(self, "deadline", 0.0)
        if isinstance(dl, (list, tuple)):
            return float(dl[0]) if dl else 0.0
        return float(dl)

    def _restart_overhead_seconds(self) -> float:
        ro = getattr(self, "restart_overhead", 0.0)
        if isinstance(ro, (list, tuple)):
            return float(ro[0]) if ro else 0.0
        return float(ro)

    def _get_done_sum(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            self._done_sum = 0.0
            self._done_len = 0
            return 0.0
        n = len(tdt)
        if n == self._done_len:
            return self._done_sum
        if n < self._done_len:
            self._done_sum = float(sum(tdt))
            self._done_len = n
            return self._done_sum
        self._done_sum += float(sum(tdt[self._done_len : n]))
        self._done_len = n
        return self._done_sum

    @staticmethod
    def _coerce_bool(x: Any) -> Optional[bool]:
        if isinstance(x, bool):
            return x
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return bool(x >= 0.5)
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ("true", "t", "yes", "y", "1", "available", "avail", "up"):
                return True
            if s in ("false", "f", "no", "n", "0", "unavailable", "down"):
                return False
            try:
                v = float(s)
                return bool(v >= 0.5)
            except Exception:
                return None
        return None

    def _load_trace_file(self, path: str) -> List[bool]:
        if not path or not os.path.exists(path):
            return []
        with open(path, "r") as f:
            content = f.read()
        content_stripped = content.strip()
        if not content_stripped:
            return []
        data = None
        try:
            data = json.loads(content_stripped)
        except Exception:
            data = None

        out: List[bool] = []

        def emit(v: Any) -> None:
            b = self._coerce_bool(v)
            if b is not None:
                out.append(b)

        if data is not None:
            if isinstance(data, list):
                for el in data:
                    if isinstance(el, dict):
                        for k in ("available", "spot", "has_spot", "value", "avail"):
                            if k in el:
                                emit(el[k])
                                break
                        else:
                            if el:
                                emit(next(iter(el.values())))
                    else:
                        emit(el)
                return out
            if isinstance(data, dict):
                best_list = None
                for v in data.values():
                    if isinstance(v, list) and (best_list is None or len(v) > len(best_list)):
                        best_list = v
                if best_list is not None:
                    for el in best_list:
                        if isinstance(el, dict):
                            for k in ("available", "spot", "has_spot", "value", "avail"):
                                if k in el:
                                    emit(el[k])
                                    break
                            else:
                                if el:
                                    emit(next(iter(el.values())))
                        else:
                            emit(el)
                    return out
                for k in ("available", "spot", "has_spot", "value", "avail"):
                    if k in data:
                        v = data[k]
                        if isinstance(v, list):
                            for el in v:
                                emit(el)
                        else:
                            emit(v)
                        return out

        for line in content.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            if "#" in s:
                s = s.split("#", 1)[0].strip()
                if not s:
                    continue
            toks = []
            if "," in s:
                toks = [t.strip() for t in s.split(",") if t.strip()]
            else:
                toks = [t for t in s.split() if t]
            if not toks:
                continue
            parsed = None
            for tok in reversed(toks):
                b = self._coerce_bool(tok)
                if b is not None:
                    parsed = b
                    break
            if parsed is None:
                continue
            out.append(parsed)
        return out

    def _init_traces(self) -> None:
        num_regions = int(self.env.get_num_regions())
        if not self._trace_files or len(self._trace_files) < num_regions:
            self._use_traces = False
            return

        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        if gap <= 0:
            self._use_traces = False
            return

        deadline = self._deadline_seconds()
        steps_needed = int(math.ceil(deadline / gap)) + 8

        avail: List[List[bool]] = []
        for r in range(num_regions):
            tr = self._load_trace_file(self._trace_files[r])
            if len(tr) < steps_needed:
                fill = tr[-1] if tr else False
                tr = tr + [fill] * (steps_needed - len(tr))
            elif len(tr) > steps_needed:
                tr = tr[:steps_needed]
            avail.append(tr)

        # run_len[r][t]: consecutive True starting at t
        run_len: List[List[int]] = [[0] * steps_needed for _ in range(num_regions)]
        for r in range(num_regions):
            rl = run_len[r]
            ar = avail[r]
            for t in range(steps_needed - 1, -1, -1):
                if ar[t]:
                    rl[t] = 1 + (rl[t + 1] if t + 1 < steps_needed else 0)
                else:
                    rl[t] = 0

        any_spot = [False] * steps_needed
        best_run_any = [0] * steps_needed
        best_region_at = [-1] * steps_needed

        for t in range(steps_needed):
            best_r = -1
            best_len = 0
            any_now = False
            for r in range(num_regions):
                if avail[r][t]:
                    any_now = True
                    l = run_len[r][t]
                    if l > best_len:
                        best_len = l
                        best_r = r
            any_spot[t] = any_now
            best_run_any[t] = best_len
            best_region_at[t] = best_r

        next_any_spot = [steps_needed] * steps_needed
        nxt = steps_needed
        for t in range(steps_needed - 1, -1, -1):
            if any_spot[t]:
                nxt = t
            next_any_spot[t] = nxt

        over = self._restart_overhead_seconds()
        min_len_for_progress = int(math.floor(over / gap)) + 1  # need total_time > over
        if min_len_for_progress < 1:
            min_len_for_progress = 1

        next_good_spot = [steps_needed] * steps_needed
        nxtg = steps_needed
        for t in range(steps_needed - 1, -1, -1):
            if best_run_any[t] >= min_len_for_progress:
                nxtg = t
            next_good_spot[t] = nxtg

        self._avail = avail
        self._any_spot = any_spot
        self._best_run_any = best_run_any
        self._best_region_at = best_region_at
        self._next_any_spot = next_any_spot
        self._next_good_spot = next_good_spot
        self._min_len_for_progress = min_len_for_progress
        self._steps_needed = steps_needed
        self._use_traces = True
        self._trace_mismatch = 0

    def _t_index(self) -> int:
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        if gap <= 0:
            return 0
        t = int(self.env.elapsed_seconds // gap)
        if t < 0:
            return 0
        return t

    def _spot_pred(self, region: int, t: int) -> Optional[bool]:
        if not self._use_traces:
            return None
        if region < 0 or region >= len(self._avail):
            return None
        if t < 0:
            return None
        if t >= len(self._avail[region]):
            return self._avail[region][-1] if self._avail[region] else None
        return self._avail[region][t]

    def _maybe_disable_traces_on_mismatch(self, cur_region: int, t: int, has_spot: bool) -> None:
        if not self._use_traces:
            return
        pred = self._spot_pred(cur_region, t)
        if pred is None:
            return
        if bool(pred) != bool(has_spot):
            self._trace_mismatch += 1
            if self._trace_mismatch >= self._trace_mismatch_disable_threshold and t < 200:
                self._use_traces = False

    def _urgent_should_lock_on_demand(
        self,
        last_cluster_type: ClusterType,
        remaining_work: float,
        time_left: float,
    ) -> bool:
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        over = self._restart_overhead_seconds()
        rr = float(getattr(self, "remaining_restart_overhead", 0.0) or 0.0)

        if remaining_work <= 0:
            return False

        # Conservative: assume we may need one restart overhead unless already on-demand and no pending overhead.
        extra = 0.0
        if not (last_cluster_type == ClusterType.ON_DEMAND and rr <= 0.0):
            extra = over

        need = remaining_work + extra
        if need >= time_left - 1e-9:
            return True

        # Additional safety margin to absorb any small mispredictions / overhead resets.
        slack = time_left - remaining_work
        if slack <= over + 3.0 * gap:
            return True

        return False

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        t = self._t_index()
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        over = self._restart_overhead_seconds()
        deadline = self._deadline_seconds()
        task_duration = self._task_duration_seconds()

        done = self._get_done_sum()
        remaining_work = task_duration - done
        if remaining_work <= 0:
            return ClusterType.NONE

        time_left = deadline - float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        if time_left <= 0:
            self._lock_on_demand = True
            return ClusterType.ON_DEMAND

        cur_region = int(self.env.get_current_region())

        self._maybe_disable_traces_on_mismatch(cur_region, t, has_spot)

        rr = float(getattr(self, "remaining_restart_overhead", 0.0) or 0.0)
        if rr > 0.0:
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            if last_cluster_type == ClusterType.SPOT:
                if has_spot:
                    return ClusterType.SPOT
                slack = time_left - remaining_work
                if slack > 4.0 * over + 6.0 * gap:
                    return ClusterType.NONE
                return ClusterType.ON_DEMAND

        if self._lock_on_demand or self._urgent_should_lock_on_demand(last_cluster_type, remaining_work, time_left):
            self._lock_on_demand = True
            return ClusterType.ON_DEMAND

        # If currently on-demand (non-urgent), opportunistically switch back to spot if stable spot window exists.
        if last_cluster_type == ClusterType.ON_DEMAND and not self._lock_on_demand:
            if self._use_traces and t < len(self._any_spot) and self._any_spot[t]:
                slack = time_left - remaining_work
                best_len = self._best_run_any[t] if t < len(self._best_run_any) else 0
                if slack >= over + 4.0 * gap and best_len >= getattr(self, "_min_len_for_progress", 1):
                    best_r = self._best_region_at[t]
                    if best_r is not None and best_r >= 0 and best_r != cur_region:
                        self.env.switch_region(int(best_r))
                    return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # Prefer SPOT if available now (in current region) to avoid restart overhead from switching.
        if has_spot:
            return ClusterType.SPOT

        # No spot in current region. Try multi-region switching if traces available.
        if self._use_traces and t < len(self._any_spot) and self._any_spot[t]:
            best_r = self._best_region_at[t]
            best_len = self._best_run_any[t] if t < len(self._best_run_any) else 0

            # Only start spot if the predicted availability streak is long enough to make progress after overhead.
            if best_r is not None and best_r >= 0 and best_len >= getattr(self, "_min_len_for_progress", 1):
                if int(best_r) != cur_region:
                    self.env.switch_region(int(best_r))
                return ClusterType.SPOT

            # Spot exists somewhere but likely too short to overcome overhead; decide wait vs on-demand.
            slack = time_left - remaining_work
            t_next = self._next_good_spot[t] if t < len(self._next_good_spot) else len(self._any_spot)
            if t_next < len(self._any_spot):
                dt = float((t_next - t) * gap)
                if slack > dt + over + 6.0 * gap:
                    return ClusterType.NONE
            return ClusterType.ON_DEMAND

        # No spot (or no reliable traces). Decide NONE vs ON_DEMAND.
        slack = time_left - remaining_work

        if self._use_traces and t < len(self._next_good_spot):
            t_next = self._next_good_spot[t]
            if t_next < len(self._any_spot):
                dt = float((t_next - t) * gap)
                if slack > dt + over + 6.0 * gap:
                    return ClusterType.NONE
            else:
                # No good spot expected until deadline; if slack huge, still idle, else on-demand.
                if slack > 10.0 * over + 12.0 * gap:
                    return ClusterType.NONE
            return ClusterType.ON_DEMAND

        if slack > 4.0 * over + 10.0 * gap:
            return ClusterType.NONE
        return ClusterType.ON_DEMAND