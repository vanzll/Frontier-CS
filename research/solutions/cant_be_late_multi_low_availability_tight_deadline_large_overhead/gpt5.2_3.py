import json
import math
import os
from argparse import Namespace
from array import array
from typing import List, Optional, Sequence, Union

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _load_trace_file(path: str, max_entries: Optional[int] = None) -> Optional[List[int]]:
    if not path:
        return None
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".npy", ".npz"):
            try:
                import numpy as np  # type: ignore
            except Exception:
                return None
            if ext == ".npy":
                arr = np.load(path, allow_pickle=False)
            else:
                npz = np.load(path, allow_pickle=False)
                if len(npz.files) == 0:
                    return None
                arr = npz[npz.files[0]]
            arr = arr.astype(bool).ravel()
            if max_entries is not None and arr.size > max_entries:
                arr = arr[:max_entries]
            return [1 if v else 0 for v in arr.tolist()]

        if ext == ".json":
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k in ("trace", "availability", "spot", "spots", "values", "data"):
                    if k in data and isinstance(data[k], (list, tuple)):
                        data = data[k]
                        break
                else:
                    for v in data.values():
                        if isinstance(v, (list, tuple)):
                            data = v
                            break
            if isinstance(data, (list, tuple)):
                out = []
                for v in data:
                    if isinstance(v, bool):
                        out.append(1 if v else 0)
                    elif isinstance(v, (int, float)):
                        out.append(1 if float(v) > 0 else 0)
                    elif isinstance(v, str):
                        vv = v.strip().lower()
                        if vv in ("1", "true", "t", "yes", "y"):
                            out.append(1)
                        elif vv in ("0", "false", "f", "no", "n"):
                            out.append(0)
                        else:
                            fv = _safe_float(vv, None)
                            out.append(1 if (fv is not None and fv > 0) else 0)
                    else:
                        out.append(0)
                    if max_entries is not None and len(out) >= max_entries:
                        break
                return out

        out = []
        with open(path, "r") as f:
            for line in f:
                if max_entries is not None and len(out) >= max_entries:
                    break
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                s2 = s.replace(",", " ").replace("\t", " ")
                parts = [p for p in s2.split(" ") if p]
                if not parts:
                    continue
                token = parts[-1]
                tl = token.strip().lower()
                if tl in ("1", "true", "t", "yes", "y"):
                    out.append(1)
                elif tl in ("0", "false", "f", "no", "n"):
                    out.append(0)
                else:
                    fv = _safe_float(tl, None)
                    out.append(1 if (fv is not None and fv > 0) else 0)
        return out if out else None
    except Exception:
        return None


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_multi_region_v1"

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

        self._trace_paths: List[str] = list(config.get("trace_files", []) or [])
        max_entries = None
        try:
            dh = float(config.get("deadline", 0.0))
            if dh > 0:
                max_entries = int(dh * 3600) + 10
                if max_entries < 1000:
                    max_entries = 1000
        except Exception:
            max_entries = None

        self._raw_traces: List[Optional[List[int]]] = []
        for p in self._trace_paths:
            self._raw_traces.append(_load_trace_file(p, max_entries=max_entries))

        self._preprocessed = False
        self._T = 0
        self._trace_bits: List[bytearray] = []
        self._next_true: List[array] = []

        self._td_len = 0
        self._work_done = 0.0

        self._committed_on_demand = False

        self._overhead_decays_on_none: Optional[bool] = None
        self._last_action: Optional[ClusterType] = None
        self._last_seen_overhead: Optional[float] = None

        return self

    def _scalar(self, x: Union[float, Sequence[float]]) -> float:
        if isinstance(x, (list, tuple)):
            return float(x[0]) if x else 0.0
        return float(x)

    def _ensure_preprocessed(self) -> None:
        if self._preprocessed:
            return

        try:
            gap = float(self.env.gap_seconds)
            if gap <= 0:
                gap = 1.0
        except Exception:
            gap = 1.0

        deadline = float(self._scalar(self.deadline))
        T = int(math.ceil(deadline / gap)) + 3
        if T < 8:
            T = 8

        num_regions = 0
        try:
            num_regions = int(self.env.get_num_regions())
        except Exception:
            num_regions = len(self._raw_traces)

        if num_regions <= 0:
            num_regions = max(1, len(self._raw_traces))

        traces: List[bytearray] = []
        for r in range(num_regions):
            tr = self._raw_traces[r] if r < len(self._raw_traces) else None
            if tr is None:
                traces.append(bytearray(T))
                continue
            if len(tr) >= T:
                traces.append(bytearray(tr[:T]))
            else:
                ba = bytearray(tr)
                if len(ba) < T:
                    ba.extend(b"\x00" * (T - len(ba)))
                traces.append(ba)

        next_true: List[array] = []
        for r in range(num_regions):
            tb = traces[r]
            nt = array("I", [0]) * (T + 1)
            nxt = T
            nt[T] = T
            for i in range(T - 1, -1, -1):
                if tb[i]:
                    nxt = i
                nt[i] = nxt
            next_true.append(nt)

        self._T = T
        self._trace_bits = traces
        self._next_true = next_true
        self._preprocessed = True

    def _update_work_done(self) -> None:
        td = self.task_done_time
        n = len(td)
        if n > self._td_len:
            self._work_done += float(sum(td[self._td_len : n]))
            self._td_len = n

    def _should_commit_on_demand(self, time_left: float, remaining_work: float) -> bool:
        gap = float(self.env.gap_seconds)
        ro = float(self._scalar(self.restart_overhead))
        try:
            pending = float(self.remaining_restart_overhead)
        except Exception:
            pending = 0.0

        needed = remaining_work + pending + ro
        safety = gap
        return time_left <= needed + safety

    def _pick_best_region_for_next_spot(self, t_next: int) -> Optional[int]:
        if not self._preprocessed or not self._next_true:
            return None
        if t_next < 0:
            t_next = 0
        if t_next >= self._T:
            t_next = self._T - 1
        best_r = None
        best_t = None
        for r, nt in enumerate(self._next_true):
            v = int(nt[t_next])
            if best_t is None or v < best_t:
                best_t = v
                best_r = r
        return best_r

    def _maybe_learn_overhead_decay(self) -> None:
        if self._last_action is None or self._last_seen_overhead is None:
            return
        if self._overhead_decays_on_none is not None:
            return
        if self._last_action != ClusterType.NONE:
            return
        try:
            cur = float(self.remaining_restart_overhead)
        except Exception:
            cur = 0.0
        prev = float(self._last_seen_overhead)
        if cur + 1e-9 < prev:
            self._overhead_decays_on_none = True
        else:
            self._overhead_decays_on_none = False

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._ensure_preprocessed()
        self._maybe_learn_overhead_decay()
        self._update_work_done()

        gap = float(self.env.gap_seconds)
        task_duration = float(self._scalar(self.task_duration))
        deadline = float(self._scalar(self.deadline))
        ro = float(self._scalar(self.restart_overhead))

        elapsed = float(self.env.elapsed_seconds)
        time_left = max(0.0, deadline - elapsed)
        remaining_work = max(0.0, task_duration - float(self._work_done))

        try:
            pending_overhead = float(self.remaining_restart_overhead)
        except Exception:
            pending_overhead = 0.0

        if remaining_work <= 1e-9:
            action = ClusterType.NONE
            self._last_action = action
            self._last_seen_overhead = pending_overhead
            return action

        if not self._committed_on_demand and self._should_commit_on_demand(time_left, remaining_work):
            self._committed_on_demand = True

        if pending_overhead > 1e-9 and self._overhead_decays_on_none is not False:
            if time_left > remaining_work + pending_overhead + gap:
                action = ClusterType.NONE
                self._last_action = action
                self._last_seen_overhead = pending_overhead
                return action

        if self._committed_on_demand:
            action = ClusterType.ON_DEMAND
            self._last_action = action
            self._last_seen_overhead = pending_overhead
            return action

        if has_spot:
            action = ClusterType.SPOT
            self._last_action = action
            self._last_seen_overhead = pending_overhead
            return action

        t = int(elapsed / gap) if gap > 0 else int(elapsed)
        t_next = t + 1

        best_r = self._pick_best_region_for_next_spot(t_next)
        if best_r is not None:
            try:
                cur_r = int(self.env.get_current_region())
            except Exception:
                cur_r = 0
            if best_r != cur_r:
                try:
                    self.env.switch_region(int(best_r))
                except Exception:
                    pass

        slack = time_left - remaining_work
        if slack > (2.0 * ro + gap):
            action = ClusterType.NONE
        else:
            self._committed_on_demand = True
            action = ClusterType.ON_DEMAND

        self._last_action = action
        self._last_seen_overhead = pending_overhead
        return action