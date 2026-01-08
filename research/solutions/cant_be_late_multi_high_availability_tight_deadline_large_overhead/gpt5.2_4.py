import json
import math
import os
import gzip
import pickle
from argparse import Namespace
from array import array
from typing import Any, List, Optional, Sequence, Tuple

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


def _to_bool_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return 1 if x else 0
    if isinstance(x, (int, float)):
        return 1 if x > 0 else 0
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return 1
        if s in ("0", "false", "f", "no", "n", "off"):
            return 0
        try:
            v = float(s)
            return 1 if v > 0 else 0
        except Exception:
            return None
    return None


def _extract_list_from_json_obj(obj: Any) -> Optional[List[Any]]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("trace", "traces", "availability", "avail", "spot", "spots", "data", "values", "series"):
            v = obj.get(k, None)
            if isinstance(v, list):
                return v
        for _, v in obj.items():
            if isinstance(v, list):
                return v
            if isinstance(v, dict):
                vv = _extract_list_from_json_obj(v)
                if vv is not None:
                    return vv
    return None


def _parse_trace_sequence(seq: Sequence[Any]) -> List[int]:
    out: List[int] = []
    for e in seq:
        b = _to_bool_int(e)
        if b is not None:
            out.append(b)
            continue
        if isinstance(e, dict):
            for k in ("available", "avail", "spot", "value", "v", "y"):
                if k in e:
                    bb = _to_bool_int(e[k])
                    if bb is not None:
                        out.append(bb)
                        break
            else:
                found = None
                for _, vv in e.items():
                    bb = _to_bool_int(vv)
                    if bb is not None:
                        found = bb
                        break
                if found is None:
                    continue
                out.append(found)
            continue
        if isinstance(e, (list, tuple)) and len(e) > 0:
            bb = _to_bool_int(e[-1])
            if bb is not None:
                out.append(bb)
                continue
    return out


def _read_text_trace(text: str) -> List[int]:
    out: List[int] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tok = s
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if parts:
                tok = parts[-1]
        else:
            parts = s.split()
            if parts:
                tok = parts[-1]
        b = _to_bool_int(tok)
        if b is not None:
            out.append(b)
    return out


def _open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rb"), path[:-3].lower()
    return open(path, "rb"), path.lower()


def _load_trace_file(path: str) -> List[int]:
    f, low = _open_maybe_gzip(path)
    try:
        if low.endswith(".pkl") or low.endswith(".pickle"):
            obj = pickle.load(f)
            if isinstance(obj, (list, tuple)):
                return _parse_trace_sequence(obj)
            if isinstance(obj, dict):
                seq = _extract_list_from_json_obj(obj)
                if seq is not None:
                    return _parse_trace_sequence(seq)
            return []
        if low.endswith(".json"):
            obj = json.load(f)
            seq = _extract_list_from_json_obj(obj)
            if seq is None and isinstance(obj, (list, tuple)):
                seq = list(obj)
            if seq is None:
                return []
            return _parse_trace_sequence(seq)
        raw = f.read()
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            return []
        return _read_text_trace(text)
    finally:
        try:
            f.close()
        except Exception:
            pass


class Solution(MultiRegionStrategy):
    NAME = "deadline_aware_multiregion"

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
        self._have_traces = False
        self._avail: List[bytearray] = []
        self._run_len: List[array] = []
        self._any_spot: bytearray = bytearray()
        self._best_region: array = array("h")
        self._best_run: array = array("I")
        self._suffix_any_seconds: List[float] = []
        self._num_regions = 0
        self._n_steps = 0

        self._done = 0.0
        self._done_len = 0

        self._lock_on_demand = False

        self._env_has_spot_attr = None
        self._env_has_spot_call = None
        self._init_env_spot_query()

        self._prepare_traces()

        gap = float(getattr(self.env, "gap_seconds", 1.0))
        ro = float(self.restart_overhead)
        self._safety_buffer = max(2.0 * ro, 5.0 * gap)
        self._pause_margin = max(1.0 * ro, 2.0 * gap)
        self._switch_back_min_steps = int(math.ceil((3.0 * ro) / max(gap, 1e-9)))

        return self

    def _init_env_spot_query(self):
        env = getattr(self, "env", None)
        if env is None:
            return
        if hasattr(env, "get_has_spot") and callable(getattr(env, "get_has_spot")):
            self._env_has_spot_call = getattr(env, "get_has_spot")
            return
        if hasattr(env, "has_spot"):
            attr = getattr(env, "has_spot")
            if callable(attr):
                self._env_has_spot_call = attr
            else:
                self._env_has_spot_attr = "has_spot"
            return
        for name in ("spot_available", "get_spot_available"):
            if hasattr(env, name):
                a = getattr(env, name)
                if callable(a):
                    self._env_has_spot_call = a
                else:
                    self._env_has_spot_attr = name
                return

    def _query_env_has_spot(self) -> Optional[bool]:
        env = self.env
        try:
            if self._env_has_spot_call is not None:
                v = self._env_has_spot_call()
                if isinstance(v, bool):
                    return v
                b = _to_bool_int(v)
                if b is not None:
                    return bool(b)
                return None
            if self._env_has_spot_attr is not None:
                v = getattr(env, self._env_has_spot_attr, None)
                if isinstance(v, bool):
                    return v
                b = _to_bool_int(v)
                if b is not None:
                    return bool(b)
                return None
        except Exception:
            return None
        return None

    def _prepare_traces(self):
        try:
            num_regions = int(self.env.get_num_regions())
        except Exception:
            num_regions = len(self._trace_files)
        self._num_regions = max(0, num_regions)

        if self._num_regions <= 0 or not self._trace_files:
            self._have_traces = False
            return

        gap = float(getattr(self.env, "gap_seconds", 1.0))
        deadline = float(self.deadline)
        n_req = int(math.ceil(deadline / max(gap, 1e-9))) + 2
        if n_req <= 0:
            self._have_traces = False
            return

        raw_traces: List[List[int]] = []
        min_len = None
        for i in range(min(self._num_regions, len(self._trace_files))):
            p = self._trace_files[i]
            if not p or not os.path.exists(p):
                raw_traces.append([])
                continue
            tr = _load_trace_file(p)
            raw_traces.append(tr)
            if tr:
                min_len = len(tr) if min_len is None else min(min_len, len(tr))

        if min_len is None or min_len <= 0:
            self._have_traces = False
            return

        n_use = min(n_req, min_len)
        self._n_steps = n_req

        avail: List[bytearray] = []
        for r in range(self._num_regions):
            tr = raw_traces[r] if r < len(raw_traces) else []
            ba = bytearray(n_req)
            if tr:
                m = min(len(tr), n_req)
                for t in range(m):
                    ba[t] = 1 if tr[t] else 0
            avail.append(ba)
        self._avail = avail

        run_len: List[array] = []
        for r in range(self._num_regions):
            rl = array("I", [0]) * n_req
            ar = avail[r]
            streak = 0
            for t in range(n_req - 1, -1, -1):
                if ar[t]:
                    streak += 1
                    rl[t] = streak
                else:
                    streak = 0
                    rl[t] = 0
            run_len.append(rl)
        self._run_len = run_len

        any_spot = bytearray(n_req)
        best_region = array("h", [-1]) * n_req
        best_run = array("I", [0]) * n_req
        for t in range(n_req):
            br = 0
            bi = -1
            for r in range(self._num_regions):
                l = run_len[r][t]
                if l > br:
                    br = l
                    bi = r
            if bi >= 0:
                any_spot[t] = 1
                best_region[t] = bi
                best_run[t] = br
        self._any_spot = any_spot
        self._best_region = best_region
        self._best_run = best_run

        suffix_any_seconds = [0.0] * (n_req + 1)
        for t in range(n_req - 1, -1, -1):
            suffix_any_seconds[t] = suffix_any_seconds[t + 1] + (gap if any_spot[t] else 0.0)
        self._suffix_any_seconds = suffix_any_seconds

        self._have_traces = True

    def _update_done(self):
        lst = self.task_done_time
        n = len(lst)
        i = self._done_len
        while i < n:
            self._done += float(lst[i])
            i += 1
        self._done_len = n

    def _time_index(self) -> int:
        gap = float(getattr(self.env, "gap_seconds", 1.0))
        if gap <= 0:
            return 0
        t = int(self.env.elapsed_seconds / gap + 1e-9)
        if self._have_traces and self._n_steps > 0:
            if t < 0:
                return 0
            if t >= self._n_steps:
                return self._n_steps - 1
        return max(0, t)

    def _urgent_need_on_demand(self, remaining_work: float, time_left: float, last_cluster_type: ClusterType) -> bool:
        if time_left <= 0:
            return True
        ro = float(self.restart_overhead)
        rem_ov = float(getattr(self, "remaining_restart_overhead", 0.0) or 0.0)
        if last_cluster_type == ClusterType.ON_DEMAND:
            startup = max(0.0, rem_ov)
        else:
            startup = max(ro, rem_ov)
        return time_left <= remaining_work + startup + self._safety_buffer

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_done()

        remaining_work = float(self.task_duration) - float(self._done)
        if remaining_work <= 0:
            return ClusterType.NONE

        time_left = float(self.deadline) - float(self.env.elapsed_seconds)
        if time_left <= 0:
            return ClusterType.NONE

        if self._lock_on_demand or self._urgent_need_on_demand(remaining_work, time_left, last_cluster_type):
            self._lock_on_demand = True
            return ClusterType.ON_DEMAND

        rem_ov = float(getattr(self, "remaining_restart_overhead", 0.0) or 0.0)
        cur_region = int(self.env.get_current_region())

        t = self._time_index()
        gap = float(getattr(self.env, "gap_seconds", 1.0))
        ro = float(self.restart_overhead)

        if self._have_traces:
            if t >= self._n_steps:
                t = self._n_steps - 1
            best_r = int(self._best_region[t])
            spot_any = best_r >= 0
            best_run_steps = int(self._best_run[t]) if spot_any else 0
            cur_has_spot_now = bool(has_spot)
            if cur_region < 0 or cur_region >= self._num_regions:
                cur_has_spot_now = bool(has_spot)
            else:
                if has_spot:
                    cur_has_spot_now = True
                else:
                    cur_has_spot_now = False
        else:
            best_r = -1
            spot_any = bool(has_spot)
            best_run_steps = 0
            cur_has_spot_now = bool(has_spot)

        if rem_ov > 0:
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            if spot_any:
                if cur_has_spot_now and last_cluster_type == ClusterType.SPOT:
                    return ClusterType.SPOT
                if self._have_traces and best_r >= 0 and best_r != cur_region:
                    try:
                        self.env.switch_region(best_r)
                        env_sp = self._query_env_has_spot()
                        if env_sp is not None:
                            return ClusterType.SPOT if env_sp else ClusterType.ON_DEMAND
                    except Exception:
                        pass
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        if spot_any:
            if last_cluster_type == ClusterType.SPOT and cur_has_spot_now:
                return ClusterType.SPOT

            if last_cluster_type == ClusterType.ON_DEMAND:
                if self._have_traces and best_r >= 0 and best_run_steps >= self._switch_back_min_steps:
                    if best_r != cur_region:
                        try:
                            self.env.switch_region(best_r)
                        except Exception:
                            pass
                    env_sp = self._query_env_has_spot()
                    if env_sp is None:
                        return ClusterType.SPOT
                    return ClusterType.SPOT if env_sp else ClusterType.ON_DEMAND
                return ClusterType.ON_DEMAND

            if self._have_traces and best_r >= 0 and best_r != cur_region:
                try:
                    self.env.switch_region(best_r)
                except Exception:
                    pass
                env_sp = self._query_env_has_spot()
                if env_sp is None:
                    return ClusterType.SPOT
                return ClusterType.SPOT if env_sp else ClusterType.ON_DEMAND

            if cur_has_spot_now:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        if self._have_traces:
            future_spot_sec = self._suffix_any_seconds[t] if t < len(self._suffix_any_seconds) else 0.0
            margin = self._pause_margin + 0.5 * ro
            if remaining_work > max(0.0, future_spot_sec - margin):
                return ClusterType.ON_DEMAND
            return ClusterType.NONE

        if time_left > remaining_work + self._safety_buffer:
            return ClusterType.NONE
        return ClusterType.ON_DEMAND