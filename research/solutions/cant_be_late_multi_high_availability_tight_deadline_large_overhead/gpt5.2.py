import json
import math
from argparse import Namespace
from array import array
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _read_json_maybe(text: str) -> Optional[Any]:
    s = text.lstrip()
    if not s:
        return None
    if s[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_trace_sequence(obj: Any) -> Optional[Sequence[Any]]:
    if obj is None:
        return None
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("availability", "avail", "spot", "trace", "data", "values"):
            v = obj.get(k)
            if isinstance(v, list):
                return v
        for v in obj.values():
            if isinstance(v, list):
                return v
    return None


def _parse_token_to_bool(tok: str) -> Optional[bool]:
    t = tok.strip().lower()
    if not t:
        return None
    if t in ("1", "true", "t", "yes", "y", "spot", "available", "up"):
        return True
    if t in ("0", "false", "f", "no", "n", "none", "unavailable", "down"):
        return False
    try:
        v = float(t)
        return v >= 0.5
    except Exception:
        return None


def _load_trace_file(path: str) -> List[int]:
    try:
        with open(path, "r") as f:
            text = f.read()
    except Exception:
        return []
    obj = _read_json_maybe(text)
    seq = _extract_trace_sequence(obj)
    if seq is not None:
        out: List[int] = []
        for x in seq:
            if isinstance(x, bool):
                out.append(1 if x else 0)
            elif isinstance(x, (int, float)):
                out.append(1 if float(x) >= 0.5 else 0)
            elif isinstance(x, str):
                b = _parse_token_to_bool(x)
                if b is None:
                    continue
                out.append(1 if b else 0)
            else:
                try:
                    out.append(1 if float(x) >= 0.5 else 0)
                except Exception:
                    continue
        return out

    out2: List[int] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
        else:
            parts = [p for p in s.split() if p]
        if not parts:
            continue
        # Prefer last token as availability value
        b = _parse_token_to_bool(parts[-1])
        if b is None and len(parts) >= 2:
            b = _parse_token_to_bool(parts[-2])
        if b is None:
            continue
        out2.append(1 if b else 0)
    return out2


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_multiregion_v1"

    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path) as f:
            config = json.load(f)

        self._config: Dict[str, Any] = dict(config)
        self._trace_files: List[str] = list(config.get("trace_files", []))

        args = Namespace(
            deadline_hours=_safe_float(config.get("deadline")),
            task_duration_hours=[_safe_float(config.get("duration"))],
            restart_overhead_hours=[_safe_float(config.get("overhead"))],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        self._initialized = False
        self._use_traces = True
        self._committed_on_demand = False

        self._n_steps = 0
        self._gap = 0.0
        self._task_duration_sec = 0.0
        self._deadline_sec = 0.0
        self._restart_overhead_sec = 0.0
        self._min_steps_positive = 1

        self._avail_by_region: List[bytearray] = []
        self._run_by_region: List[array] = []
        self._next_by_region: List[array] = []
        self._best_region: Optional[array] = None
        self._best_run: Optional[array] = None
        self._next_good: Optional[array] = None

        self._work_done = 0.0
        self._task_done_len = 0

        return self

    def _lazy_init(self) -> None:
        if self._initialized:
            return

        self._gap = float(getattr(self.env, "gap_seconds", 1.0)) or 1.0

        td = getattr(self, "task_duration", 0.0)
        dl = getattr(self, "deadline", 0.0)
        ro = getattr(self, "restart_overhead", 0.0)

        if isinstance(td, (list, tuple)):
            td = td[0] if td else 0.0
        if isinstance(dl, (list, tuple)):
            dl = dl[0] if dl else 0.0
        if isinstance(ro, (list, tuple)):
            ro = ro[0] if ro else 0.0

        self._task_duration_sec = float(td)
        self._deadline_sec = float(dl)
        self._restart_overhead_sec = float(ro)

        self._n_steps = int(math.ceil(self._deadline_sec / self._gap)) + 8
        if self._n_steps < 16:
            self._n_steps = 16

        self._min_steps_positive = int(math.floor(self._restart_overhead_sec / self._gap)) + 1
        if self._min_steps_positive < 1:
            self._min_steps_positive = 1

        num_regions = int(self.env.get_num_regions())
        trace_files = self._trace_files[:num_regions]

        if len(trace_files) != num_regions:
            self._use_traces = False

        if self._use_traces:
            avail_by_region: List[bytearray] = []
            run_by_region: List[array] = []
            next_by_region: List[array] = []

            for i in range(num_regions):
                seq = _load_trace_file(trace_files[i])
                avail = bytearray(self._n_steps)
                m = min(len(seq), self._n_steps)
                for t in range(m):
                    avail[t] = 1 if seq[t] else 0
                avail_by_region.append(avail)

                run = array("I", [0]) * (self._n_steps + 1)
                nxt = array("I", [self._n_steps]) * (self._n_steps + 1)

                for t in range(self._n_steps - 1, -1, -1):
                    if avail[t]:
                        run[t] = run[t + 1] + 1
                        nxt[t] = t
                    else:
                        run[t] = 0
                        nxt[t] = nxt[t + 1]

                run_by_region.append(run)
                next_by_region.append(nxt)

            best_region = array("h", [-1]) * self._n_steps
            best_run = array("I", [0]) * self._n_steps

            for t in range(self._n_steps):
                br = 0
                bi = -1
                for i in range(num_regions):
                    rl = run_by_region[i][t]
                    if rl > br:
                        br = rl
                        bi = i
                best_region[t] = bi
                best_run[t] = br

            next_good = array("I", [self._n_steps]) * (self._n_steps + 1)
            next_good[self._n_steps] = self._n_steps
            for t in range(self._n_steps - 1, -1, -1):
                if best_run[t] >= self._min_steps_positive:
                    next_good[t] = t
                else:
                    next_good[t] = next_good[t + 1]

            self._avail_by_region = avail_by_region
            self._run_by_region = run_by_region
            self._next_by_region = next_by_region
            self._best_region = best_region
            self._best_run = best_run
            self._next_good = next_good
        else:
            self._avail_by_region = []
            self._run_by_region = []
            self._next_by_region = []
            self._best_region = None
            self._best_run = None
            self._next_good = None

        self._initialized = True

    def _update_work_done(self) -> None:
        td = self.task_done_time
        n = len(td)
        if n > self._task_done_len:
            s = 0.0
            for i in range(self._task_done_len, n):
                s += float(td[i])
            self._work_done += s
            self._task_done_len = n

    def _time_index(self) -> int:
        t = int(float(getattr(self.env, "elapsed_seconds", 0.0)) // self._gap)
        if t < 0:
            return 0
        if t >= self._n_steps:
            return self._n_steps - 1
        return t

    def _remaining_work_time(self) -> float:
        rem = self._task_duration_sec - self._work_done
        return rem if rem > 0.0 else 0.0

    def _required_time_if_on_demand(self, last_cluster_type: ClusterType) -> float:
        rem_work = self._remaining_work_time()
        if last_cluster_type == ClusterType.ON_DEMAND:
            overhead = float(getattr(self, "remaining_restart_overhead", 0.0) or 0.0)
        else:
            overhead = self._restart_overhead_sec
        return rem_work + overhead

    def _should_commit_on_demand(self, last_cluster_type: ClusterType, remaining_time: float) -> bool:
        gap = self._gap
        required = self._required_time_if_on_demand(last_cluster_type)
        return required + 2.0 * gap >= remaining_time

    def _wait_or_on_demand(self, last_cluster_type: ClusterType, t: int, remaining_time: float) -> ClusterType:
        rem_work = self._remaining_work_time()
        slack = remaining_time - rem_work
        if slack <= 0.0:
            return ClusterType.ON_DEMAND

        if not self._use_traces or self._next_good is None:
            if slack >= self._restart_overhead_sec + 2.0 * self._gap:
                return ClusterType.NONE
            return ClusterType.ON_DEMAND

        ng = int(self._next_good[t])
        if ng >= self._n_steps:
            return ClusterType.ON_DEMAND

        wait_seconds = (ng - t) * self._gap
        if slack >= wait_seconds + self._restart_overhead_sec + 2.0 * self._gap:
            return ClusterType.NONE
        return ClusterType.ON_DEMAND

    def _pick_spot_region(self, t: int, cur_region: int) -> int:
        if not self._use_traces or self._best_region is None:
            return cur_region
        br = int(self._best_region[t])
        if br < 0:
            return -1
        if br == cur_region:
            return cur_region

        cur_len = int(self._run_by_region[cur_region][t]) if self._run_by_region else 0
        best_len = int(self._run_by_region[br][t]) if self._run_by_region else 0

        if cur_len > 0:
            if (best_len - cur_len) * self._gap <= self._restart_overhead_sec:
                return cur_region
        return br

    def _spot_available_in_region(self, region: int, t: int, has_spot_current: bool, cur_region: int) -> bool:
        if not self._use_traces or not self._avail_by_region:
            return has_spot_current if region == cur_region else False
        return bool(self._avail_by_region[region][t])

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._lazy_init()
        self._update_work_done()

        if self._remaining_work_time() <= 0.0:
            return ClusterType.NONE

        t = self._time_index()
        cur_region = int(self.env.get_current_region())

        if self._use_traces and self._avail_by_region:
            try:
                expected = bool(self._avail_by_region[cur_region][t])
                if bool(has_spot) != expected:
                    self._use_traces = False
                    self._avail_by_region = []
                    self._run_by_region = []
                    self._next_by_region = []
                    self._best_region = None
                    self._best_run = None
                    self._next_good = None
            except Exception:
                self._use_traces = False

        remaining_time = self._deadline_sec - float(getattr(self.env, "elapsed_seconds", 0.0))

        if self._committed_on_demand or self._should_commit_on_demand(last_cluster_type, remaining_time):
            self._committed_on_demand = True
            return ClusterType.ON_DEMAND

        # If any spot is available now (or current has spot in fallback mode), prefer spot
        best_region = -1
        if self._use_traces and self._best_region is not None:
            best_region = int(self._best_region[t])
        else:
            best_region = cur_region if has_spot else -1

        if best_region >= 0:
            target = self._pick_spot_region(t, cur_region)
            if target < 0:
                return self._wait_or_on_demand(last_cluster_type, t, remaining_time)

            # Determine if starting spot would require a restart overhead and if the segment is long enough
            need_restart = (last_cluster_type != ClusterType.SPOT) or (target != cur_region)
            if need_restart and self._use_traces and self._run_by_region:
                if int(self._run_by_region[target][t]) < self._min_steps_positive:
                    return self._wait_or_on_demand(last_cluster_type, t, remaining_time)

            # Avoid resetting overhead while already paying it, unless necessary
            pending_overhead = float(getattr(self, "remaining_restart_overhead", 0.0) or 0.0)
            if pending_overhead > 0.0 and target != cur_region:
                cur_has = self._spot_available_in_region(cur_region, t, has_spot, cur_region)
                if last_cluster_type == ClusterType.SPOT and cur_has:
                    target = cur_region

            if target != cur_region:
                self.env.switch_region(int(target))

            # In fallback mode, only return SPOT if current region has spot per provided signal
            if not self._use_traces:
                return ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND

            return ClusterType.SPOT

        return self._wait_or_on_demand(last_cluster_type, t, remaining_time)