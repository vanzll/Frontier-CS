import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "jit_deadline_guard"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass

        self._committed_to_od = False

        self._tdt_mode: Optional[str] = None  # "scalar" | "series" | "durations" | "segments"
        self._seg_idx = 0
        self._seg_sum = 0.0
        self._last_done = 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _is_number(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def _parse_segment_duration(self, seg: Any) -> float:
        if seg is None:
            return 0.0

        # Numeric => treat as duration
        if self._is_number(seg):
            return float(seg)

        # Dict segment formats
        if isinstance(seg, dict):
            d = seg
            if "duration" in d and self._is_number(d["duration"]):
                return float(d["duration"])
            if "seconds" in d and self._is_number(d["seconds"]):
                return float(d["seconds"])
            if "time" in d and self._is_number(d["time"]):
                return float(d["time"])
            if "start" in d and "end" in d and self._is_number(d["start"]) and self._is_number(d["end"]):
                a = float(d["start"])
                b = float(d["end"])
                return max(0.0, b - a)
            if "begin" in d and "finish" in d and self._is_number(d["begin"]) and self._is_number(d["finish"]):
                a = float(d["begin"])
                b = float(d["finish"])
                return max(0.0, b - a)
            if "done" in d and self._is_number(d["done"]):
                # Ambiguous; treated as duration contribution.
                return float(d["done"])
            return 0.0

        # Tuple/list segment formats
        if isinstance(seg, (tuple, list)):
            if len(seg) >= 2:
                a, b = seg[0], seg[1]
                # (start, end)
                if self._is_number(a) and self._is_number(b):
                    af, bf = float(a), float(b)
                    if bf >= af:
                        return bf - af
                    # Fallback: treat second as duration if looks like it
                    return max(0.0, bf)
                # (cluster_type, duration) or (name, duration)
                if self._is_number(b):
                    return float(b)
            return 0.0

        # Object with duration-like attributes
        for attr in ("duration", "seconds", "time"):
            if hasattr(seg, attr):
                v = getattr(seg, attr)
                if self._is_number(v):
                    return float(v)

        return 0.0

    def _infer_tdt_mode(self, tdt: Any) -> str:
        if tdt is None:
            return "scalar"
        if self._is_number(tdt):
            return "scalar"
        if not isinstance(tdt, (list, tuple)):
            return "scalar"
        if len(tdt) == 0:
            return "segments"

        first = tdt[0]
        if self._is_number(first):
            # Numeric list: could be cumulative series or list of durations.
            # Detect monotonic (series) vs non-monotonic (durations).
            n = len(tdt)
            check_n = min(n - 1, 50)
            mono = True
            prev = float(tdt[0])
            for i in range(1, check_n + 1):
                cur = float(tdt[i])
                if cur + 1e-12 < prev:
                    mono = False
                    break
                prev = cur
            if mono:
                return "series"
            return "durations"

        # Non-numeric elements: likely segments
        return "segments"

    def _get_done_seconds(self) -> float:
        tdt = getattr(self, "task_done_time", None)

        if self._tdt_mode is None:
            self._tdt_mode = self._infer_tdt_mode(tdt)

        done = 0.0

        if self._tdt_mode == "scalar":
            if self._is_number(tdt):
                done = float(tdt)
            elif isinstance(tdt, (list, tuple)) and len(tdt) > 0 and self._is_number(tdt[-1]):
                done = float(tdt[-1])
            else:
                done = float(self._last_done)

        elif self._tdt_mode == "series":
            if isinstance(tdt, (list, tuple)) and len(tdt) > 0 and self._is_number(tdt[-1]):
                done = float(tdt[-1])
            else:
                # fallback
                done = float(self._last_done)

        elif self._tdt_mode == "durations":
            if isinstance(tdt, (list, tuple)):
                # Incremental sum for numeric durations list
                if self._seg_idx > len(tdt):
                    self._seg_idx = 0
                    self._seg_sum = 0.0
                for i in range(self._seg_idx, len(tdt)):
                    x = tdt[i]
                    if self._is_number(x):
                        self._seg_sum += float(x)
                    else:
                        self._seg_sum += self._parse_segment_duration(x)
                self._seg_idx = len(tdt)
                done = self._seg_sum
            else:
                done = float(self._last_done)

        else:  # "segments"
            if isinstance(tdt, (list, tuple)):
                if self._seg_idx > len(tdt):
                    self._seg_idx = 0
                    self._seg_sum = 0.0
                for i in range(self._seg_idx, len(tdt)):
                    self._seg_sum += self._parse_segment_duration(tdt[i])
                self._seg_idx = len(tdt)
                done = self._seg_sum
            else:
                done = float(self._last_done)

        # Monotonic clamp
        if not math.isfinite(done):
            done = float(self._last_done)
        done = max(done, float(self._last_done))

        td = float(getattr(self, "task_duration", 0.0) or 0.0)
        if td > 0.0:
            done = min(done, td)

        self._last_done = done
        return done

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = getattr(self, "env", None)
        elapsed = float(getattr(env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(env, "gap_seconds", 1.0) or 1.0)
        if gap <= 0.0:
            gap = 1.0

        td = float(getattr(self, "task_duration", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        restart = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        done = self._get_done_seconds()
        remaining = max(0.0, td - done)

        if remaining <= 0.0:
            return ClusterType.NONE

        # If we've ever started on-demand (or inherited an OD state), stay on-demand for safety.
        if self._committed_to_od or last_cluster_type == ClusterType.ON_DEMAND:
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        time_left = max(0.0, deadline - elapsed)

        # Buffers expressed relative to environment scales (works whether "seconds" are truly seconds or hours).
        spot_buffer = max(0.25 * gap, 0.25 * restart, 0.0005 * td) if td > 0.0 else max(0.25 * gap, 0.25 * restart)
        nospot_buffer = max(0.5 * gap, 0.5 * restart, 0.001 * td) if td > 0.0 else max(0.5 * gap, 0.5 * restart)

        overhead_start_od = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else restart

        if has_spot:
            # If spot is available, allow spot as long as we can still finish if spot disappears next step.
            critical = remaining + overhead_start_od + spot_buffer
        else:
            # If spot is unavailable and we choose NONE now, we lose one full gap of time.
            # Ensure that after waiting one step, we can still complete on on-demand.
            critical = remaining + restart + gap + nospot_buffer

        if time_left <= critical:
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)