from __future__ import annotations

from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args: Any):
        try:
            super().__init__(args)
        except Exception:
            pass
        self._ema_p: Optional[float] = None
        self._beta_p: float = 0.05
        self._od_lock: bool = False
        self._last_has_spot: Optional[bool] = None
        self._steps: int = 0
        self._spot_steps: int = 0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _done_seconds(self) -> float:
        val = getattr(self, "task_done_time", None)
        if val is None:
            return 0.0

        td = float(getattr(self, "task_duration", 0.0) or 0.0)

        try:
            if isinstance(val, (int, float)):
                done = float(val)
                if done < 0:
                    return 0.0
                if td > 0:
                    return min(done, td)
                return done

            if isinstance(val, dict):
                if "done" in val and isinstance(val["done"], (int, float)):
                    done = float(val["done"])
                    if td > 0:
                        return max(0.0, min(done, td))
                    return max(0.0, done)
                if "segments" in val:
                    val = val["segments"]
                else:
                    return 0.0

            if isinstance(val, (list, tuple)):
                if not val:
                    return 0.0

                total = 0.0
                mx = 0.0
                mono = True
                prev = None

                for item in val:
                    seg = 0.0
                    if isinstance(item, (int, float)):
                        x = float(item)
                        mx = max(mx, x)
                        if prev is not None and x < prev - 1e-9:
                            mono = False
                        prev = x
                        seg = x
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        a = float(item[0])
                        b = float(item[1])
                        if b >= a:
                            seg = b - a
                        else:
                            seg = max(0.0, b)
                    elif isinstance(item, dict):
                        if "duration" in item and isinstance(item["duration"], (int, float)):
                            seg = float(item["duration"])
                        elif "start" in item and "end" in item:
                            a = float(item["start"])
                            b = float(item["end"])
                            seg = max(0.0, b - a)
                        elif "done" in item and isinstance(item["done"], (int, float)):
                            seg = float(item["done"])
                            mx = max(mx, seg)
                            if prev is not None and seg < prev - 1e-9:
                                mono = False
                            prev = seg
                        else:
                            seg = 0.0
                    else:
                        seg = 0.0

                    if seg > 0:
                        total += seg

                if td > 0:
                    # Heuristic: if values look cumulative (monotone, max close to td, sum too large), use max.
                    if mono and mx <= td * 1.05 and total > td * 1.20:
                        return max(0.0, min(mx, td))
                    return max(0.0, min(total, td))

                if mono and total > mx * 1.20:
                    return max(0.0, total)
                return max(0.0, total)
        except Exception:
            return 0.0

        return 0.0

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._steps += 1
        if has_spot:
            self._spot_steps += 1

        if self._ema_p is None:
            self._ema_p = 1.0 if has_spot else 0.0
        else:
            x = 1.0 if has_spot else 0.0
            self._ema_p = (1.0 - self._beta_p) * self._ema_p + self._beta_p * x

        self._last_has_spot = has_spot

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        done = self._done_seconds()
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        remaining = max(0.0, task_duration - done)

        if remaining <= 1e-9:
            return ClusterType.NONE

        time_left = deadline - elapsed
        if time_left <= 0.0:
            return ClusterType.NONE

        # Guard buffers (discrete-time + restart overhead)
        base_lock_buffer = max(0.0, ro + 6.0 * gap)
        switch_back_buffer = max(0.0, 3.0 * ro + 10.0 * gap)
        wait_buffer = max(0.0, 2.0 * ro + 8.0 * gap)
        wait_buffer_from_od = wait_buffer + max(0.0, ro * 0.5 + 2.0 * gap)

        need_switch_to_od = (last_cluster_type != ClusterType.ON_DEMAND)
        needed_od_now = remaining + (ro if need_switch_to_od else 0.0)
        slack_od = time_left - needed_od_now

        if self._od_lock or slack_od <= base_lock_buffer:
            self._od_lock = True
            return ClusterType.ON_DEMAND

        p = float(self._ema_p or 0.0)
        p = 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)

        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND:
                if slack_od >= switch_back_buffer and p >= 0.55 and remaining >= 4.0 * ro:
                    return ClusterType.SPOT
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        # No spot available
        if p <= 0.08:
            return ClusterType.ON_DEMAND

        if last_cluster_type == ClusterType.ON_DEMAND:
            if slack_od >= wait_buffer_from_od:
                return ClusterType.NONE
            return ClusterType.ON_DEMAND

        if slack_od >= wait_buffer:
            return ClusterType.NONE
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)