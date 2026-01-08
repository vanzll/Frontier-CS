import argparse
import json
import os
from enum import Enum
from typing import Any, Optional

try:
    from sky_spot.strategies.strategy import Strategy  # type: ignore
    from sky_spot.utils import ClusterType  # type: ignore
except Exception:
    class ClusterType(Enum):
        SPOT = "spot"
        ON_DEMAND = "on_demand"
        NONE = "none"

    class Strategy:
        def __init__(self, args: Optional[Any] = None):
            self.args = args
            self.env = None
            self.task_duration = 0.0
            self.task_done_time = []
            self.deadline = 0.0
            self.restart_overhead = 0.0


class Solution(Strategy):
    NAME = "deadline_guard_spot"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except Exception:
            self.args = args
        self._od_locked = False
        self._spot_avail_streak = 0
        self._cfg = {
            "urgent_buffer_restart_mult": 3.0,
            "urgent_buffer_gap_mult": 2.0,
            "switch_to_spot_streak": 3,
            "switch_to_spot_slack_restart_mult": 2.0,
            "switch_to_spot_slack_gap_mult": 1.0,
        }

    def solve(self, spec_path: str) -> "Solution":
        cfg = None
        if spec_path and isinstance(spec_path, str) and os.path.exists(spec_path):
            try:
                with open(spec_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = None
        if isinstance(cfg, dict):
            user_cfg = cfg.get("strategy", cfg.get("solution", cfg.get("config", cfg)))
            if isinstance(user_cfg, dict):
                for k in list(self._cfg.keys()):
                    if k in user_cfg:
                        self._cfg[k] = user_cfg[k]
        return self

    @staticmethod
    def _work_done_from_segments(segments: Any) -> float:
        if segments is None:
            return 0.0
        if isinstance(segments, (int, float)):
            return float(segments)
        total = 0.0
        try:
            it = list(segments)
        except Exception:
            return 0.0
        for seg in it:
            if seg is None:
                continue
            if isinstance(seg, (int, float)):
                total += float(seg)
            elif isinstance(seg, (tuple, list)) and len(seg) >= 2:
                try:
                    a = float(seg[0])
                    b = float(seg[1])
                    if b >= a:
                        total += (b - a)
                except Exception:
                    continue
            elif isinstance(seg, dict):
                added = False
                for k in ("duration", "work", "done", "seconds"):
                    v = seg.get(k, None)
                    if isinstance(v, (int, float)):
                        total += float(v)
                        added = True
                        break
                if not added and "start" in seg and "end" in seg:
                    try:
                        a = float(seg["start"])
                        b = float(seg["end"])
                        if b >= a:
                            total += (b - a)
                    except Exception:
                        pass
        return total

    def _buffers(self) -> tuple[float, float, int]:
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        urgent_buffer = max(
            self._cfg["urgent_buffer_gap_mult"] * gap,
            self._cfg["urgent_buffer_restart_mult"] * ro,
        )
        switch_slack = (
            self._cfg["switch_to_spot_slack_restart_mult"] * ro
            + self._cfg["switch_to_spot_slack_gap_mult"] * gap
            + urgent_buffer
        )
        streak = int(self._cfg["switch_to_spot_streak"])
        if streak < 1:
            streak = 1
        return urgent_buffer, switch_slack, streak

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if has_spot:
            self._spot_avail_streak += 1
        else:
            self._spot_avail_streak = 0

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)

        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        done = self._work_done_from_segments(getattr(self, "task_done_time", None))
        remaining = task_duration - done
        if remaining <= 0.0:
            return ClusterType.NONE

        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        time_left = deadline - elapsed
        if time_left <= 0.0:
            self._od_locked = True
            return ClusterType.ON_DEMAND

        restart_overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        urgent_buffer, switch_slack, switch_streak = self._buffers()

        if self._od_locked:
            return ClusterType.ON_DEMAND

        od_start_overhead = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else restart_overhead
        urgent = time_left <= (remaining + od_start_overhead + urgent_buffer)
        if urgent:
            self._od_locked = True
            return ClusterType.ON_DEMAND

        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND:
                slack = time_left - remaining
                if self._spot_avail_streak >= switch_streak and slack >= switch_slack:
                    return ClusterType.SPOT
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        if last_cluster_type == ClusterType.ON_DEMAND:
            return ClusterType.ON_DEMAND

        if (time_left - gap) >= (remaining + restart_overhead + urgent_buffer):
            return ClusterType.NONE

        self._od_locked = True
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)