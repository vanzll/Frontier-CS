import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "deadline_guard_v3"

    def __init__(self, args=None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass
        self.args = args
        self._commit_od = False
        self._min_margin_seconds = 60.0
        self._margin_multiplier = 1.5
        # For efficient progress tracking
        self._accum_done = 0.0
        self._last_tdt_count = 0
        self._last_tdt_id = None

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _seg_duration(self, seg) -> float:
        if seg is None:
            return 0.0
        try:
            if isinstance(seg, (int, float)):
                return float(seg)
            if isinstance(seg, dict):
                if 'duration' in seg:
                    return float(seg['duration'])
                if 'end' in seg and 'start' in seg:
                    return float(seg['end']) - float(seg['start'])
            # Try tuple/list [start, end]
            if hasattr(seg, '__len__'):
                if len(seg) >= 2:
                    return float(seg[1]) - float(seg[0])
        except Exception:
            pass
        try:
            return float(seg)
        except Exception:
            return 0.0

    def _update_done_accum(self):
        tdt = getattr(self, 'task_done_time', None)
        if tdt is None:
            return self._accum_done
        # If it's a scalar, use it directly
        if isinstance(tdt, (int, float)):
            try:
                self._accum_done = float(tdt)
                self._last_tdt_count = 1
                self._last_tdt_id = id(tdt)
                return self._accum_done
            except Exception:
                pass
        # Handle sized iterables efficiently
        try:
            length = len(tdt)
            current_id = id(tdt)
        except Exception:
            # Fallback: try to iterate and sum
            total = 0.0
            try:
                for seg in tdt:
                    total += self._seg_duration(seg)
            except Exception:
                total = 0.0
            self._accum_done = total
            self._last_tdt_count = 0
            self._last_tdt_id = None
            return self._accum_done

        if current_id != self._last_tdt_id:
            # List object replaced; rescan fully
            total = 0.0
            for seg in tdt:
                total += self._seg_duration(seg)
            self._accum_done = total
            self._last_tdt_id = current_id
            self._last_tdt_count = length
            return self._accum_done

        # Same list: sum only newly appended segments if any
        if length > self._last_tdt_count:
            for i in range(self._last_tdt_count, length):
                try:
                    self._accum_done += self._seg_duration(tdt[i])
                except Exception:
                    pass
            self._last_tdt_count = length

        return self._accum_done

    def _remaining_seconds(self) -> float:
        try:
            total = float(self.task_duration)
        except Exception:
            return 0.0
        done = self._update_done_accum()
        if done < 0.0:
            done = 0.0
        if done > total:
            done = total
        return max(total - done, 0.0)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        now = float(getattr(self.env, 'elapsed_seconds', 0.0) or 0.0)
        gap = float(getattr(self.env, 'gap_seconds', 60.0) or 60.0)
        current_type = getattr(self.env, 'cluster_type', last_cluster_type)
        deadline = float(getattr(self, 'deadline', now + 1e9))
        restart_overhead = float(getattr(self, 'restart_overhead', 0.0))

        remaining = self._remaining_seconds()
        time_left = deadline - now

        if remaining <= 0.0 or time_left <= 0.0:
            self._commit_od = False
            return ClusterType.NONE

        margin = max(self._margin_multiplier * gap, self._min_margin_seconds)

        on_od_now = (current_type == ClusterType.ON_DEMAND)
        overhead_needed = 0.0 if (on_od_now or self._commit_od) else restart_overhead
        slack = time_left - (remaining + overhead_needed)

        if slack <= margin:
            self._commit_od = True

        if self._commit_od:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable: decide to wait or switch to OD
        # Wait if we can afford to consume one more gap of slack plus margin buffer
        if (slack - gap) >= margin:
            return ClusterType.NONE

        self._commit_od = True
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        try:
            parser.add_argument('--seed', type=int, default=0)
        except Exception:
            pass
        args, _ = parser.parse_known_args()
        return cls(args)