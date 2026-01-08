import math

try:
    from sky_spot.strategies.strategy import Strategy
    from sky_spot.utils import ClusterType
except ImportError:  # Fallbacks for local testing; real evaluator provides these.
    from enum import Enum

    class ClusterType(Enum):
        SPOT = "spot"
        ON_DEMAND = "on_demand"
        NONE = "none"

    class Strategy:
        def __init__(self, args=None):
            class DummyEnv:
                elapsed_seconds = 0.0
                gap_seconds = 1.0
                cluster_type = ClusterType.NONE

            self.env = DummyEnv()
            self.task_duration = 0.0
            self.task_done_time = []
            self.deadline = 0.0
            self.restart_overhead = 0.0


class Solution(Strategy):
    NAME = "cant_be_late_buffered_spot_first"

    def __init__(self, args=None):
        try:
            super().__init__(args)
        except TypeError:
            super().__init__()
        # Progress tracking caches
        self._progress_cached_value = 0.0
        self._progress_cached_len = 0
        self._progress_list_id = None
        # Once set, we always use on-demand
        self._use_od_permanently = False

    def solve(self, spec_path: str) -> "Solution":
        # Optional: load configuration from spec_path if needed.
        return self

    def _compute_progress(self) -> float:
        """Compute total completed task time from self.task_done_time."""
        tdt = getattr(self, "task_done_time", None)

        # Numeric simple representation: treat as total progress directly.
        if isinstance(tdt, (int, float)):
            value = float(tdt)
            self._progress_cached_value = value
            self._progress_cached_len = 0
            self._progress_list_id = None
            return value

        # If not a list or None, fall back to cached value.
        if not isinstance(tdt, list):
            return self._progress_cached_value

        cur_id = id(tdt)
        n = len(tdt)

        # If the underlying list object changed or shrank, recompute from scratch.
        if cur_id != self._progress_list_id or n < self._progress_cached_len:
            total = 0.0
            for seg in tdt:
                try:
                    if isinstance(seg, (int, float)):
                        total += float(seg)
                    else:
                        start, end = seg
                        total += float(end) - float(start)
                except Exception:
                    continue
            self._progress_cached_value = total
            self._progress_cached_len = n
            self._progress_list_id = cur_id
            return total

        # Incremental update: same list object, grew in length.
        if n > self._progress_cached_len:
            total = self._progress_cached_value
            for i in range(self._progress_cached_len, n):
                seg = tdt[i]
                try:
                    if isinstance(seg, (int, float)):
                        total += float(seg)
                    else:
                        start, end = seg
                        total += float(end) - float(start)
                except Exception:
                    continue
            self._progress_cached_value = total
            self._progress_cached_len = n
            self._progress_list_id = cur_id
            return total

        # No changes.
        return self._progress_cached_value

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Basic environment attributes with safe defaults.
        env = self.env
        now = float(getattr(env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(env, "gap_seconds", 0.0) or 0.0)
        deadline = getattr(self, "deadline", None)
        task_duration = getattr(self, "task_duration", None)
        restart_overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        # If critical parameters are missing or gap is zero, fall back to simple policy.
        if deadline is None or task_duration is None or gap <= 0.0:
            if has_spot:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # Compute task progress and remaining work.
        progress = self._compute_progress()
        work_remaining = max(0.0, float(task_duration) - float(progress))
        time_remaining = float(deadline) - now

        # If task is done or we are out of time, don't spend more.
        if work_remaining <= 0.0 or time_remaining <= 0.0:
            return ClusterType.NONE

        # Required continuous on-demand time to finish from now (worst-case),
        # including at most one future restart overhead.
        required_od_time = work_remaining + restart_overhead

        # Slack we can afford to "waste" before we must switch to on-demand-only.
        slack = time_remaining - required_od_time

        # If we've already committed to on-demand permanently, keep using it.
        if self._use_od_permanently:
            return ClusterType.ON_DEMAND

        # Safety buffer: require at least two gap intervals of slack to risk
        # an additional step that might yield no progress (SPOT or NONE).
        # This ensures that after a worst-case wasted step we still have at
        # least one gap of spare time when we switch to on-demand.
        if slack < 2.0 * gap:
            self._use_od_permanently = True
            return ClusterType.ON_DEMAND

        # We have enough slack to risk a potentially wasted step.
        if has_spot:
            # Use cheaper spot when available.
            return ClusterType.SPOT

        # Spot not available and we can still afford to wait: stay idle to avoid OD cost.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)