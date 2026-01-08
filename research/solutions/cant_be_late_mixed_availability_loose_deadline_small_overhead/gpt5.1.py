import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_solution"

    def __init__(self, args=None):
        super().__init__(args)
        self.args = args
        self.force_on_demand = False
        self._initialized = False
        self._switch_threshold = None  # seconds
        self._task_done_cached = 0.0
        self._task_done_list_len = 0

    def solve(self, spec_path: str) -> "Solution":
        # Optional: could parse spec_path if needed.
        return self

    def _initialize(self):
        env = self.env
        gap = getattr(env, "gap_seconds", 60.0)
        if gap is None or gap <= 0:
            gap = 60.0
        overhead = getattr(self, "restart_overhead", 0.0)
        if overhead is None or overhead < 0.0:
            overhead = 0.0

        # Threshold (seconds) of spare slack at which we permanently switch to on-demand.
        # Must be > gap + overhead to absorb last-step discretization and one restart.
        base_threshold = max(4.0 * gap, 10.0 * overhead, 900.0)  # at least 15 minutes
        self._switch_threshold = base_threshold
        self._initialized = True

    def _segment_to_duration(self, seg) -> float:
        """Convert a task_done_time entry to a duration in seconds."""
        try:
            if isinstance(seg, (list, tuple)):
                if len(seg) >= 2:
                    start = float(seg[0])
                    end = float(seg[1])
                    return max(0.0, end - start)
            elif hasattr(seg, "__len__") and not isinstance(seg, (str, bytes, int, float)):
                # Handle array-like objects (e.g., numpy arrays)
                try:
                    if len(seg) >= 2:
                        start = float(seg[0])
                        end = float(seg[1])
                        return max(0.0, end - start)
                except TypeError:
                    pass
            # Fallback: treat as scalar duration
            return float(seg)
        except (TypeError, ValueError):
            return 0.0

    def _compute_done_work(self) -> float:
        """Compute total completed work (seconds) from task_done_time."""
        segments = getattr(self, "task_done_time", None)
        if not segments:
            return 0.0
        try:
            seg_list = list(segments)
        except TypeError:
            return 0.0

        n = len(seg_list)
        # If list shrank (unexpected), recompute from scratch.
        if n < self._task_done_list_len:
            total = 0.0
            for seg in seg_list:
                total += self._segment_to_duration(seg)
            self._task_done_list_len = n
            self._task_done_cached = total
            return total

        # Incrementally add new segments if list grew.
        total = self._task_done_cached
        for idx in range(self._task_done_list_len, n):
            total += self._segment_to_duration(seg_list[idx])
        self._task_done_list_len = n
        self._task_done_cached = total
        return total

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._initialized:
            self._initialize()

        # Compute remaining work.
        done_work = self._compute_done_work()
        task_duration = float(self.task_duration)
        remaining_work = task_duration - done_work
        if remaining_work <= 0.0:
            # Job is done; no need to run any instances.
            return ClusterType.NONE

        # Time remaining until deadline.
        elapsed = float(self.env.elapsed_seconds)
        deadline = float(self.deadline)
        remaining_time = deadline - elapsed

        # If somehow past deadline already, still try best-effort with on-demand.
        if remaining_time < 0.0:
            remaining_time = 0.0

        # Spare slack if we were to run purely on on-demand from now.
        spare_slack = remaining_time - remaining_work  # seconds

        # Decide whether to permanently switch to on-demand.
        # Once switched, never go back to spot to avoid deadline risk.
        if self.force_on_demand or spare_slack <= self._switch_threshold:
            self.force_on_demand = True
            return ClusterType.ON_DEMAND

        # Still in slack zone: prefer spot when available, otherwise idle.
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)