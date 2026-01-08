from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_on_demand = False
        self._safety_margin_seconds = None

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _initialize_if_needed(self):
        if self._safety_margin_seconds is not None:
            return
        try:
            deadline = float(self.deadline)
            task_duration = float(self.task_duration)
            restart_overhead = float(self.restart_overhead)
            gap = float(self.env.gap_seconds)
        except Exception:
            # Fallback: minimal margin if something is missing
            self._safety_margin_seconds = 0.0
            return

        total_slack = max(deadline - task_duration, 0.0)
        if total_slack <= 0.0:
            margin = restart_overhead
        else:
            frac = 0.15  # use 15% of total slack as baseline margin
            margin = max(total_slack * frac,
                         2.0 * restart_overhead,
                         4.0 * gap)
            # Do not consume more than half of slack as safety margin
            max_margin = 0.5 * total_slack
            if margin > max_margin:
                margin = max_margin
        self._safety_margin_seconds = margin

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._initialize_if_needed()

        # Compute remaining work
        done_list = getattr(self, "task_done_time", None)
        if done_list:
            work_done = float(sum(done_list))
        else:
            work_done = 0.0

        remaining = max(float(self.task_duration) - work_done, 0.0)
        if remaining <= 0.0:
            return ClusterType.NONE

        elapsed = float(self.env.elapsed_seconds)
        deadline = float(self.deadline)
        time_left = deadline - elapsed

        if time_left <= 0.0:
            # Already at or beyond deadline; best effort is on-demand
            return ClusterType.ON_DEMAND

        # Decide whether we must switch to on-demand now to guarantee completion
        if not self.force_on_demand:
            gap = float(self.env.gap_seconds)
            margin = float(self._safety_margin_seconds or 0.0)
            restart_overhead = float(self.restart_overhead)

            # Safe slack after accounting for worst-case future ON_DEMAND start
            safe_slack = time_left - (remaining + restart_overhead + margin)

            # If we cannot afford to wait one more step, lock into ON_DEMAND
            if safe_slack < gap:
                self.force_on_demand = True

        if self.force_on_demand:
            return ClusterType.ON_DEMAND

        # Risk-taking phase: use spot if available, otherwise pause
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)