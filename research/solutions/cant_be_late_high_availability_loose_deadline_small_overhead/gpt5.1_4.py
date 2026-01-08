from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args=None):
        super().__init__(args)
        self.args = args
        self._reset_episode_state()

    def _reset_episode_state(self):
        self.use_on_demand_only = False
        self._last_elapsed = -1.0
        self._last_task_done_list_id = None
        self._last_task_done_len = 0
        self._last_work_done_sum = 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _update_work_done_cache(self) -> float:
        td = self.task_done_time
        if td is None:
            self._last_task_done_list_id = None
            self._last_task_done_len = 0
            self._last_work_done_sum = 0.0
            return 0.0

        curr_id = id(td)
        length = len(td)

        if self._last_task_done_list_id != curr_id:
            # List object replaced; recompute from scratch.
            total = float(sum(td))
            self._last_task_done_list_id = curr_id
            self._last_task_done_len = length
            self._last_work_done_sum = total
            return total

        # Same list object; try incremental update.
        if length > self._last_task_done_len:
            incremental = float(sum(td[self._last_task_done_len:]))
            self._last_work_done_sum += incremental
            self._last_task_done_len = length
        elif length < self._last_task_done_len:
            # Length decreased; fallback to full recompute.
            total = float(sum(td))
            self._last_task_done_len = length
            self._last_work_done_sum = total

        return self._last_work_done_sum

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = self.env
        elapsed = getattr(env, "elapsed_seconds", 0.0)

        # Detect new episode by time reset.
        if elapsed < self._last_elapsed:
            self._reset_episode_state()
        self._last_elapsed = elapsed

        # Compute remaining work.
        work_done = self._update_work_done_cache()
        remaining_work = self.task_duration - work_done

        if remaining_work <= 0:
            # Task is complete; no further work needed.
            return ClusterType.NONE

        if self.use_on_demand_only:
            return ClusterType.ON_DEMAND

        deadline = self.deadline
        remaining_time = deadline - elapsed

        if remaining_time <= 0:
            # Already at or past deadline; use on-demand as best effort.
            self.use_on_demand_only = True
            return ClusterType.ON_DEMAND

        overhead = self.restart_overhead

        gap = getattr(env, "gap_seconds", 0.0)
        if gap <= 0:
            gap = 60.0  # Fallback to 1 minute if unspecified.

        # Slack between available time and worst-case required time (remaining work + one restart).
        safe_margin = remaining_time - (remaining_work + overhead)

        # When slack becomes as small as one step, switch permanently to on-demand.
        if safe_margin <= gap:
            self.use_on_demand_only = True
            return ClusterType.ON_DEMAND

        # In the slackful region, prefer spot when available.
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable and we still have sufficient slack: wait.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)