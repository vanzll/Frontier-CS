from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "gpt_balanced_spot_od_v1"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _initialize(self):
        self._init_done = True
        self.committed_to_on_demand = False

        # Progress tracking
        self._last_task_done_len = 0
        self._accumulated_work_done = 0.0

        # Environment parameters
        self.gap = float(getattr(self.env, "gap_seconds", 0.0))
        self.restart = float(getattr(self, "restart_overhead", 0.0))

        if self.gap < 0.0:
            self.gap = 0.0
        if self.restart < 0.0:
            self.restart = 0.0

        base_slack = float(self.deadline - self.task_duration)
        if base_slack < 0.0:
            base_slack = 0.0
        self.base_slack = base_slack

        if base_slack > 0.0:
            commit_frac = 0.25
            idle_frac = 0.5

            commit_floor = commit_frac * base_slack
            idle_floor = idle_frac * base_slack

            # Ensure enough buffer for at least a couple of restarts
            min_commit = 2.0 * self.restart
            if commit_floor < min_commit:
                commit_floor = min_commit

            # Idle floor should be above commit floor by at least one restart
            min_idle = commit_floor + self.restart
            if idle_floor < min_idle:
                idle_floor = min_idle

            # Clamp thresholds to available slack
            if idle_floor > base_slack:
                idle_floor = base_slack
            if commit_floor > idle_floor:
                commit_floor = idle_floor
        else:
            commit_floor = 0.0
            idle_floor = 0.0

        self.commit_slack_floor = commit_floor
        self.idle_slack_floor = idle_floor

        # Small epsilon to avoid oscillations around thresholds
        if self.gap > 0.0:
            self.epsilon = 0.5 * self.gap
        else:
            self.epsilon = 1e-6

    def _update_progress(self):
        lst = self.task_done_time
        if not lst:
            return
        start_idx = self._last_task_done_len
        if start_idx >= len(lst):
            return
        for i in range(start_idx, len(lst)):
            seg = lst[i]
            dur = 0.0
            if isinstance(seg, (list, tuple)):
                if len(seg) >= 2:
                    try:
                        dur = float(seg[1]) - float(seg[0])
                    except (TypeError, ValueError):
                        dur = 0.0
                elif len(seg) == 1:
                    try:
                        dur = float(seg[0])
                    except (TypeError, ValueError):
                        dur = 0.0
            else:
                try:
                    dur = float(seg)
                except (TypeError, ValueError):
                    dur = 0.0
            if dur > 0.0:
                self._accumulated_work_done += dur
        self._last_task_done_len = len(lst)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not hasattr(self, "_init_done"):
            self._initialize()

        # Update progress accounting
        self._update_progress()

        # Remaining work and time
        work_done = self._accumulated_work_done
        if work_done < 0.0:
            work_done = 0.0
        if work_done > self.task_duration:
            work_done = self.task_duration

        remaining_work = self.task_duration - work_done
        remaining_time = self.deadline - float(self.env.elapsed_seconds)

        # If already done or no time left, do nothing
        if remaining_work <= 0.0 or remaining_time <= 0.0:
            return ClusterType.NONE

        slack = remaining_time - remaining_work

        # Decide if we must commit to on-demand
        if not self.committed_to_on_demand:
            if slack <= self.commit_slack_floor:
                self.committed_to_on_demand = True

        if self.committed_to_on_demand:
            return ClusterType.ON_DEMAND

        # Before commitment: use spot whenever available
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable: decide between waiting and on-demand
        slack_after_idle = slack - self.gap

        # Wait if we can afford to burn this gap and stay above idle slack floor
        if slack_after_idle >= self.idle_slack_floor + self.epsilon:
            return ClusterType.NONE

        # Otherwise, use on-demand to preserve schedule
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)