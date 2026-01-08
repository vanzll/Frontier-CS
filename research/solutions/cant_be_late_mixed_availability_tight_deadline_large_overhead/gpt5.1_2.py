from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_safe_spot"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _init_run_state(self):
        self._run_initialized = True
        self.commit_to_od = False

        task_done = getattr(self, "task_done_time", None)
        if task_done:
            self._task_done_sum = float(sum(task_done))
            self._last_task_done_len = len(task_done)
        else:
            self._task_done_sum = 0.0
            self._last_task_done_len = 0

        self.initial_slack = float(self.deadline - self.task_duration)
        if self.initial_slack < 0.0:
            self.initial_slack = 0.0

        safety_factor = 1.0
        self.safe_slack_threshold = float(self.restart_overhead * safety_factor)
        if self.safe_slack_threshold < self.restart_overhead:
            self.safe_slack_threshold = float(self.restart_overhead)

        self._last_elapsed = getattr(self.env, "elapsed_seconds", 0.0)
        self.preempt_count = 0

    def _ensure_run_state(self):
        elapsed = getattr(self.env, "elapsed_seconds", 0.0)
        if (
            not hasattr(self, "_run_initialized")
            or not self._run_initialized
            or elapsed < getattr(self, "_last_elapsed", -1.0)
        ):
            self._init_run_state()
        self._last_elapsed = elapsed

    def _update_task_done_sum(self):
        task_done = getattr(self, "task_done_time", None)
        if not task_done:
            return
        cur_len = len(task_done)
        last_len = getattr(self, "_last_task_done_len", 0)
        if cur_len > last_len:
            new_segments = task_done[last_len:cur_len]
            self._task_done_sum += float(sum(new_segments))
            self._last_task_done_len = cur_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._ensure_run_state()

        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self.preempt_count += 1

        self._update_task_done_sum()

        remaining_work = self.task_duration - self._task_done_sum
        if remaining_work <= 0.0:
            return ClusterType.NONE

        elapsed = self.env.elapsed_seconds
        t_finish_od = elapsed + remaining_work
        slack = self.deadline - t_finish_od

        if getattr(self, "commit_to_od", False):
            return ClusterType.ON_DEMAND

        if slack <= self.safe_slack_threshold:
            self.commit_to_od = True
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)