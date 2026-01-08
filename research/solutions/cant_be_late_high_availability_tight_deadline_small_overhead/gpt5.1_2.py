from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_heuristic_v1"

    def __init__(self, args=None):
        super().__init__(args)
        self._initialized = False
        self._last_env_elapsed = 0.0
        self._last_task_done_len = 0
        self._total_done = 0.0
        # Safety factor for deadline buffer
        self._safety_factor = 2.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _reset_progress_tracking_if_needed(self):
        # Detect new episode by elapsed time going backwards or first call
        current_elapsed = getattr(self.env, "elapsed_seconds", 0.0)
        if (not self._initialized) or current_elapsed < self._last_env_elapsed:
            self._initialized = True
            self._last_task_done_len = 0
            self._total_done = 0.0
        self._last_env_elapsed = current_elapsed

    def _estimate_total_done(self) -> float:
        # Incremental summation of task_done_time assuming each entry is a segment duration.
        td_list = getattr(self, "task_done_time", None)
        if not td_list:
            return self._total_done

        current_len = len(td_list)
        if current_len < self._last_task_done_len:
            # New episode detected via list shrink; reset.
            self._last_task_done_len = 0
            self._total_done = 0.0

        if current_len > self._last_task_done_len:
            delta_sum = 0.0
            for i in range(self._last_task_done_len, current_len):
                delta_sum += td_list[i]
            self._total_done += delta_sum
            self._last_task_done_len = current_len

        return self._total_done

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Reset per-episode tracking if needed
        self._reset_progress_tracking_if_needed()

        # Estimate how much work has been completed
        total_done = self._estimate_total_done()
        remaining = max(self.task_duration - total_done, 0.0)

        # If we believe the task is done, don't run anything
        if remaining <= 0.0:
            return ClusterType.NONE

        elapsed = self.env.elapsed_seconds
        time_left = self.deadline - elapsed
        if time_left <= 0.0:
            # Already at or past deadline; still try to run on-demand
            return ClusterType.ON_DEMAND

        gap = getattr(self.env, "gap_seconds", 0.0)
        restart_overhead = getattr(self, "restart_overhead", 0.0)

        # Overhead to switch and then stick with on-demand
        if self.env.cluster_type == ClusterType.ON_DEMAND:
            overhead_to_od = 0.0
        else:
            overhead_to_od = restart_overhead

        # Safety buffer: multiple of step size / restart overhead
        base_scale = max(restart_overhead, gap)
        safety_buffer = self._safety_factor * base_scale

        # Slack if we immediately commit to on-demand
        slack_if_switch_now = time_left - overhead_to_od - remaining

        # Worst-case slack after wasting one decision step (no useful progress)
        worst_case_future_slack = slack_if_switch_now - gap

        # If after one wasted step we would not have enough buffer, commit to OD now
        if worst_case_future_slack <= safety_buffer:
            return ClusterType.ON_DEMAND

        # Otherwise, we are in a "safe" zone: can use Spot/idle strategically

        if has_spot:
            # Prefer Spot when available while we still have ample slack
            return ClusterType.SPOT

        # No spot available: decide between idling and on-demand
        # If we still have comfortable slack, we can afford to wait for Spot
        if slack_if_switch_now > safety_buffer + 2.0 * gap:
            return ClusterType.NONE

        # Slack is shrinking; start using on-demand to maintain schedule
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)