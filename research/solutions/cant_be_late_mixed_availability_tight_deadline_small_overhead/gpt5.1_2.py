from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_heuristic"

    def solve(self, spec_path: str) -> "Solution":
        # No special initialization needed for now
        return self

    def _ensure_env_state(self):
        """Initialize or reset per-environment state."""
        env_id = id(self.env)
        if getattr(self, "_env_id", None) == env_id:
            return

        # New environment detected (or first call)
        self._env_id = env_id

        # Cached work-done tracking
        self._cache_work_done = 0.0
        self._cache_task_done_len = 0

        # Commitment flag: once True, always use on-demand
        self._committed_to_od = False

        # Precompute slack-related parameters
        self._task_duration = float(self.task_duration)
        self._deadline = float(self.deadline)
        self._restart_overhead = float(self.restart_overhead)
        self._gap = float(self.env.gap_seconds)

        self._slack_total = self._deadline - self._task_duration  # seconds

        if self._slack_total > 0:
            # Base floor: keep at least 10% of total slack as a safety buffer
            base_floor = 0.10 * self._slack_total
            # Ensure floor is also large enough to cover one overhead plus a bit more than one time step
            overhead_based_floor = self._restart_overhead + 1.1 * self._gap
            self._commit_slack_floor = max(base_floor, overhead_based_floor)
        else:
            # No slack: effectively always on-demand
            self._commit_slack_floor = 0.0

    def _update_work_done_cache(self):
        """Incrementally track total work done to avoid O(n) per step."""
        task_list = self.task_done_time
        n = len(task_list)

        # If list length decreased (env reset without _ensure_env_state catching),
        # fall back to full recomputation.
        if n < self._cache_task_done_len:
            self._cache_work_done = float(sum(task_list))
            self._cache_task_done_len = n
            return

        if n > self._cache_task_done_len:
            # Sum only new segments
            self._cache_work_done += float(sum(task_list[self._cache_task_done_len:n]))
            self._cache_task_done_len = n

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Ensure per-environment state is initialized
        self._ensure_env_state()

        # Update cached amount of work completed
        self._update_work_done_cache()
        work_done = self._cache_work_done

        remaining_work = max(0.0, self._task_duration - work_done)

        # If job is effectively done, do nothing to avoid extra cost
        if remaining_work <= 0.0:
            return ClusterType.NONE

        elapsed = float(self.env.elapsed_seconds)
        remaining_time = max(0.0, self._deadline - elapsed)

        # Slack if from now on we run only on-demand
        slack_if_od_now = remaining_time - remaining_work

        # If we are already committed to on-demand, always use it
        if self._committed_to_od:
            return ClusterType.ON_DEMAND

        # Decide if we must now commit to on-demand to safely finish
        if slack_if_od_now <= self._commit_slack_floor:
            # Once committed, never go back to spot
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        # Not yet committed: use spot whenever available
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable and not yet committed:
        # Decide between waiting (NONE) and running on-demand.
        # Waiting for one more step consumes `gap` seconds of slack (no progress).
        # Only wait if after waiting we still have at least the commit floor of slack.
        gap = self._gap
        if slack_if_od_now - gap >= self._commit_slack_floor:
            return ClusterType.NONE

        # Can't safely wait any longer; use on-demand this step.
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)