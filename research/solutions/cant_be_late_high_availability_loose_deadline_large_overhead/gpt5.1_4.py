from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_heuristic"

    def solve(self, spec_path: str) -> "Solution":
        # Optional: load configuration from spec_path if needed.
        return self

    def _initialize_if_needed(self):
        env = self.env
        elapsed = getattr(env, "elapsed_seconds", 0.0)

        if not hasattr(self, "_initialized") or not getattr(self, "_initialized"):
            # First call overall.
            self._initialized = True
            self._last_seen_elapsed = elapsed
            self._reset_run_state()
        else:
            # Detect a new run by elapsed time reset.
            if elapsed < self._last_seen_elapsed:
                self._reset_run_state()
            self._last_seen_elapsed = elapsed

    def _reset_run_state(self):
        # Per-run state
        self._committed_to_on_demand = False
        self._switch_to_on_demand_time = None

        # Cache for computing progress efficiently
        self._task_done_cached_index = 0
        self._task_done_cached_total = 0.0

        # Precompute thresholds using current environment parameters.
        try:
            gap = float(self.env.gap_seconds)
        except Exception:
            gap = 60.0

        try:
            task_duration = float(self.task_duration)
        except Exception:
            task_duration = 0.0

        try:
            deadline = float(self.deadline)
        except Exception:
            deadline = 0.0

        try:
            restart_overhead = float(self.restart_overhead)
        except Exception:
            restart_overhead = 0.0

        initial_slack = max(deadline - (task_duration + restart_overhead), 0.0)
        self._initial_slack = initial_slack

        if initial_slack <= 0.0:
            # No slack: must be very conservative; effectively always on-demand.
            self._commit_slack_threshold = 0.0
            self._wait_slack_threshold = 0.0
        else:
            # Commit threshold: when remaining slack shrinks to ~5% of initial or at least one gap.
            commit_slack = max(0.05 * initial_slack, gap)
            # Avoid committing too early on small-slack scenarios.
            if commit_slack > initial_slack * 0.5:
                commit_slack = max(initial_slack * 0.5, gap)
            self._commit_slack_threshold = commit_slack

            # Waiting threshold: below this slack, we stop pure waiting and start using on-demand
            # when spot is unavailable. Use about 30% of slack, at least 3x commit.
            wait_slack = max(0.3 * initial_slack, 3.0 * commit_slack)
            if wait_slack > initial_slack:
                wait_slack = initial_slack
            if wait_slack < commit_slack:
                wait_slack = commit_slack
            self._wait_slack_threshold = wait_slack

    def _update_task_done_cache(self) -> float:
        """Incrementally compute total completed work duration in seconds."""
        total = getattr(self, "_task_done_cached_total", 0.0)
        idx = getattr(self, "_task_done_cached_index", 0)
        segments = self.task_done_time
        n = len(segments)

        while idx < n:
            seg = segments[idx]
            if isinstance(seg, (int, float)):
                total += float(seg)
            else:
                # Try interpreting as (start, end) segment.
                try:
                    start = seg[0]
                    end = seg[1]
                    total += float(end - start)
                except Exception:
                    # Fallback: attempt to cast directly.
                    try:
                        total += float(seg)
                    except Exception:
                        # If cannot interpret, skip this segment.
                        pass
            idx += 1

        self._task_done_cached_total = total
        self._task_done_cached_index = idx
        return total

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Ensure per-run state and thresholds are initialized.
        self._initialize_if_needed()

        # If we've already committed to on-demand, never go back.
        if getattr(self, "_committed_to_on_demand", False):
            return ClusterType.ON_DEMAND

        env = self.env
        elapsed = env.elapsed_seconds
        time_left = max(self.deadline - elapsed, 0.0)

        # Compute progress done so far.
        work_done = self._update_task_done_cache()
        work_left = max(self.task_duration - work_done, 0.0)

        # If the task is already complete, avoid incurring any more cost.
        if work_left <= 0.0:
            return ClusterType.NONE

        restart_overhead = self.restart_overhead
        # Remaining slack beyond minimum required time if we switched to on-demand now.
        slack_remaining = time_left - (work_left + restart_overhead)

        commit_threshold = getattr(self, "_commit_slack_threshold", 0.0)
        wait_threshold = getattr(self, "_wait_slack_threshold", 0.0)

        # If slack is critically low, permanently commit to on-demand.
        if slack_remaining <= commit_threshold:
            self._committed_to_on_demand = True
            self._switch_to_on_demand_time = elapsed
            return ClusterType.ON_DEMAND

        # Flexible phase: we can still leverage spot while being careful with slack.

        if has_spot:
            # Normally prefer spot when available.
            # However, if we're already on on-demand and slack is close to commit threshold,
            # it's safer (and often cheaper overall) to avoid flapping back to spot, which
            # could introduce another restart overhead.
            near_commit = commit_threshold > 0.0 and slack_remaining <= 2.0 * commit_threshold
            if near_commit and last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        # Spot is unavailable: decide whether to wait or to use on-demand.
        if slack_remaining > wait_threshold:
            # Plenty of slack left: we can afford to wait for cheap spot instances.
            return ClusterType.NONE

        # Slack is getting tighter: run on on-demand to keep progress on track.
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)