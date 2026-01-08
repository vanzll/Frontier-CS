from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        # Initialize internal state
        self.committed_to_od = False
        self.policy_initialized = False
        self.initial_slack = None
        self.wait_slack_seconds = None
        self.commit_slack_seconds = None
        self.last_slack = None
        return self

    def _init_policy_if_needed(self):
        if getattr(self, "policy_initialized", False):
            return

        # Guard against missing attributes; in that case fall back to trivial policy later.
        try:
            total_slack = self.deadline - (self.task_duration + self.restart_overhead)
        except Exception:
            total_slack = 0.0

        if total_slack <= 0:
            # No slack: essentially must use on-demand all the way.
            self.initial_slack = 0.0
            self.commit_slack_seconds = 0.0
            self.wait_slack_seconds = 0.0
        else:
            self.initial_slack = total_slack

            # Fractions of initial slack to determine phases
            commit_frac = 0.15  # slack to keep when committing to full on-demand
            wait_frac = 0.5     # slack level separating wait-only vs hybrid

            commit_slack = total_slack * commit_frac
            # Ensure some minimal safety: at least a couple of restart_overheads
            try:
                oh = float(self.restart_overhead)
            except Exception:
                oh = 0.0
            min_commit = 2.0 * oh
            if commit_slack < min_commit:
                commit_slack = min_commit

            wait_slack = total_slack * wait_frac
            # Ensure wait_slack > commit_slack with some margin
            if wait_slack <= commit_slack:
                wait_slack = commit_slack + 0.1 * total_slack

            # Cap wait_slack to leave some slack for later stages
            max_wait = 0.9 * total_slack
            if wait_slack > max_wait:
                wait_slack = max_wait

            self.commit_slack_seconds = commit_slack
            self.wait_slack_seconds = wait_slack

        self.policy_initialized = True

    def _estimate_progress_seconds(self) -> float:
        # Default progress estimate: each completed work segment is one gap-sized chunk.
        try:
            gap = float(self.env.gap_seconds)
        except Exception:
            gap = 0.0
        try:
            done_segments = len(self.task_done_time)
        except Exception:
            done_segments = 0

        progress = done_segments * gap

        try:
            total = float(self.task_duration)
            if progress > total:
                progress = total
        except Exception:
            pass

        return progress

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Fallback if environment is not attached yet: simple spot-first policy.
        if not hasattr(self, "env") or self.env is None:
            if has_spot:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # Ensure internal state exists
        if not hasattr(self, "committed_to_od"):
            self.committed_to_od = False
        if not hasattr(self, "policy_initialized"):
            self.policy_initialized = False

        self._init_policy_if_needed()

        # If already committed to on-demand, always choose on-demand.
        if self.committed_to_od:
            return ClusterType.ON_DEMAND

        # If we have no valid timing/task info, fall back to simple spot-first policy.
        try:
            current_time = float(self.env.elapsed_seconds)
            deadline = float(self.deadline)
            total_duration = float(self.task_duration)
            restart_overhead = float(self.restart_overhead)
        except Exception:
            if has_spot:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        progress = self._estimate_progress_seconds()
        remaining_compute = max(total_duration - progress, 0.0)

        # Slack if we switch to on-demand now (including a single restart overhead)
        slack = deadline - (current_time + remaining_compute + restart_overhead)
        self.last_slack = slack

        # If even immediate full on-demand can't meet the deadline, still choose on-demand.
        if slack <= 0.0:
            self.committed_to_od = True
            return ClusterType.ON_DEMAND

        # Commit phase: from now on, always use on-demand to guarantee completion.
        if slack <= self.commit_slack_seconds:
            self.committed_to_od = True
            return ClusterType.ON_DEMAND

        # High-slack phase: use spot when available, otherwise wait (NONE) to save cost.
        if slack > self.wait_slack_seconds:
            if has_spot:
                return ClusterType.SPOT
            return ClusterType.NONE

        # Medium-slack (hybrid) phase: use spot opportunistically, but fall back to
        # on-demand when spot is unavailable to avoid burning too much slack on idle.
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)