from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args=None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except TypeError:
                pass
        self.args = args
        self._policy_initialized = False
        self._done_work = 0.0
        self._last_task_done_len = 0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _init_policy(self):
        if self._policy_initialized:
            return

        deadline = float(getattr(self, "deadline", 0.0))
        task_duration = float(getattr(self, "task_duration", 0.0))
        restart_overhead = float(getattr(self, "restart_overhead", 0.0))
        gap = float(getattr(getattr(self, "env", None), "gap_seconds", 0.0))

        # Effective slack if we assume we may need one restart in the future.
        eff_slack0 = max(0.0, deadline - task_duration - restart_overhead)
        self._eff_initial_slack = eff_slack0

        if eff_slack0 <= 0.0:
            # Essentially no slack: avoid idling and be very conservative with spot.
            self._spot_min_slack = restart_overhead
            self._idle_slack_threshold = 0.0
        else:
            # Minimum slack to still allow using spot.
            # At least 1.5x restart_overhead and at least 10% of initial effective slack.
            self._spot_min_slack = max(restart_overhead * 1.5, 0.1 * eff_slack0)
            if self._spot_min_slack > eff_slack0:
                self._spot_min_slack = eff_slack0 * 0.5

            # Idle only when slack is comfortably above spot_min_slack.
            self._idle_slack_threshold = self._spot_min_slack + 2.0 * gap
            if self._idle_slack_threshold > eff_slack0:
                self._idle_slack_threshold = eff_slack0 * 0.8
            if self._idle_slack_threshold < self._spot_min_slack:
                self._idle_slack_threshold = self._spot_min_slack

        self._policy_initialized = True

    def _update_done_work(self):
        segments = getattr(self, "task_done_time", None)
        if segments is None:
            return
        ln = len(segments)
        if ln <= self._last_task_done_len:
            return

        for seg in segments[self._last_task_done_len:]:
            add = 0.0
            if isinstance(seg, (int, float)):
                add = float(seg)
            else:
                try:
                    # Assume sequence-like [start, end]
                    s, e = seg[0], seg[1]
                    add = float(e) - float(s)
                except Exception:
                    # Fallback to attributes .start and .end
                    s = getattr(seg, "start", None)
                    e = getattr(seg, "end", None)
                    if s is not None and e is not None:
                        add = float(e) - float(s)
            if add > 0.0:
                self._done_work += add

        self._last_task_done_len = ln

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._policy_initialized:
            self._init_policy()

        self._update_done_work()

        task_duration = float(getattr(self, "task_duration", 0.0))
        deadline = float(getattr(self, "deadline", 0.0))
        restart_overhead = float(getattr(self, "restart_overhead", 0.0))
        env = getattr(self, "env", None)
        now = float(getattr(env, "elapsed_seconds", 0.0))
        gap = float(getattr(env, "gap_seconds", 0.0))

        remaining = task_duration - self._done_work
        if remaining <= 0.0:
            # Task finished; no need to run more.
            return ClusterType.NONE

        time_left = deadline - now
        if time_left <= 0.0:
            # Past deadline: run on-demand to finish as much as possible.
            return ClusterType.ON_DEMAND

        # Effective time left if we still need to pay one restart overhead.
        eff_time_left = time_left - restart_overhead
        if eff_time_left < 0.0:
            eff_time_left = 0.0

        # Slack in seconds: how much non-productive time we can still afford.
        slack_raw = eff_time_left - remaining

        # If slack_raw <= 0, we cannot afford any more waste; must run on-demand.
        if slack_raw <= 0.0:
            return ClusterType.ON_DEMAND

        # Ensure policy parameters initialized.
        eff_slack0 = getattr(self, "_eff_initial_slack", 0.0)
        if eff_slack0 <= 0.0:
            # Extremely tight schedule: always prefer on-demand when available.
            if has_spot:
                # Use spot only if absolutely necessary; here we still allow it,
                # but prefer on-demand for reliability.
                return ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND

        spot_min_slack = self._spot_min_slack
        idle_slack_threshold = self._idle_slack_threshold

        # Late phase: slack too low to risk spot preemptions.
        if slack_raw <= spot_min_slack:
            return ClusterType.ON_DEMAND

        # Middle/early phase.
        if has_spot:
            # Plenty of slack and spot available: use spot.
            return ClusterType.SPOT

        # No spot available. Decide between idling and on-demand.
        # Idling for one step reduces slack_raw by gap.
        # Only idle if we'll still be comfortably above the idle_slack_threshold.
        if slack_raw - gap >= idle_slack_threshold and gap > 0.0:
            return ClusterType.NONE

        # Otherwise, use on-demand to preserve remaining slack.
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)