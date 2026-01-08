from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args=None):
        super().__init__(args)
        self._initialized_run_params = False
        self._slack_total = 0.0
        self._commit_threshold = 0.0
        self._restart_overhead = 0.0
        self._force_on_demand = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _init_run_params_if_needed(self):
        if self._initialized_run_params:
            return
        # These attributes are guaranteed by the problem description to exist in _step.
        task_duration = float(self.task_duration)
        deadline = float(self.deadline)
        self._restart_overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        self._slack_total = max(0.0, deadline - task_duration)

        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)

        # Minimal safe threshold accounts for at most one full-step of extra loss plus one restart.
        min_safe_threshold = self._restart_overhead + gap
        half_slack = 0.5 * self._slack_total

        # Commit threshold: be conservative in high-loss regimes but cheap when traces are good.
        # - At least min_safe_threshold for safety against discretization.
        # - At least 3 * restart_overhead for slack beyond one restart.
        # - At least half of total slack so we never consume more than ~half slack to uncertainty.
        thr = max(min_safe_threshold, 3.0 * self._restart_overhead, half_slack)
        if thr > self._slack_total:
            thr = self._slack_total
        self._commit_threshold = thr
        self._initialized_run_params = True

    def _compute_progress(self) -> float:
        segments = getattr(self, "task_done_time", None)
        if not segments:
            return 0.0
        total = 0.0
        try:
            first = segments[0]
        except Exception:
            return 0.0

        if isinstance(first, (tuple, list)) and len(first) >= 2:
            # Interpret as (start, end) segments.
            for seg in segments:
                try:
                    start, end = seg[0], seg[1]
                    total += float(end) - float(start)
                except Exception:
                    continue
        else:
            # Interpret as durations; fall back to (start, end) if needed.
            for seg in segments:
                try:
                    total += float(seg)
                except Exception:
                    try:
                        start, end = seg[0], seg[1]
                        total += float(end) - float(start)
                    except Exception:
                        continue

        try:
            td = float(self.task_duration)
            if total > td:
                total = td
        except Exception:
            pass

        if total < 0.0:
            total = 0.0
        return total

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._init_run_params_if_needed()

        progress = self._compute_progress()
        task_duration = float(self.task_duration)
        remaining_work = max(0.0, task_duration - progress)

        if remaining_work <= 0.0:
            return ClusterType.NONE

        elapsed = float(self.env.elapsed_seconds)
        deadline = float(self.deadline)
        remaining_time = max(0.0, deadline - elapsed)

        if remaining_time <= 0.0:
            # Past deadline; still try to finish as fast as possible.
            return ClusterType.ON_DEMAND

        # Remaining slack if we made zero further losses from now on.
        slack_remaining = remaining_time - remaining_work

        # Once we decide to stick with on-demand, never go back to spot.
        if self._force_on_demand:
            return ClusterType.ON_DEMAND

        # If slack remaining is small, switch permanently to on-demand to avoid future losses.
        if slack_remaining <= self._commit_threshold:
            self._force_on_demand = True
            return ClusterType.ON_DEMAND

        # Normal regime: always compute (no intentional idling).
        # Use spot when available, otherwise on-demand.
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)