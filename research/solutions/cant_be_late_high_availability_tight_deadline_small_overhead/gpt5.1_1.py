from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_solution"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = "SPOT"  # Modes: "SPOT" or "ON_DEMAND"
        self._initialized_slack = False
        self._initial_slack = 0.0
        self._commit_slack = 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _initialize_slack_params(self):
        if self._initialized_slack:
            return
        try:
            deadline = float(self.deadline)
            task_duration = float(self.task_duration)
        except Exception:
            self._initial_slack = 0.0
            self._commit_slack = 0.0
            self._initialized_slack = True
            return

        self._initial_slack = max(deadline - task_duration, 0.0)
        gap = float(getattr(self.env, "gap_seconds", 1.0) or 1.0)
        overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        if self._initial_slack <= 0.0:
            self._commit_slack = 0.0
        else:
            base_from_fraction = 0.1 * self._initial_slack  # 10% of initial slack
            base_from_overhead = overhead + 2.0 * gap        # overhead + discretization margin
            base = max(base_from_fraction, base_from_overhead, 0.0)
            max_allowed = 0.9 * self._initial_slack          # do not commit immediately at start
            self._commit_slack = min(base, max_allowed)

        if self._commit_slack < 0.0:
            self._commit_slack = 0.0

        self._initialized_slack = True

    def _compute_completed_work(self) -> float:
        segments = getattr(self, "task_done_time", None)
        if segments is None:
            return 0.0

        # If segments is not sized (e.g., a single float)
        try:
            n = len(segments)
        except TypeError:
            try:
                return float(segments)
            except Exception:
                return 0.0

        if n == 0:
            return 0.0

        total = 0.0
        for seg in segments:
            if isinstance(seg, (int, float)):
                total += float(seg)
            elif isinstance(seg, (list, tuple)):
                if len(seg) >= 2:
                    try:
                        total += float(seg[1]) - float(seg[0])
                    except Exception:
                        # Fallback: sum elements if possible
                        try:
                            for v in seg:
                                total += float(v)
                        except Exception:
                            continue
                elif len(seg) == 1:
                    try:
                        total += float(seg[0])
                    except Exception:
                        continue
            else:
                try:
                    total += float(seg)
                except Exception:
                    continue

        if total < 0.0:
            total = 0.0
        return total

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._initialize_slack_params()

        # Remaining work
        done = self._compute_completed_work()
        try:
            task_duration = float(self.task_duration)
        except Exception:
            task_duration = 0.0
        remaining = max(task_duration - done, 0.0)

        if remaining <= 0.0:
            return ClusterType.NONE

        # Time left until deadline
        try:
            elapsed = float(self.env.elapsed_seconds)
        except Exception:
            elapsed = 0.0
        try:
            deadline = float(self.deadline)
        except Exception:
            deadline = elapsed
        time_left = max(deadline - elapsed, 0.0)

        slack = time_left - remaining

        # If we've already committed to on-demand, keep using it
        if self._mode == "ON_DEMAND":
            return ClusterType.ON_DEMAND

        gap = float(getattr(self.env, "gap_seconds", 1.0) or 1.0)
        overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        commit_now = False

        # Primary rule: commit when slack is below threshold
        if self._initial_slack > 0.0 and slack <= self._commit_slack:
            commit_now = True

        # Safety rule: if time left is barely enough for OD (including one restart overhead), commit
        worst_needed = remaining + overhead
        if time_left <= worst_needed + gap:
            commit_now = True

        # If out of time but work remains, force OD (though it's likely already too late)
        if time_left <= 0.0 and remaining > 0.0:
            commit_now = True

        if commit_now:
            self._mode = "ON_DEMAND"
            return ClusterType.ON_DEMAND

        # Still in SPOT-focused mode
        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)