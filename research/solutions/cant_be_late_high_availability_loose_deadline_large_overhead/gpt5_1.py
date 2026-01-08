from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_threshold"

    def __init__(self, args=None):
        super().__init__(args)
        self._od_locked = False
        self._locked_reason = ""
        self._switch_to_od_time = None
        self._margin_seconds_static = None

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _compute_done(self):
        dt = getattr(self, "task_done_time", None)
        if dt is None:
            return 0.0
        if isinstance(dt, (list, tuple)):
            try:
                return float(sum(dt))
            except TypeError:
                total = 0.0
                for v in dt:
                    try:
                        total += float(v)
                    except Exception:
                        continue
                return total
        try:
            return float(dt)
        except Exception:
            return 0.0

    def _margin_seconds(self):
        m = self._margin_seconds_static
        if m is not None:
            return m
        gap = getattr(self.env, "gap_seconds", 300.0) or 300.0
        restart = getattr(self, "restart_overhead", getattr(self.env, "restart_overhead", 0.0))
        restart = 0.0 if restart is None else float(restart)
        base = max(float(gap), 300.0)
        m = base + 0.25 * restart
        return m

    def _lock_on_demand(self, reason: str = ""):
        self._od_locked = True
        self._locked_reason = reason
        try:
            self._switch_to_od_time = float(self.env.elapsed_seconds)
        except Exception:
            self._switch_to_od_time = None

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        done = self._compute_done()
        task_duration = float(getattr(self, "task_duration", 0.0))
        remaining = max(0.0, task_duration - done)
        if remaining <= 0.0:
            return ClusterType.NONE

        if self._od_locked:
            return ClusterType.ON_DEMAND

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        deadline = float(getattr(self, "deadline", getattr(self.env, "deadline", 0.0)))
        gap = float(getattr(self.env, "gap_seconds", 300.0))
        overhead = float(getattr(self, "restart_overhead", getattr(self.env, "restart_overhead", 0.0)))
        T_rem = max(0.0, deadline - elapsed)

        margin = float(self._margin_seconds())

        def must_use_od_now():
            return T_rem <= (remaining + overhead + margin)

        def can_wait_one_step():
            return (T_rem - gap) > (remaining + overhead + margin)

        if must_use_od_now():
            self._lock_on_demand("deadline_guard")
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        if can_wait_one_step():
            return ClusterType.NONE
        else:
            self._lock_on_demand("spot_unavailable_no_slack")
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)