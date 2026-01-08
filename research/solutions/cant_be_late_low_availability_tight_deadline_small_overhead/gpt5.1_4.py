from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cb_late_safe_spot"

    def __init__(self, args=None):
        super().__init__(args)
        self.locked_to_od = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    def reset(self, env):
        super().reset(env)
        self.locked_to_od = False

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Get environment parameters with safe fallbacks
        elapsed = getattr(self.env, "elapsed_seconds", 0.0)
        gap = getattr(self.env, "gap_seconds", 0.0) or 0.0
        deadline = getattr(self, "deadline", None)
        task_duration = getattr(self, "task_duration", None)
        restart_overhead = getattr(self, "restart_overhead", 0.0)

        # If critical attributes are missing, default to safest behavior
        if deadline is None or task_duration is None:
            return ClusterType.ON_DEMAND

        time_left = deadline - elapsed
        if time_left <= 0:
            # At or past deadline: always run on-demand to minimize further delay
            self.locked_to_od = True
            return ClusterType.ON_DEMAND

        # Worst-case continuous on-demand time needed from the moment we commit
        min_needed = task_duration + restart_overhead

        # If we've already committed, or we no longer have enough slack to waste another step,
        # stay on on-demand for the rest of the run.
        if self.locked_to_od or time_left < (min_needed + gap):
            self.locked_to_od = True
            return ClusterType.ON_DEMAND

        # Flexible phase: use spot when available, otherwise pause to save cost
        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)