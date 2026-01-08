from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "safe_spot_then_on_demand"

    def solve(self, spec_path: str) -> "Solution":
        self._force_on_demand = False
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Ensure flag exists even if solve() was not called for some reason
        if not hasattr(self, "_force_on_demand"):
            self._force_on_demand = False

        t = self.env.elapsed_seconds
        dt = self.env.gap_seconds
        # Conservative worst-case: assume 0 work has been done so far.
        # Only allow "gambling" (SPOT or NONE) this step if, even after
        # waiting dt more seconds, we can still finish the full task on OD
        # plus a single restart overhead before the deadline.
        if not self._force_on_demand:
            if t + dt + self.task_duration + self.restart_overhead > self.deadline:
                self._force_on_demand = True

        if self._force_on_demand:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)