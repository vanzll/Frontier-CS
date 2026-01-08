from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_safe_spot_v1"

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        """
        # Lazy initialization in _step to avoid relying on solve() being called.
        return self

    def _ensure_internal_state(self):
        """Initialize internal state if not already done."""
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self._commit_time = None
        self._force_on_demand = False

    def _ensure_commit_time(self):
        """Compute the latest time we can risk not running on-demand."""
        if self._commit_time is not None:
            return

        deadline = getattr(self, "deadline", None)
        task_duration = getattr(self, "task_duration", None)
        restart_overhead = getattr(self, "restart_overhead", None)
        gap = getattr(self.env, "gap_seconds", None)

        # If any key parameter is missing, fall back to always-on on-demand
        if (
            deadline is None
            or task_duration is None
            or restart_overhead is None
            or gap is None
        ):
            self._commit_time = 0.0
            self._force_on_demand = True
            return

        # Safety buffer to account for discrete time steps:
        # we must ensure that after the *next* step, there's still
        # enough time to finish entirely on on-demand, including one restart.
        safety_buffer = gap

        commit_time = deadline - (task_duration + restart_overhead + safety_buffer)

        if commit_time <= 0:
            # No slack: always use on-demand from the start.
            self._commit_time = 0.0
            self._force_on_demand = True
        else:
            self._commit_time = commit_time

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide which cluster type to use at each time step.
        """
        self._ensure_internal_state()
        self._ensure_commit_time()

        current_time = getattr(self.env, "elapsed_seconds", 0.0)

        # Once we reach the commit time, we must switch to on-demand and never go back
        if not self._force_on_demand and current_time >= self._commit_time:
            self._force_on_demand = True

        if self._force_on_demand:
            return ClusterType.ON_DEMAND

        # Before commit time: opportunistically use spot, otherwise pause.
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)