import argparse

try:
    from sky_spot.strategies.strategy import Strategy
    from sky_spot.utils import ClusterType
except ImportError:  # Fallback definitions for local/testing environments
    from enum import Enum

    class ClusterType(Enum):
        SPOT = "spot"
        ON_DEMAND = "on_demand"
        NONE = "none"

    class Strategy:
        def __init__(self, *args, **kwargs):
            class DummyEnv:
                def __init__(self):
                    self.elapsed_seconds = 0.0
                    self.gap_seconds = 60.0
                    self.cluster_type = ClusterType.NONE

            self.env = DummyEnv()
            self.task_duration = 0.0
            self.task_done_time = []
            self.deadline = 0.0
            self.restart_overhead = 0.0

        def solve(self, spec_path: str):
            return self

        @classmethod
        def _from_args(cls, parser):
            args, _ = parser.parse_known_args()
            return cls(args)


class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args=None):
        super().__init__(args)
        self.args = args

    def solve(self, spec_path: str) -> "Solution":
        # Optional: parse spec_path if needed. Not used in this strategy.
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = self.env

        elapsed = getattr(env, "elapsed_seconds", 0.0)
        gap = getattr(env, "gap_seconds", 1.0)
        deadline = getattr(self, "deadline", 0.0)
        task_duration = getattr(self, "task_duration", 0.0)
        restart_overhead = getattr(self, "restart_overhead", 0.0)

        # Time left until the hard deadline
        time_left = deadline - elapsed

        # If we don't have valid info, fall back to a simple safe policy
        if task_duration <= 0 or deadline <= 0 or time_left <= 0:
            if has_spot:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # Conservative assumption: ignore completed work and assume the full task remains.
        work_remaining_est = task_duration

        # Estimated slack: how much time beyond the (remaining) work we have
        slack_est = time_left - work_remaining_est

        # Safety buffer (in seconds):
        # - at least the restart overhead
        # - scaled by factors to stay clear of the deadline relative to step size
        buffer_from_overhead = restart_overhead * 5.0
        buffer_from_gap = gap * 10.0
        B = max(buffer_from_overhead, buffer_from_gap, restart_overhead)

        # Once slack is at or below the buffer, commit to on-demand only.
        # This guarantees completion before deadline as long as it's feasible at all.
        if slack_est <= B:
            return ClusterType.ON_DEMAND

        # Far from the deadline: aggressively use spot when available.
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable and we're still far from the deadline: wait (no cost).
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)