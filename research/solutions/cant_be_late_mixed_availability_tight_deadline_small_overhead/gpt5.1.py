from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "slack_threshold_strategy"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Get environment parameters
        env = self.env
        gap = env.gap_seconds
        time_now = env.elapsed_seconds

        # Compute total work done so far
        done_segments = getattr(self, "task_done_time", None)
        if done_segments:
            try:
                total_done = sum(done_segments)
            except TypeError:
                # Fallback in case task_done_time has unexpected structure
                total_done = 0.0
        else:
            total_done = 0.0

        # Remaining work and time
        remaining_work = max(self.task_duration - total_done, 0.0)
        time_left = max(self.deadline - time_now, 0.0)

        # If somehow already done, don't run anything
        if remaining_work <= 0.0:
            return ClusterType.NONE

        # Slack: how much "no-progress" time we can still afford
        slack = time_left - remaining_work

        # Safety threshold for taking risk (SPOT or idling)
        # Ensure we can always recover with at most one gap of no progress
        # plus one restart_overhead before switching to guaranteed on-demand.
        safe_slack_for_gamble = self.restart_overhead + gap

        # If we don't have enough slack to gamble, always use on-demand
        if slack <= safe_slack_for_gamble:
            return ClusterType.ON_DEMAND

        # We have enough slack to gamble
        if has_spot:
            # Prefer spot when available
            return ClusterType.SPOT

        # Spot not available; with sufficient slack, wait for spot to return
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)