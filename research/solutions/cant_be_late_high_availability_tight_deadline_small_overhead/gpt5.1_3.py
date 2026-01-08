from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_hybrid_v1"

    def solve(self, spec_path: str) -> "Solution":
        # Initialize internal state; spec_path not used in this heuristic.
        self._fallback_mode = False
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Lazy init in case solve() was not called.
        if not hasattr(self, "_fallback_mode"):
            self._fallback_mode = False

        # Basic environment values
        env = self.env
        current_time = env.elapsed_seconds
        gap = getattr(env, "gap_seconds", 0.0)
        overhead = getattr(self, "restart_overhead", 0.0)

        # Compute total work done so far.
        # task_done_time is documented as list of completed work segments.
        work_done = float(sum(self.task_done_time)) if self.task_done_time else 0.0
        remaining_work = max(self.task_duration - work_done, 0.0)

        # If task is complete, no need to run more.
        if remaining_work <= 0.0:
            return ClusterType.NONE

        remaining_time = self.deadline - current_time

        # If no wall-clock time left, just run on-demand (nothing else can help).
        if remaining_time <= 0.0:
            return ClusterType.ON_DEMAND

        # Slack remaining is how much "idle/overhead" time we can still afford.
        # slack_remaining = (time until deadline) - (work left)
        slack_remaining = remaining_time - remaining_work

        # If slack_remaining is strongly negative, we're already behind schedule;
        # best we can do is always on-demand.
        if slack_remaining < -1e-6:
            self._fallback_mode = True

        # Total slack from the start (may be <= 0 in pathological specs).
        slack_total = self.deadline - self.task_duration

        # Safety buffers (seconds)
        # critical_buffer: when remaining slack is this small, switch to always on-demand.
        # Ensure it's at least overhead + one step to cover final restart and discretization.
        critical_buffer = max(overhead + gap, 2.0 * gap, 60.0)

        # If total slack is smaller than this, effectively we should be very conservative.
        if slack_total > 0.0:
            critical_buffer = min(critical_buffer, slack_total)
        else:
            critical_buffer = 0.0

        # Mid buffer: when slack_remaining <= mid_buffer, use hybrid (SPOT if available, else OD).
        # Before this, use aggressive spot (SPOT or NONE).
        if slack_total > 0.0:
            mid_buffer = 0.5 * slack_total  # use about half of slack before switching to hybrid
            mid_buffer = max(mid_buffer, 3.0 * critical_buffer)
            mid_buffer = min(mid_buffer, slack_total)
        else:
            mid_buffer = 0.0

        # Once we enter fallback mode, stay there: always on-demand until completion.
        if self._fallback_mode:
            return ClusterType.ON_DEMAND

        # Enter fallback mode if slack is critically low.
        if slack_remaining <= critical_buffer + 1e-9:
            self._fallback_mode = True
            return ClusterType.ON_DEMAND

        # Hybrid region: protect remaining slack by using OD when spot is unavailable.
        if slack_remaining <= mid_buffer + 1e-9:
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND

        # High-slack region: be aggressive on spot to minimize cost.
        if has_spot:
            return ClusterType.SPOT
        else:
            # Use idle time (NONE) while we still have plenty of slack.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)