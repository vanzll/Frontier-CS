from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args):
        super().__init__(args)
        self._global_args = args
        self._init_state()

    def _init_state(self):
        self.force_od = False
        self._progress_done = 0.0
        self._last_seg_index = 0
        self._last_elapsed_seconds = None

    def solve(self, spec_path: str) -> "Solution":
        # Optional: read spec_path if needed. We currently ignore it.
        self._init_state()
        return self

    def _reset_run_state_if_needed(self):
        elapsed = self.env.elapsed_seconds
        if self._last_elapsed_seconds is None or elapsed < self._last_elapsed_seconds:
            # New trace/run detected; reset per-run state.
            self.force_od = False
            self._progress_done = 0.0
            self._last_seg_index = 0

    def _update_progress(self):
        segments = self.task_done_time
        current_len = len(segments)
        if current_len > self._last_seg_index:
            # Incrementally add new completed work segments.
            self._progress_done += sum(segments[self._last_seg_index:current_len])
            self._last_seg_index = current_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Detect new run and reset if necessary.
        self._reset_run_state_if_needed()

        # Update cached progress from environment.
        self._update_progress()

        elapsed = self.env.elapsed_seconds
        time_left = self.deadline - elapsed

        remaining = self.task_duration - self._progress_done
        if remaining <= 0.0 or time_left <= 0.0:
            decision = ClusterType.NONE
        else:
            if not self.force_od:
                gap = float(self.env.gap_seconds)
                # Fudge factor to stay safely ahead of the last-resort OD boundary:
                # at least two steps or 15 minutes, whichever is larger.
                fudge = max(2.0 * gap, 900.0)
                safe_needed = remaining + self.restart_overhead
                if time_left <= safe_needed + fudge:
                    self.force_od = True

            if self.force_od:
                decision = ClusterType.ON_DEMAND
            else:
                if has_spot:
                    decision = ClusterType.SPOT
                else:
                    decision = ClusterType.NONE

        self._last_elapsed_seconds = elapsed
        return decision

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)