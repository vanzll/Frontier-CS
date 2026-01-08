import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Deadline-safe multi-region scheduling strategy with spot preference."""

    NAME = "deadline_safe_spot_first"

    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        # Per-solver (possibly multi-episode) state
        self._episode_last_elapsed = None
        self._accumulated_work = 0.0
        self._last_segments_len = 0
        self._committed_od = False

        return self

    # --- Internal helpers -------------------------------------------------

    def _reset_episode_state(self) -> None:
        """Reset state at the beginning of a new episode."""
        segs = getattr(self, "task_done_time", None)
        if segs:
            self._accumulated_work = float(sum(segs))
            self._last_segments_len = len(segs)
        else:
            self._accumulated_work = 0.0
            self._last_segments_len = 0
        self._committed_od = False

    def _update_progress(self) -> None:
        """Incrementally maintain total work done."""
        segs = self.task_done_time
        if not segs:
            return
        current_len = len(segs)
        if current_len > self._last_segments_len:
            self._accumulated_work += float(
                sum(segs[self._last_segments_len:current_len])
            )
            self._last_segments_len = current_len

    # --- Core decision logic ----------------------------------------------

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = self.env
        elapsed = env.elapsed_seconds

        # Detect new episode (elapsed time reset) and reinitialize state.
        if self._episode_last_elapsed is None or elapsed < self._episode_last_elapsed:
            self._reset_episode_state()
        self._episode_last_elapsed = elapsed

        # Maintain cumulative work done.
        self._update_progress()

        step_sec = env.gap_seconds
        work_done = self._accumulated_work
        remaining_work = max(0.0, self.task_duration - work_done)
        time_left = self.deadline - elapsed

        # If task is already completed, no need to run anything.
        if remaining_work <= 0.0:
            return ClusterType.NONE

        # If we are past the deadline, nothing can help now.
        if time_left <= 0.0:
            return ClusterType.NONE

        # If not yet committed to on-demand, decide whether to commit.
        if not self._committed_od:
            # Conservative upper bound on time to finish if we switch to on-demand:
            # - remaining_work seconds of compute
            # - one restart_overhead for the switch
            T_od = remaining_work + self.restart_overhead

            # Slack before switching to on-demand.
            slack = time_left - T_od

            # Maximum time we are willing to "risk" by doing non-on-demand work
            # (spot or waiting) before we must still be able to safely bail out.
            # We consider at most one step plus one restart overhead as worst-case.
            max_extra_risk = step_sec + self.restart_overhead

            if slack <= max_extra_risk:
                self._committed_od = True

        # Once committed, always use on-demand to avoid further risk.
        if self._committed_od:
            return ClusterType.ON_DEMAND

        # We still have comfortable slack and are not committed to on-demand.

        # If spot is available, always prefer it.
        if has_spot:
            return ClusterType.SPOT

        # Spot not available: decide between waiting (NONE) and using on-demand.
        # Recompute a slightly more conservative slack to decide.
        T_od = remaining_work + self.restart_overhead
        slack = time_left - T_od
        # If slack is getting somewhat tight, start on-demand instead of waiting.
        if slack <= 2.0 * (step_sec + self.restart_overhead):
            self._committed_od = True
            return ClusterType.ON_DEMAND

        # Otherwise, we can afford to wait for cheaper future spot availability.
        return ClusterType.NONE