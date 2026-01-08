import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy with hard-deadline guarantee."""
    NAME = "cant_be_late_multiregion_v1"

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

        # Normalize scalar parameters from potential list forms
        td = getattr(self, "task_duration", None)
        if isinstance(td, (list, tuple)):
            td = td[0]
        dl = getattr(self, "deadline", None)
        if isinstance(dl, (list, tuple)):
            dl = dl[0]
        ro = getattr(self, "restart_overhead", None)
        if isinstance(ro, (list, tuple)):
            ro = ro[0]

        # Store scalar versions for our own logic (seconds)
        self._task_duration_scalar = float(td)
        self._deadline_scalar = float(dl)
        self._restart_overhead_scalar = float(ro)

        # Remaining slack in ideal world (no more overheads or idling)
        total_slack = max(0.0, self._deadline_scalar - self._task_duration_scalar)

        # Commit to on-demand when remaining slack dips below this threshold.
        # Use both an absolute multiple of overhead and a fraction of total slack.
        commit_slack_from_overhead = 2.0 * self._restart_overhead_scalar
        commit_slack_from_fraction = 0.2 * total_slack
        self.commit_slack_seconds = max(commit_slack_from_overhead, commit_slack_from_fraction)

        # Do not let commit_slack exceed total_slack (if slack is tiny, fall back to 50% of it)
        if self.commit_slack_seconds > total_slack and total_slack > 0.0:
            self.commit_slack_seconds = 0.5 * total_slack

        # Once we decide to switch to on-demand, we never go back to spot.
        self.force_ondemand = False

        # Efficient tracking of completed work without re-summing the whole list every step.
        self._prev_task_done_len = 0
        self._total_work_done = 0.0

        return self

    def _update_work_done_cache(self) -> None:
        """Incrementally update cached total work done from task_done_time list."""
        task_done_list = getattr(self, "task_done_time", [])
        curr_len = len(task_done_list)
        if curr_len > self._prev_task_done_len:
            # Sum only the newly appended segments
            new_segments = task_done_list[self._prev_task_done_len:curr_len]
            self._total_work_done += sum(new_segments)
            self._prev_task_done_len = curr_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update cached progress
        self._update_work_done_cache()

        remaining_work = max(0.0, self._task_duration_scalar - self._total_work_done)

        # If task is effectively done, stop using any cluster.
        if remaining_work <= 0.0:
            return ClusterType.NONE

        # Remaining wall-clock time until deadline
        time_left = self._deadline_scalar - self.env.elapsed_seconds

        # If somehow past deadline, still run on-demand to minimize penalty impact.
        if time_left <= 0.0:
            self.force_ondemand = True
            return ClusterType.ON_DEMAND

        # Remaining ideal slack = time_left - remaining_work
        slack = time_left - remaining_work

        # If slack falls below threshold, commit to on-demand for the rest of the run.
        if (not self.force_ondemand) and (slack <= self.commit_slack_seconds):
            self.force_ondemand = True

        if self.force_ondemand:
            # Guaranteed progress to meet deadline.
            return ClusterType.ON_DEMAND

        # Still in spot-preferred phase: use spot when available, otherwise idle.
        if has_spot:
            return ClusterType.SPOT

        # No spot currently; consume some slack by idling without incurring on-demand cost.
        return ClusterType.NONE