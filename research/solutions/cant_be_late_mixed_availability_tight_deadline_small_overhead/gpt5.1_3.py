from typing import Any
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args: Any):
        super().__init__(args)
        self.args = args
        # Whether we have permanently committed to on-demand for this environment.
        self._od_commit = False
        # To detect when a new environment is attached.
        self._env_internal_id = None

    def solve(self, spec_path: str) -> "Solution":
        # Optional initialization from spec_path; unused in this solution.
        return self

    def _ensure_env_state(self):
        """Reset per-environment state when a new env is attached."""
        env_id = id(self.env) if hasattr(self, "env") else None
        if env_id != self._env_internal_id:
            self._env_internal_id = env_id
            self._od_commit = False

    def _compute_work_done(self) -> float:
        """Robustly compute total work done from task_done_time."""
        segments = getattr(self, "task_done_time", None)
        if not segments:
            return 0.0
        total = 0.0
        for seg in segments:
            if isinstance(seg, (int, float)):
                total += float(seg)
            elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
                try:
                    start = float(seg[0])
                    end = float(seg[1])
                    total += max(0.0, end - start)
                except (TypeError, ValueError):
                    continue
        return total

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Make sure per-environment state is initialized.
        self._ensure_env_state()

        # Compute remaining work.
        work_done = self._compute_work_done()
        task_duration = float(self.task_duration)
        work_remaining = max(0.0, task_duration - work_done)

        # If work is done (or effectively done), do nothing.
        gap = float(self.env.gap_seconds)
        if work_remaining <= gap * 0.25:
            return ClusterType.NONE

        # Time left until deadline.
        elapsed = float(self.env.elapsed_seconds)
        deadline = float(self.deadline)
        time_left = deadline - elapsed

        # If no time left, nothing meaningful to do.
        if time_left <= 0.0:
            return ClusterType.NONE

        # If we've already committed to on-demand, keep using it.
        if self._od_commit:
            return ClusterType.ON_DEMAND

        # Estimate minimum time needed if we switch to on-demand now.
        restart_overhead = float(self.restart_overhead)

        # Assume we pay one restart overhead when moving from non-OD to OD.
        overhead_needed = restart_overhead if last_cluster_type != ClusterType.ON_DEMAND else 0.0

        base_required_time = work_remaining + overhead_needed

        # Safety margin to account for discrete timesteps and modeling errors.
        # Ensure at least one full timestep and one overhead margin.
        safety_seconds = max(2.0 * gap, restart_overhead)

        min_time_needed = base_required_time + safety_seconds

        # If time remaining is tight, permanently commit to on-demand.
        if time_left <= min_time_needed:
            self._od_commit = True
            return ClusterType.ON_DEMAND

        # Otherwise, we are sufficiently early: favor spot, idle when spot unavailable.
        if has_spot:
            return ClusterType.SPOT

        # No spot available and we have plenty of slack: wait.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)