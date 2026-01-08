from __future__ import annotations

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_safe_v1"

    def solve(self, spec_path: str) -> "Solution":
        # Initialize internal state
        self._commit_to_od = False
        return self

    def _compute_work_done(self) -> float:
        segments = getattr(self, "task_done_time", None)
        if not segments:
            return 0.0

        total = 0.0
        for seg in segments:
            if seg is None:
                continue

            # Simple numeric duration
            if isinstance(seg, (int, float)):
                total += float(seg)
                continue

            # Tuple/list representations
            if isinstance(seg, (list, tuple)):
                if len(seg) == 2 and all(isinstance(x, (int, float)) for x in seg):
                    total += float(seg[1] - seg[0])
                    continue
                if len(seg) == 1 and isinstance(seg[0], (int, float)):
                    total += float(seg[0])
                    continue

            # Object with start/end or start_time/end_time
            start = None
            end = None

            if hasattr(seg, "start"):
                start = seg.start
            elif hasattr(seg, "start_time"):
                start = seg.start_time

            if hasattr(seg, "end"):
                end = seg.end
            elif hasattr(seg, "end_time"):
                end = seg.end_time

            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                total += float(end - start)
                continue

            # Fallback: duration attribute
            dur = getattr(seg, "duration", None)
            if isinstance(dur, (int, float)):
                total += float(dur)
                continue

        return total

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Ensure internal flag exists
        if not hasattr(self, "_commit_to_od"):
            self._commit_to_od = False

        task_duration = getattr(self, "task_duration", None)
        if task_duration is None:
            # If we don't know the task duration, safest is always on-demand
            return ClusterType.ON_DEMAND

        work_done = self._compute_work_done()
        remaining = max(0.0, float(task_duration) - float(work_done))

        # If task is complete, no need to run anything
        if remaining <= 0.0:
            return ClusterType.NONE

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        deadline = getattr(self, "deadline", None)
        if deadline is None:
            # No deadline info; to avoid any chance of lateness, use on-demand
            return ClusterType.ON_DEMAND

        time_left = float(deadline) - elapsed
        if time_left <= 0.0:
            # Already at or past deadline; nothing sensible to do
            return ClusterType.NONE

        # If we've already committed to on-demand, stay there until done
        if self._commit_to_od:
            return ClusterType.ON_DEMAND

        R = float(getattr(self, "restart_overhead", 0.0))
        gap = float(getattr(self.env, "gap_seconds", 0.0))

        # If we don't know timing parameters, fall back to always on-demand
        if R <= 0.0 or gap <= 0.0:
            self._commit_to_od = True
            return ClusterType.ON_DEMAND

        # Viability slack: extra time beyond what we need to finish on OD,
        # assuming at most one future restart overhead R.
        slack_viable = time_left - (remaining + R)

        # Conservative upper bound on worst-case slack loss from one more
        # speculative step (spot or idle): step duration plus multiple
        # potential overheads.
        risk_cost_per_step = gap + 4.0 * R

        # If we don't have enough slack to safely risk even one more speculative
        # step, commit to on-demand now and never go back to spot.
        if slack_viable <= risk_cost_per_step:
            self._commit_to_od = True
            return ClusterType.ON_DEMAND

        # Still in speculative region: prefer spot if available
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable and we still have slack: wait (NONE)
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)