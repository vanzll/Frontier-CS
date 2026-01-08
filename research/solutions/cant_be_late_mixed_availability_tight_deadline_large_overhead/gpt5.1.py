from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType
import numbers
import json
import os


class Solution(Strategy):
    NAME = "cant_be_late_threshold_v1"

    def __init__(self, args=None):
        super().__init__(args)
        self._spec = None
        self._loaded_spec = False

    def solve(self, spec_path: str) -> "Solution":
        if spec_path and os.path.exists(spec_path):
            try:
                with open(spec_path, "r") as f:
                    self._spec = json.load(f)
            except Exception:
                self._spec = None
        self._loaded_spec = True
        return self

    def _compute_work_done(self) -> float:
        # Try environment-provided aggregate attributes first (if any)
        if hasattr(self, "env"):
            for attr in (
                "task_elapsed_seconds",
                "task_run_seconds",
                "task_progress_seconds",
                "work_done_seconds",
            ):
                val = getattr(self.env, attr, None)
                if isinstance(val, numbers.Number):
                    return float(val)

        segments = getattr(self, "task_done_time", None)
        if not segments:
            return 0.0

        try:
            first = segments[0]
        except Exception:
            return 0.0

        # Case 1: plain numeric durations
        if isinstance(first, numbers.Number):
            total = 0.0
            for s in segments:
                try:
                    total += float(s)
                except Exception:
                    continue
            return total

        # Case 2: (start, end) tuples/lists
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            if isinstance(first[0], numbers.Number) and isinstance(first[1], numbers.Number):
                total = 0.0
                for s in segments:
                    try:
                        start, end = float(s[0]), float(s[1])
                        if end > start:
                            total += end - start
                    except Exception:
                        continue
                return total

        # Case 3: objects with .duration
        if hasattr(first, "duration"):
            total = 0.0
            for s in segments:
                try:
                    total += float(s.duration)
                except Exception:
                    continue
            return total

        # Case 4: objects with .start and .end
        if hasattr(first, "start") and hasattr(first, "end"):
            total = 0.0
            for s in segments:
                try:
                    start = float(s.start)
                    end = float(s.end)
                    if end > start:
                        total += end - start
                except Exception:
                    continue
            return total

        # Fallback: best-effort numeric cast
        total = 0.0
        for s in segments:
            try:
                total += float(s)
            except Exception:
                continue
        return total

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Compute remaining work
        work_done = self._compute_work_done()
        try:
            total_duration = float(self.task_duration)
        except Exception:
            total_duration = work_done
        remaining_work = max(total_duration - work_done, 0.0)

        # If job appears done, avoid incurring further cost
        if remaining_work <= 0.0:
            return ClusterType.NONE

        # Time bookkeeping
        elapsed = getattr(self.env, "elapsed_seconds", 0.0)
        try:
            deadline = float(self.deadline)
        except Exception:
            deadline = elapsed + remaining_work

        time_left = deadline - elapsed
        if time_left <= 0.0:
            # Already out of time; best effort is OD
            return ClusterType.ON_DEMAND

        # Overhead / slack calculations
        try:
            overhead = float(self.restart_overhead)
        except Exception:
            overhead = 0.0

        # Slack S = time_left - (remaining_work + overhead buffer)
        slack = time_left - (remaining_work + overhead)

        # If we already have no slack, must run OD to maximize chance to finish
        if slack <= 0.0:
            return ClusterType.ON_DEMAND

        # Total theoretical slack if we had run OD from start with one overhead
        try:
            total_slack = float(self.deadline) - float(self.task_duration)
        except Exception:
            total_slack = time_left - remaining_work

        if total_slack < 0.0:
            total_slack = 0.0

        # Parameterization of thresholds
        # Use multiples of overhead, but clamp by total_slack to stay reasonable
        spot_slack_mult = 4.0
        idle_slack_mult = 8.0

        min_spot_slack = overhead * spot_slack_mult
        idle_slack_threshold = overhead * idle_slack_mult

        # Ensure thresholds are not unrealistically large compared to total slack
        if total_slack > 0.0:
            max_idling = 0.5 * total_slack
            if idle_slack_threshold > max_idling:
                idle_slack_threshold = max_idling
            if min_spot_slack > 0.5 * total_slack:
                min_spot_slack = 0.5 * total_slack

        # Also ensure thresholds are at least a couple of timesteps to react
        gap = getattr(self.env, "gap_seconds", 0.0)
        min_reactive = 2.0 * gap
        if min_reactive > 0.0:
            if min_spot_slack < min_reactive:
                min_spot_slack = min_reactive
            if idle_slack_threshold < min_reactive:
                idle_slack_threshold = min_reactive

        # Decision logic
        if has_spot:
            # If slack is small, avoid risking further preemption; go OD.
            if slack <= min_spot_slack:
                return ClusterType.ON_DEMAND
            # Otherwise, exploit cheaper spot capacity.
            return ClusterType.SPOT
        else:
            # No spot currently available.
            # If we have plenty of slack, we can afford to wait.
            if slack > idle_slack_threshold:
                return ClusterType.NONE
            # Slack is getting tight: pay for OD to maintain schedule.
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)