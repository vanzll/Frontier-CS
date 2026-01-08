from typing import Any, List, Tuple
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args: Any = None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except TypeError:
                pass
        self.args = args
        self._commit_to_od = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _estimate_work_done(self) -> float:
        segments = getattr(self, "task_done_time", None)
        if not segments:
            return 0.0

        try:
            if isinstance(segments, list):
                first = segments[0]
            else:
                # Fallback for non-list sequences
                first = next(iter(segments))
        except (StopIteration, TypeError):
            return 0.0

        total = 0.0

        # Case 1: list of numeric values (durations or cumulative progress)
        if isinstance(first, (int, float)):
            seg_list = list(segments)
            n = len(seg_list)
            if n == 1:
                done = float(seg_list[0])
            else:
                increasing = True
                last_val = float(seg_list[0])
                for x in seg_list[1:]:
                    val = float(x)
                    if val < last_val - 1e-9:
                        increasing = False
                        break
                    last_val = val
                if increasing:
                    # Treat as cumulative progress or end times
                    done = float(seg_list[-1])
                else:
                    # Treat as durations
                    done = float(sum(float(x) for x in seg_list))
            try:
                task_duration = float(self.task_duration)
            except Exception:
                task_duration = done
            if task_duration > 0:
                return min(done, task_duration)
            return done

        # Case 2: list of segments, likely (start, end) pairs
        total = 0.0
        for seg in segments:
            if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                try:
                    start = float(seg[0])
                    end = float(seg[1])
                    if end > start:
                        total += end - start
                except (TypeError, ValueError):
                    continue

        try:
            task_duration = float(self.task_duration)
        except Exception:
            task_duration = total
        if task_duration > 0:
            return min(total, task_duration)
        return total

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If we've already committed to on-demand, always stay on on-demand
        if self._commit_to_od:
            return ClusterType.ON_DEMAND

        # Safely fetch environment parameters
        try:
            elapsed = float(self.env.elapsed_seconds)
        except Exception:
            elapsed = 0.0
        try:
            gap = float(self.env.gap_seconds)
        except Exception:
            gap = 0.0
        try:
            deadline = float(self.deadline)
        except Exception:
            deadline = elapsed
        try:
            task_duration = float(self.task_duration)
        except Exception:
            task_duration = 0.0
        try:
            restart_overhead = float(self.restart_overhead)
        except Exception:
            restart_overhead = 0.0

        work_done = self._estimate_work_done()
        remaining_work = max(0.0, task_duration - work_done)
        time_remaining = max(0.0, deadline - elapsed)

        # If we think the task is done, do nothing
        if remaining_work <= 0.0:
            self._commit_to_od = False
            return ClusterType.NONE

        # Decide if we must commit to on-demand to meet the deadline
        if not self._commit_to_od:
            # Slack if we immediately switch to OD and pay one restart overhead
            slack_od = time_remaining - (remaining_work + restart_overhead)
            # Commit margin: require enough slack for two time steps or at least
            # a fraction of the restart overhead, to be conservative.
            commit_margin = max(2.0 * gap, 0.5 * restart_overhead)

            # If slack is small, we can no longer gamble on spot/idle.
            if slack_od <= commit_margin:
                self._commit_to_od = True
                return ClusterType.ON_DEMAND

        # Pre-commit phase: use spot whenever available, else pause (NONE)
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)