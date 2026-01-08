from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        # Safety margin in seconds to account for discretization and multiple restarts.
        # 1800s = 0.5 hours.
        self.safety_margin = 1800.0
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Ensure safety_margin is defined even if solve() was not called.
        if not hasattr(self, "safety_margin"):
            self.safety_margin = 1800.0

        # Total work done so far.
        task_done_time = getattr(self, "task_done_time", [])
        done = 0.0
        try:
            for seg in task_done_time:
                if isinstance(seg, (int, float)):
                    done += float(seg)
                elif isinstance(seg, dict):
                    if "duration" in seg:
                        done += float(seg["duration"])
                    elif "len" in seg:
                        done += float(seg["len"])
                    elif "length" in seg:
                        done += float(seg["length"])
                    elif "start" in seg and "end" in seg:
                        done += float(seg["end"]) - float(seg["start"])
                else:
                    # Assume iterable of (start, end)
                    try:
                        if len(seg) == 2:
                            start, end = seg
                            done += float(end) - float(start)
                    except Exception:
                        # Unknown format; ignore.
                        pass
        except Exception:
            # In case task_done_time is not iterable or has unexpected structure,
            # fall back to zero (conservative).
            done = 0.0

        remaining = self.task_duration - done
        if remaining <= 0:
            # Job already finished; no need to run more.
            return ClusterType.NONE

        t = self.env.elapsed_seconds
        dt = getattr(self.env, "gap_seconds", 0.0)
        D = self.deadline
        O = self.restart_overhead
        buffer_time = self.safety_margin

        # Fallbacks for robustness.
        if dt is None:
            dt = 0.0
        if O is None:
            O = 0.0
        if buffer_time is None:
            buffer_time = 0.0

        # Check if we can afford to "waste" this step (i.e., potentially get no progress)
        # and still finish by the deadline if we switch to on-demand starting next step.
        # Worst-case: 0 progress this step, then overhead O, then remaining work.
        # We also subtract an additional safety margin (buffer_time).
        can_wait = (t + dt + O + remaining + buffer_time) <= D

        if can_wait:
            if has_spot:
                # Cheap and still safe to rely on spot for this step.
                return ClusterType.SPOT
            else:
                # Spot not available but safe to wait.
                return ClusterType.NONE
        else:
            # Not safe to wait any longer; commit to on-demand.
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)