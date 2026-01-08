from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_schedule_v1"

    def __init__(self, args=None):
        super().__init__(args)
        self.force_on_demand = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _estimate_done_work(self) -> float:
        td = getattr(self, "task_done_time", None)
        if td is None:
            return 0.0

        total = 0.0

        # If it's a single numeric value
        if isinstance(td, (int, float)):
            val = float(td)
            return val if val > 0.0 else 0.0

        # Try treating it as an iterable of segments
        try:
            for seg in td:
                dur = 0.0
                if isinstance(seg, (int, float)):
                    dur = float(seg)
                elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
                    try:
                        start = float(seg[0])
                        end = float(seg[1])
                        dur = end - start
                    except Exception:
                        dur = 0.0
                else:
                    try:
                        dur = float(seg)
                    except Exception:
                        dur = 0.0
                if dur > 0.0:
                    total += dur
        except TypeError:
            # Fallback if not iterable
            try:
                val = float(td)
                total = val if val > 0.0 else 0.0
            except Exception:
                total = 0.0

        # Clamp to task_duration if available
        try:
            tdur = float(self.task_duration)
            if tdur >= 0.0 and total > tdur:
                total = tdur
        except Exception:
            pass

        return total

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If we've already committed to on-demand, keep using it
        if getattr(self, "force_on_demand", False):
            return ClusterType.ON_DEMAND

        # Estimate remaining work
        try:
            task_duration = float(self.task_duration)
        except Exception:
            task_duration = 0.0

        done_work = self._estimate_done_work()
        remaining_work = max(task_duration - done_work, 0.0)

        # If job is effectively done, do nothing
        if remaining_work <= 0.0:
            self.force_on_demand = False
            return ClusterType.NONE

        # Retrieve timing parameters
        elapsed = getattr(self.env, "elapsed_seconds", 0.0)
        gap = getattr(self.env, "gap_seconds", 1.0)

        try:
            deadline = float(self.deadline)
        except Exception:
            deadline = elapsed + remaining_work

        try:
            overhead = float(self.restart_overhead)
        except Exception:
            overhead = 0.0

        time_left = deadline - elapsed

        # If we've somehow passed the deadline, use on-demand to finish ASAP
        if time_left <= 0.0:
            self.force_on_demand = True
            return ClusterType.ON_DEMAND

        # If switching to on-demand now is barely sufficient, commit to it
        if time_left <= remaining_work + overhead:
            self.force_on_demand = True
            return ClusterType.ON_DEMAND

        # Check if it's safe to risk one more step without guaranteed progress
        safe_to_wait_one_step = (time_left - gap) >= (remaining_work + overhead)

        if has_spot:
            if safe_to_wait_one_step:
                return ClusterType.SPOT
            else:
                # Too close to deadline to risk spot; commit to on-demand
                self.force_on_demand = True
                return ClusterType.ON_DEMAND
        else:
            if safe_to_wait_one_step:
                # Wait for cheaper spot capacity
                return ClusterType.NONE
            else:
                # Need progress now to ensure we finish; commit to on-demand
                self.force_on_demand = True
                return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)