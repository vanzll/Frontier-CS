import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args=None):
        super().__init__(args)
        self._sum_done_time = 0.0
        self._last_len = 0
        self._committed_to_od = False
        self._initialized = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _update_progress(self):
        # Incrementally sum task_done_time to avoid repeated O(n) sums.
        try:
            segs = self.task_done_time
            if segs is None:
                return
            cur_len = len(segs)
            if cur_len > self._last_len:
                add = 0.0
                # Sum only new segments
                for v in segs[self._last_len:cur_len]:
                    try:
                        add += float(v)
                    except Exception:
                        # If any element is non-numeric, fallback to full sum once
                        self._sum_done_time = float(sum(float(x) for x in segs))
                        self._last_len = cur_len
                        return
                self._sum_done_time += add
                self._last_len = cur_len
        except Exception:
            # Fallback: best-effort (avoid crashing); keep previous sum
            pass

    def _remaining_work(self) -> float:
        self._update_progress()
        remaining = float(self.task_duration) - float(self._sum_done_time)
        if remaining < 0:
            remaining = 0.0
        return remaining

    def _latest_od_start_time(self, remaining_work: float) -> float:
        # Latest time we can start OD and still finish by deadline (including one restart overhead).
        return float(self.deadline) - float(remaining_work) - float(self.restart_overhead)

    def _should_commit_to_od_now(self, t_now: float, gap: float, latest_start: float) -> bool:
        # If waiting one more step would push us past the latest start time, commit now.
        # Add a small safety margin to account for discretization and timing jitter.
        safety_margin = max(0.0, 0.25 * gap)
        return (t_now + gap) > (latest_start - safety_margin)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Initialize lazily to access environment details
        if not self._initialized:
            self._initialized = True
            self._sum_done_time = 0.0
            self._last_len = 0
            self._committed_to_od = False

        # If already committed to OD, always continue OD to avoid extra restarts.
        if self._committed_to_od:
            return ClusterType.ON_DEMAND

        t_now = float(self.env.elapsed_seconds)
        gap = float(self.env.gap_seconds)

        remaining = self._remaining_work()
        if remaining <= 0:
            return ClusterType.NONE

        latest_start = self._latest_od_start_time(remaining)

        # If we must start OD now to guarantee we can finish by deadline, commit.
        if self._should_commit_to_od_now(t_now, gap, latest_start):
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        # Otherwise, prefer SPOT if available; if not, wait (NONE) to save cost.
        if has_spot:
            return ClusterType.SPOT

        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)