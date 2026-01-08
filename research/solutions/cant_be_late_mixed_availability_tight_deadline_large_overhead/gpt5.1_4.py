from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        # Optional initialization, nothing required for now.
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Lazy initialization and per-episode reset handling.
        env = self.env

        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._last_elapsed = getattr(env, "elapsed_seconds", 0.0)
            self._cached_done_sum = sum(getattr(self, "task_done_time", []))
            self._cached_done_len = len(getattr(self, "task_done_time", []))
            self.lock_on_demand = False
        else:
            current_elapsed = getattr(env, "elapsed_seconds", 0.0)
            # Detect new episode if elapsed time jumps backwards.
            if current_elapsed < getattr(self, "_last_elapsed", 0.0):
                self._cached_done_sum = sum(getattr(self, "task_done_time", []))
                self._cached_done_len = len(getattr(self, "task_done_time", []))
                self.lock_on_demand = False
            else:
                # Update cached sum if new segments were appended or list changed.
                task_done_time = getattr(self, "task_done_time", [])
                current_len = len(task_done_time)
                if current_len != getattr(self, "_cached_done_len", 0):
                    if current_len > self._cached_done_len:
                        self._cached_done_sum += sum(
                            task_done_time[self._cached_done_len : current_len]
                        )
                    else:
                        # Fallback: recompute if list shrank or unexpected mutation.
                        self._cached_done_sum = sum(task_done_time)
                    self._cached_done_len = current_len
            self._last_elapsed = current_elapsed

        # Shortcuts to frequently used attributes.
        elapsed = float(getattr(env, "elapsed_seconds", 0.0))
        gap = float(getattr(env, "gap_seconds", 0.0))
        deadline = float(getattr(self, "deadline", 0.0))
        task_duration = float(getattr(self, "task_duration", 0.0))
        restart_overhead = float(getattr(self, "restart_overhead", 0.0))

        time_left = deadline - elapsed
        if time_left <= 0.0:
            # No time left; if work remains this episode is effectively failed,
            # but our choice no longer matters. Return ON_DEMAND to be safe.
            remaining = task_duration - getattr(self, "_cached_done_sum", 0.0)
            if remaining > 0.0:
                return ClusterType.ON_DEMAND
            return ClusterType.NONE

        remaining_work = task_duration - getattr(self, "_cached_done_sum", 0.0)
        if remaining_work <= 0.0:
            # Task already completed; no need to run anything.
            return ClusterType.NONE

        # Once we decide to lock into on-demand, never switch back.
        if getattr(self, "lock_on_demand", False):
            return ClusterType.ON_DEMAND

        # Safety check: can we afford to not use on-demand this step?
        # Worst case, we get zero progress for this step (gap) and then
        # need restart_overhead plus remaining_work entirely on on-demand.
        buffer_time = restart_overhead + gap
        if remaining_work + buffer_time > time_left:
            # Not safe to skip ON_DEMAND this step; lock into ON_DEMAND.
            self.lock_on_demand = True
            return ClusterType.ON_DEMAND

        # Safe region: prefer SPOT when available, otherwise wait (NONE).
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)