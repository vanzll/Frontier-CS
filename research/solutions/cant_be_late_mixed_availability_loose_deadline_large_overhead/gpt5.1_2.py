from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Lazy initialization to avoid dependency on Strategy.__init__ signature.
        if not hasattr(self, "_initialized"):
            self._initialized = True

            # Base slack: deadline - pure on-demand runtime.
            self._slack0 = max(self.deadline - self.task_duration, 0.0)

            # Safety buffer before switching to pure on-demand.
            self._bail_s = 2 * 3600.0  # 2 hours
            if self._bail_s > self._slack0:
                # If slack is small, scale bail threshold down.
                self._bail_s = 0.5 * self._slack0
            if self._bail_s < 0.0:
                self._bail_s = 0.0

            # Threshold above which we are willing to wait (NONE) when spot is unavailable.
            self._wait_threshold_s = min(12 * 3600.0, self._slack0)  # 12 hours or total slack
            if self._wait_threshold_s < self._bail_s:
                self._wait_threshold_s = self._bail_s

            # Spot quality estimation parameters.
            self._min_spot_eval_time = 3600.0  # Need at least 1 hour of spot usage to judge.
            self._spot_ratio_threshold = 0.4   # Below this, spot is more expensive than OD.

            # Stats for estimating effective spot/on-demand throughput.
            self._spot_wall_time = 0.0
            self._spot_progress = 0.0
            self._od_wall_time = 0.0
            self._od_progress = 0.0
            self._spot_is_bad = False

            # Track time and progress between steps.
            self._last_elapsed = getattr(self.env, "elapsed_seconds", 0.0)
            td_list = getattr(self, "task_done_time", [])
            self._last_task_done_len = len(td_list)
            if self._last_task_done_len:
                self._cumulative_progress = float(sum(td_list))
            else:
                self._cumulative_progress = 0.0
            self._prev_cumulative_progress = self._cumulative_progress

            # For availability statistics (not heavily used but may be useful).
            self._prev_has_spot = has_spot
            self._total_time = 0.0
            self._has_spot_time = 0.0

        # Current time and elapsed since last step.
        elapsed = self.env.elapsed_seconds
        dt = elapsed - self._last_elapsed
        if dt < 0.0:
            dt = 0.0

        # Update availability stats using previous has_spot.
        if self._prev_has_spot:
            self._has_spot_time += dt
        self._total_time += dt
        self._prev_has_spot = has_spot

        # Update cumulative progress from new task_done_time entries.
        td_list = self.task_done_time
        cur_len = len(td_list)
        if cur_len > self._last_task_done_len:
            new_prog = 0.0
            for i in range(self._last_task_done_len, cur_len):
                new_prog += td_list[i]
            self._cumulative_progress += new_prog
            self._last_task_done_len = cur_len

        # Progress gained since previous step.
        dprog = self._cumulative_progress - self._prev_cumulative_progress
        if dprog < 0.0:
            dprog = 0.0

        # Update per-cluster-type throughput statistics.
        if dt > 0.0:
            if last_cluster_type == ClusterType.SPOT:
                self._spot_wall_time += dt
                self._spot_progress += dprog
            elif last_cluster_type == ClusterType.ON_DEMAND:
                self._od_wall_time += dt
                self._od_progress += dprog

        # Save for next step.
        self._prev_cumulative_progress = self._cumulative_progress
        self._last_elapsed = elapsed

        # Clamp progress and compute remaining work.
        progress = self._cumulative_progress
        if progress < 0.0:
            progress = 0.0
        if progress > self.task_duration:
            progress = self.task_duration

        remaining_work = self.task_duration - progress
        if remaining_work <= 0.0:
            # Task already done; no need to schedule compute.
            return ClusterType.NONE

        # Compute "salvage time" = extra slack remaining if we switch to full on-demand now.
        # salvage = (deadline - current_time) - remaining_on_demand_work
        #         = (deadline - elapsed) - (task_duration - progress)
        #         = (deadline - task_duration) + progress - elapsed
        salvage = self._slack0 + progress - elapsed

        # Update spot effectiveness estimate if we have sufficient data.
        if self._spot_wall_time >= self._min_spot_eval_time:
            if self._spot_wall_time > 0.0:
                spot_rate = self._spot_progress / self._spot_wall_time
            else:
                spot_rate = 0.0
            if spot_rate < 0.0:
                spot_rate = 0.0
            elif spot_rate > 1.2:
                spot_rate = 1.2
            self._spot_is_bad = spot_rate < self._spot_ratio_threshold

        # If we're already behind schedule even with pure on-demand, just use on-demand.
        if salvage <= 0.0:
            return ClusterType.ON_DEMAND

        # If salvage is small, always fall back to on-demand regardless of spot.
        bail_s = self._bail_s
        if salvage <= bail_s:
            return ClusterType.ON_DEMAND

        # Now salvage is comfortably positive.
        if has_spot:
            # Spot is available: prefer it unless it's clearly cost-ineffective.
            if self._spot_is_bad:
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT
        else:
            # Spot unavailable: either wait (NONE) if we have lots of slack, or use on-demand.
            if self._slack0 > 0.0 and salvage >= self._wait_threshold_s:
                return ClusterType.NONE
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)