import json
import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    MODE_SPOT_WAIT = 0
    MODE_HYBRID = 1
    MODE_OD_LOCK = 2

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass
        self.args = args
        self._initialized = False

    def solve(self, spec_path: str) -> "Solution":
        # Optional: read configuration if present, but keep robust to missing/unknown schema.
        self._spec = None
        try:
            with open(spec_path, "r") as f:
                self._spec = json.load(f)
        except Exception:
            self._spec = None
        return self

    def _init_state(self) -> None:
        self.mode = self.MODE_SPOT_WAIT

        self._steps = 0
        self._spot_steps = 0

        self._prev_has_spot: Optional[bool] = None
        self._up_streak = 0
        self._down_streak = 0

        self._avg_up_steps = 6.0
        self._avg_down_steps = 6.0
        self._ewma_beta = 0.20

        self._hybrid_confirm_steps = 2

        self._initialized = True

    def _work_done_seconds(self) -> float:
        td = getattr(self, "task_duration", 0.0) or 0.0

        tdt = getattr(self, "task_done_time", None)
        if tdt is None:
            done = getattr(self, "task_done", None)
            if isinstance(done, (int, float)):
                return float(max(0.0, min(float(done), float(td) if td > 0 else float(done))))
            return 0.0

        if isinstance(tdt, (int, float)):
            return float(max(0.0, min(float(tdt), float(td) if td > 0 else float(tdt))))

        if not isinstance(tdt, (list, tuple)) or len(tdt) == 0:
            return 0.0

        try:
            last = float(tdt[-1])
        except Exception:
            return 0.0

        nondec = True
        try:
            for i in range(len(tdt) - 1):
                if float(tdt[i]) > float(tdt[i + 1]):
                    nondec = False
                    break
        except Exception:
            nondec = False

        if td > 0 and nondec and 0.0 <= last <= td * 1.01:
            return float(last)

        try:
            s = 0.0
            for x in tdt:
                s += float(x)
            if td > 0:
                s = min(s, float(td))
            return max(0.0, s)
        except Exception:
            return 0.0

    def _get_time_params(self) -> tuple[float, float, float, float, float]:
        env = getattr(self, "env", None)
        elapsed = float(getattr(env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(env, "gap_seconds", 60.0) or 60.0)
        deadline = float(getattr(self, "deadline", elapsed) or elapsed)
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        restart_overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        return elapsed, gap, deadline, task_duration, restart_overhead

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._initialized:
            self._init_state()

        elapsed, gap, deadline, task_duration, restart_overhead = self._get_time_params()

        # Update availability statistics/streaks
        self._steps += 1
        if has_spot:
            self._spot_steps += 1

        if self._prev_has_spot is None:
            self._prev_has_spot = has_spot
            if has_spot:
                self._up_streak = 1
                self._down_streak = 0
            else:
                self._down_streak = 1
                self._up_streak = 0
        else:
            if has_spot:
                if not self._prev_has_spot:
                    # Ended a down streak
                    if self._down_streak > 0:
                        self._avg_down_steps = (1.0 - self._ewma_beta) * self._avg_down_steps + self._ewma_beta * float(
                            self._down_streak
                        )
                    self._down_streak = 0
                    self._up_streak = 1
                else:
                    self._up_streak += 1
            else:
                if self._prev_has_spot:
                    # Ended an up streak
                    if self._up_streak > 0:
                        self._avg_up_steps = (1.0 - self._ewma_beta) * self._avg_up_steps + self._ewma_beta * float(
                            self._up_streak
                        )
                    self._up_streak = 0
                    self._down_streak = 1
                else:
                    self._down_streak += 1
            self._prev_has_spot = has_spot

        done = self._work_done_seconds()
        remaining_work = max(0.0, task_duration - done)

        if remaining_work <= 0.0:
            return ClusterType.NONE

        remaining_time = max(0.0, deadline - elapsed)

        # Safety buffer: at least one step and a few minutes; scale a bit with overhead.
        safety = max(gap, 600.0, 2.0 * restart_overhead)

        # If we had to start on-demand now, include one overhead if we're not already on OD.
        start_overhead_to_od = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else restart_overhead

        # Hard feasibility check: if we don't run OD now, we risk missing deadline.
        if remaining_time <= remaining_work + start_overhead_to_od + safety:
            self.mode = self.MODE_OD_LOCK

        # Estimate availability probability (smoothed)
        p_spot = (self._spot_steps + 1.0) / (self._steps + 2.0)
        p_spot = min(0.98, max(0.05, p_spot))

        # Transition from SPOT_WAIT to HYBRID if waiting-for-spot is likely too slow
        if self.mode == self.MODE_SPOT_WAIT:
            expected_total_time_if_wait_for_spot = remaining_work / p_spot
            # Add a small cushion for restarts and discretization.
            cushion = safety + 2.0 * restart_overhead
            if expected_total_time_if_wait_for_spot + cushion > remaining_time:
                self.mode = self.MODE_HYBRID
            else:
                # If we're currently in a long downtime, switch earlier.
                slack_if_od = remaining_time - (remaining_work + start_overhead_to_od)
                expected_remaining_down = max(
                    0.0, (self._avg_down_steps - float(self._down_streak)) * gap
                ) if not has_spot else 0.0
                if (not has_spot) and (slack_if_od < max(3600.0, expected_remaining_down + safety)):
                    self.mode = self.MODE_HYBRID

        # Transition to OD_LOCK if sufficiently close to end that spot volatility isn't worth it.
        if self.mode == self.MODE_HYBRID:
            # If we switched to OD and are close, keep OD locked.
            slack_if_od = remaining_time - (remaining_work + start_overhead_to_od)
            if slack_if_od < max(1800.0, safety + restart_overhead):
                self.mode = self.MODE_OD_LOCK

        # Decision based on mode
        if self.mode == self.MODE_OD_LOCK:
            return ClusterType.ON_DEMAND

        if self.mode == self.MODE_HYBRID:
            if not has_spot:
                return ClusterType.ON_DEMAND

            # has_spot == True
            if last_cluster_type == ClusterType.SPOT:
                return ClusterType.SPOT

            # Confirm spot isn't a one-step blip before switching from OD/NONE to SPOT.
            slack_if_od = remaining_time - (remaining_work + start_overhead_to_od)
            if self._up_streak >= self._hybrid_confirm_steps and slack_if_od > (safety + restart_overhead):
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # MODE_SPOT_WAIT
        if has_spot:
            return ClusterType.SPOT

        # If spot unavailable, either wait (NONE) or, if close enough, switch to OD.
        slack_if_od = remaining_time - (remaining_work + start_overhead_to_od)
        if slack_if_od < max(3600.0, safety + restart_overhead):
            self.mode = self.MODE_HYBRID
            return ClusterType.ON_DEMAND
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)