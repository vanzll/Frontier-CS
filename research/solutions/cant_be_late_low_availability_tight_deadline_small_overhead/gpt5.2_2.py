import math
from typing import Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args=None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass

        self._initialized = False

        self._done_sum = 0.0
        self._done_len = 0
        self._prev_done_sum = 0.0

        self._total_steps = 0
        self._spot_steps = 0

        self._last_has_spot: Optional[bool] = None
        self._avail_streak = 0
        self._unavail_streak = 0
        self._ema_avail_steps = 0.0
        self._ema_unavail_steps = 0.0
        self._ema_alpha = 0.15

        self._panic = False

        self._wait_budget_seconds = 0.0

        self._k_switch_to_spot = 1
        self._min_od_runtime = 0.0
        self._min_expected_spot_streak_seconds = 0.0
        self._max_wait_outage_seconds = 0.0

        self._last_switch_to_od_elapsed: Optional[float] = None
        self._last_switch_to_spot_elapsed: Optional[float] = None

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _init_if_needed(self):
        if self._initialized:
            return
        gap = float(getattr(self.env, "gap_seconds", 60.0))
        overhead = float(getattr(self, "restart_overhead", 0.0))
        slack_init = float(getattr(self, "deadline", 0.0)) - float(getattr(self, "task_duration", 0.0))
        if slack_init < 0:
            slack_init = 0.0

        # Conservative initial streak estimates (in steps): assume outages can be long.
        self._ema_unavail_steps = max(10.0, (2.0 * 3600.0) / max(gap, 1e-9))  # ~2 hours
        self._ema_avail_steps = max(5.0, (0.5 * 3600.0) / max(gap, 1e-9))  # ~30 minutes

        # Switch back to spot only after it has been continuously available long enough.
        self._k_switch_to_spot = max(1, int(math.ceil(overhead / max(gap, 1e-9))))

        # Minimum on-demand runtime before considering switching back to spot.
        self._min_od_runtime = max(10.0 * 60.0, 5.0 * overhead)  # >=10 min

        # Only switch to spot if expected spot streak is "worth it".
        self._min_expected_spot_streak_seconds = max(15.0 * 60.0, 3.0 * overhead)  # >=15 min

        # Only wait (pause) during spot outage if predicted remaining outage is short.
        self._max_wait_outage_seconds = 20.0 * 60.0  # 20 minutes

        # Small wait budget (use some slack to avoid OD during brief outages).
        self._wait_budget_seconds = min(slack_init * 0.2, 3600.0)  # up to 1 hour, typically ~48 min

        self._initialized = True

    def _update_done_sum(self):
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            self._done_sum = 0.0
            self._done_len = 0
            return
        n = len(tdt)
        if n < self._done_len:
            # In case evaluator resets list, fall back to full sum.
            self._done_sum = float(sum(tdt))
            self._done_len = n
            return
        if n == self._done_len:
            return
        s = self._done_sum
        for i in range(self._done_len, n):
            s += float(tdt[i])
        self._done_sum = s
        self._done_len = n

    def _update_spot_stats(self, has_spot: bool):
        self._total_steps += 1
        if has_spot:
            self._spot_steps += 1

        if self._last_has_spot is None:
            self._last_has_spot = has_spot
            if has_spot:
                self._avail_streak = 1
                self._unavail_streak = 0
            else:
                self._avail_streak = 0
                self._unavail_streak = 1
            return

        if self._last_has_spot and not has_spot:
            # Availability streak ended.
            if self._avail_streak > 0:
                self._ema_avail_steps = (1.0 - self._ema_alpha) * self._ema_avail_steps + self._ema_alpha * float(
                    self._avail_streak
                )
            self._avail_streak = 0
            self._unavail_streak = 1
        elif (not self._last_has_spot) and has_spot:
            # Unavailability streak ended.
            if self._unavail_streak > 0:
                self._ema_unavail_steps = (1.0 - self._ema_alpha) * self._ema_unavail_steps + self._ema_alpha * float(
                    self._unavail_streak
                )
            self._unavail_streak = 0
            self._avail_streak = 1
        else:
            if has_spot:
                self._avail_streak += 1
            else:
                self._unavail_streak += 1

        self._last_has_spot = has_spot

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._init_if_needed()

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        gap = float(getattr(self.env, "gap_seconds", 60.0))
        overhead = float(getattr(self, "restart_overhead", 0.0))

        self._update_done_sum()
        done_sum = self._done_sum
        remaining_work = float(getattr(self, "task_duration", 0.0)) - done_sum
        if remaining_work <= 0.0:
            return ClusterType.NONE

        self._update_spot_stats(bool(has_spot))

        time_left = float(getattr(self, "deadline", 0.0)) - elapsed
        safety = max(2.0 * gap, overhead)

        # Detect if we are likely in restart overhead (no progress while supposedly running).
        delta_done = done_sum - self._prev_done_sum
        self._prev_done_sum = done_sum
        likely_overhead_in_progress = (last_cluster_type != ClusterType.NONE) and (delta_done <= 1e-9)

        # Conservative "can still finish with OD from now" check.
        need_if_od_now = remaining_work + (overhead if last_cluster_type != ClusterType.ON_DEMAND else 0.0)
        if time_left <= need_if_od_now + safety:
            self._panic = True

        if self._panic:
            if last_cluster_type != ClusterType.ON_DEMAND:
                self._last_switch_to_od_elapsed = elapsed
            return ClusterType.ON_DEMAND

        # If we appear to be in overhead, avoid switching cluster types (switching resets overhead).
        if likely_overhead_in_progress:
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            if last_cluster_type == ClusterType.SPOT and has_spot:
                return ClusterType.SPOT
            # If spot vanished mid-overhead, fall through to outage handling.

        if has_spot:
            if last_cluster_type == ClusterType.SPOT:
                return ClusterType.SPOT

            if last_cluster_type == ClusterType.ON_DEMAND:
                od_started = self._last_switch_to_od_elapsed if self._last_switch_to_od_elapsed is not None else elapsed
                od_runtime = elapsed - od_started

                expected_spot_streak_seconds = float(self._ema_avail_steps) * gap
                slack = time_left - remaining_work  # optimistic slack ignoring future overheads

                if (
                    self._avail_streak >= self._k_switch_to_spot
                    and od_runtime >= self._min_od_runtime
                    and expected_spot_streak_seconds >= self._min_expected_spot_streak_seconds
                    and slack > overhead + safety
                ):
                    self._last_switch_to_spot_elapsed = elapsed
                    return ClusterType.SPOT
                return ClusterType.ON_DEMAND

            # last_cluster_type == NONE (or unknown): start spot when available
            self._last_switch_to_spot_elapsed = elapsed
            return ClusterType.SPOT

        # No spot available
        slack_if_keep_progressing = time_left - (
            remaining_work + (overhead if last_cluster_type != ClusterType.ON_DEMAND else 0.0)
        )

        exp_additional_unavail_steps = max(float(self._ema_unavail_steps) - float(self._unavail_streak), 0.0)
        exp_additional_unavail_seconds = exp_additional_unavail_steps * gap

        # Decide whether to wait for spot to return (NONE) or run on-demand.
        can_wait = (
            self._wait_budget_seconds > 0.0
            and slack_if_keep_progressing > exp_additional_unavail_seconds + safety + overhead
            and exp_additional_unavail_seconds <= self._max_wait_outage_seconds
        )

        if can_wait:
            self._wait_budget_seconds = max(0.0, self._wait_budget_seconds - gap)
            return ClusterType.NONE

        if last_cluster_type != ClusterType.ON_DEMAND:
            self._last_switch_to_od_elapsed = elapsed
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)