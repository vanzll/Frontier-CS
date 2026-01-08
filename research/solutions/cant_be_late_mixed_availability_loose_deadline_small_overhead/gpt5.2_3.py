import math
from collections import deque

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
        self.args = args

        self._initialized = False

        self._total_steps = 0
        self._spot_available_steps = 0

        self._spot_state = None
        self._spot_run_start_elapsed = 0.0

        self._off_to_on_transitions = 0
        self._transition_times = deque()

        self._window_seconds = 6 * 3600.0
        self._avail_window = deque()
        self._avail_window_sum = 0
        self._avail_window_maxlen = None

        self._ema_off = 3600.0
        self._ema_on = 1800.0
        self._ema_alpha = 0.2

        self._od_locked = False
        self._od_hold_until = 0.0
        self._od_min_run_seconds = 0.0

        self._spec_path = None

    def solve(self, spec_path: str) -> "Solution":
        self._spec_path = spec_path
        return self

    @staticmethod
    def _sum_task_done(task_done_time):
        if task_done_time is None:
            return 0.0
        if isinstance(task_done_time, (int, float)):
            return float(task_done_time)
        if isinstance(task_done_time, dict):
            total = 0.0
            for v in task_done_time.values():
                try:
                    total += float(v)
                except Exception:
                    pass
            return total
        if isinstance(task_done_time, (list, tuple)):
            if not task_done_time:
                return 0.0
            first = task_done_time[0]
            if isinstance(first, (int, float)):
                s = 0.0
                for x in task_done_time:
                    try:
                        s += float(x)
                    except Exception:
                        pass
                return s
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                s = 0.0
                for seg in task_done_time:
                    try:
                        a = float(seg[0])
                        b = float(seg[1])
                        if b > a:
                            s += (b - a)
                    except Exception:
                        continue
                return s
        return 0.0

    @staticmethod
    def _beta_mean_std(successes, trials, alpha0=2.0, beta0=2.0):
        a = alpha0 + float(successes)
        b = beta0 + float(trials - successes)
        denom = a + b
        if denom <= 0:
            return 0.0, 1.0
        mean = a / denom
        var = (a * b) / (denom * denom * (denom + 1.0))
        std = math.sqrt(max(0.0, var))
        return mean, std

    def _lazy_init(self):
        if self._initialized:
            return
        gap = float(getattr(self.env, "gap_seconds", 60.0)) if getattr(self, "env", None) is not None else 60.0
        self._avail_window_maxlen = max(10, int(self._window_seconds / max(gap, 1e-6)) + 2)
        self._od_min_run_seconds = max(15 * 60.0, 2.0 * gap, 4.0 * float(getattr(self, "restart_overhead", 0.0)))
        self._initialized = True

    def _update_spot_stats(self, elapsed, gap, has_spot):
        self._total_steps += 1
        if has_spot:
            self._spot_available_steps += 1

        bit = 1 if has_spot else 0
        self._avail_window.append(bit)
        self._avail_window_sum += bit
        if self._avail_window_maxlen is not None:
            while len(self._avail_window) > self._avail_window_maxlen:
                old = self._avail_window.popleft()
                self._avail_window_sum -= old

        if self._spot_state is None:
            self._spot_state = bool(has_spot)
            self._spot_run_start_elapsed = float(elapsed)
            if has_spot:
                self._off_to_on_transitions += 1
                self._transition_times.append(float(elapsed))
            return

        prev_state = self._spot_state
        cur_state = bool(has_spot)

        if cur_state != prev_state:
            run_len = float(elapsed) - float(self._spot_run_start_elapsed)
            if run_len < 0:
                run_len = 0.0

            if prev_state:
                self._ema_on = (1.0 - self._ema_alpha) * self._ema_on + self._ema_alpha * run_len
            else:
                self._ema_off = (1.0 - self._ema_alpha) * self._ema_off + self._ema_alpha * run_len

            self._spot_state = cur_state
            self._spot_run_start_elapsed = float(elapsed)

            if (not prev_state) and cur_state:
                self._off_to_on_transitions += 1
                self._transition_times.append(float(elapsed))

        cutoff = float(elapsed) - float(self._window_seconds)
        while self._transition_times and self._transition_times[0] < cutoff:
            self._transition_times.popleft()

        self._ema_off = min(max(self._ema_off, gap), 12 * 3600.0)
        self._ema_on = min(max(self._ema_on, gap), 12 * 3600.0)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._lazy_init()

        elapsed = float(self.env.elapsed_seconds)
        gap = float(self.env.gap_seconds)

        self._update_spot_stats(elapsed, gap, has_spot)

        done = self._sum_task_done(getattr(self, "task_done_time", None))
        task_duration = float(getattr(self, "task_duration", 0.0))
        deadline = float(getattr(self, "deadline", 0.0))
        restart_overhead = float(getattr(self, "restart_overhead", 0.0))

        remaining_work = max(0.0, task_duration - done)
        remaining_time = max(0.0, deadline - elapsed)

        if remaining_work <= 0.0:
            return ClusterType.NONE

        if remaining_time <= 0.0:
            return ClusterType.ON_DEMAND if not has_spot else ClusterType.SPOT

        slack = remaining_time - remaining_work

        target_slack = 6.0 * 3600.0
        if slack >= target_slack:
            k = 0.7
        elif slack <= 0.0:
            k = 2.7
        else:
            k = 0.7 + 2.0 * (1.0 - slack / target_slack)

        overall_mean, overall_std = self._beta_mean_std(
            self._spot_available_steps, self._total_steps, alpha0=2.0, beta0=2.0
        )
        overall_lower = max(0.0, min(1.0, overall_mean - k * overall_std))

        n_recent = len(self._avail_window)
        if n_recent > 0:
            recent_success = self._avail_window_sum
            recent_mean, recent_std = self._beta_mean_std(
                recent_success, n_recent, alpha0=1.0, beta0=1.0
            )
            recent_lower = max(0.0, min(1.0, recent_mean - (k + 0.2) * recent_std))
        else:
            recent_lower = overall_lower

        p_safe = min(overall_lower, recent_lower)

        time_so_far = max(elapsed, gap)
        global_start_rate = float(self._off_to_on_transitions) / max(time_so_far, 1e-6)
        recent_start_rate = float(len(self._transition_times)) / max(min(self._window_seconds, time_so_far), 1e-6)
        start_rate = max(global_start_rate, recent_start_rate)

        overhead_fraction = min(0.7, 1.2 * start_rate * restart_overhead)
        effective_p = max(0.0, p_safe - overhead_fraction)

        margin = max(2.0 * gap, 2.0 * restart_overhead)
        spot_only_capacity = effective_p * remaining_time

        need_od = remaining_work > (spot_only_capacity - margin)

        hedge_slack = max(30.0 * 60.0, 2.5 * self._ema_off + 2.0 * restart_overhead)

        hard_must_lock = remaining_time <= (remaining_work + 2.0 * restart_overhead + gap)
        if hard_must_lock or (slack <= hedge_slack):
            self._od_locked = True

        if self._od_locked:
            return ClusterType.ON_DEMAND

        if last_cluster_type == ClusterType.ON_DEMAND and elapsed < self._od_hold_until:
            return ClusterType.ON_DEMAND if (not has_spot) else ClusterType.ON_DEMAND

        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND:
                if (slack > max(6.0 * 3600.0, 4.0 * self._ema_off)) and (not need_od):
                    return ClusterType.SPOT
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        if last_cluster_type == ClusterType.ON_DEMAND:
            return ClusterType.ON_DEMAND

        if need_od:
            self._od_hold_until = elapsed + self._od_min_run_seconds
            return ClusterType.ON_DEMAND

        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)