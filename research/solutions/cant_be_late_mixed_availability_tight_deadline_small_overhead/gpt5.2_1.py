import math
from collections import deque

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "adaptive_slack_spot_v1"

    def __init__(self, args=None):
        super().__init__(args)
        self._inited = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _quantile(data, q: float):
        if not data:
            return None
        s = sorted(float(x) for x in data)
        if len(s) == 1:
            return s[0]
        q = 0.0 if q < 0.0 else (1.0 if q > 1.0 else q)
        pos = q * (len(s) - 1)
        i = int(pos)
        frac = pos - i
        if i + 1 < len(s):
            return s[i] * (1.0 - frac) + s[i + 1] * frac
        return s[i]

    def _ensure_init(self):
        if self._inited:
            return
        self._inited = True

        self._prev_has_spot = None
        self._avail_streak_steps = 0
        self._unavail_streak_steps = 0
        self._avail_hist_seconds = deque(maxlen=60)
        self._unavail_hist_seconds = deque(maxlen=60)

        self._switch_times = deque()
        self._od_lock = False
        self._od_lock_until = 0.0

        self._total_slack_seconds = None

    def _estimate_done_seconds(self) -> float:
        # Prefer task_done_time, which is provided by the environment.
        tdt = getattr(self, "task_done_time", None)
        if isinstance(tdt, (list, tuple)) and len(tdt) > 0:
            try:
                first = tdt[0]
                if isinstance(first, (list, tuple)) and len(first) == 2:
                    total = 0.0
                    for seg in tdt:
                        if not (isinstance(seg, (list, tuple)) and len(seg) == 2):
                            continue
                        a, b = seg
                        try:
                            a = float(a)
                            b = float(b)
                        except Exception:
                            continue
                        if b > a:
                            total += (b - a)
                    return total
                vals = [float(x) for x in tdt]
                if len(vals) == 1:
                    return vals[0]
                mono = True
                for i in range(len(vals) - 1):
                    if vals[i] > vals[i + 1] + 1e-9:
                        mono = False
                        break
                return vals[-1] if mono else sum(vals)
            except Exception:
                pass

        for attr in ("task_done_seconds", "task_completed_seconds", "done_seconds", "completed_seconds"):
            v = getattr(self, attr, None)
            if isinstance(v, (int, float)):
                return float(v)

        # Last resort: assume any elapsed time is productive (conservative for meeting deadline logic).
        try:
            return float(min(getattr(self.env, "elapsed_seconds", 0.0), getattr(self, "task_duration", 0.0)))
        except Exception:
            return 0.0

    def _update_spot_streaks(self, has_spot: bool):
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)

        if self._prev_has_spot is None:
            self._prev_has_spot = bool(has_spot)
            if has_spot:
                self._avail_streak_steps = 1
                self._unavail_streak_steps = 0
            else:
                self._avail_streak_steps = 0
                self._unavail_streak_steps = 1
            return

        if has_spot:
            if self._prev_has_spot is False and self._unavail_streak_steps > 0:
                self._unavail_hist_seconds.append(self._unavail_streak_steps * gap)
                self._unavail_streak_steps = 0
                self._avail_streak_steps = 0
            self._avail_streak_steps += 1
        else:
            if self._prev_has_spot is True and self._avail_streak_steps > 0:
                self._avail_hist_seconds.append(self._avail_streak_steps * gap)
                self._avail_streak_steps = 0
                self._unavail_streak_steps = 0
            self._unavail_streak_steps += 1

        self._prev_has_spot = bool(has_spot)

    def _predicted_remaining_outage_seconds(self) -> float:
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        if gap <= 0:
            gap = 1.0

        # Conservative typical outage (p80), bounded below by one step.
        base = self._quantile(self._unavail_hist_seconds, 0.80)
        if base is None:
            base = gap

        current = float(self._unavail_streak_steps) * gap
        total_pred = max(base, current)
        remaining = total_pred - current
        return max(gap, remaining)

    def _expected_avail_streak_seconds(self) -> float:
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        if gap <= 0:
            gap = 1.0
        med = self._quantile(self._avail_hist_seconds, 0.50)
        if med is None:
            med = 3.0 * gap
        current = float(self._avail_streak_steps) * gap
        return max(med, current, gap)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._ensure_init()
        self._update_spot_streaks(bool(has_spot))

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        if gap <= 0:
            gap = 1.0

        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        restart_overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        done = self._estimate_done_seconds()
        if done < 0:
            done = 0.0
        if task_duration > 0:
            done = min(done, task_duration)

        remaining_work = max(0.0, task_duration - done)
        time_left = max(0.0, deadline - elapsed)

        if remaining_work <= 1e-6:
            return ClusterType.NONE

        if self._total_slack_seconds is None:
            self._total_slack_seconds = max(0.0, deadline - task_duration)

        # Slack remaining equals total_slack - waste_used, where waste_used = elapsed - done.
        slack_now = self._total_slack_seconds - (elapsed - done)

        critical = max(6.0 * restart_overhead, 4.0 * gap, 900.0)

        # If we are close to consuming all slack, lock into OD.
        if slack_now <= critical or time_left <= (remaining_work + critical):
            self._od_lock = True

        # Maintain OD for a minimum time once started to avoid excessive restart overhead.
        if elapsed < self._od_lock_until:
            self._od_lock = True

        # Track switching rate and lock into OD if too volatile and slack is shrinking.
        while self._switch_times and self._switch_times[0] < elapsed - 3600.0:
            self._switch_times.popleft()
        if len(self._switch_times) >= 14 and slack_now < 3.0 * 3600.0:
            self._od_lock = True

        choice = None

        if self._od_lock:
            choice = ClusterType.ON_DEMAND
        else:
            if has_spot:
                expected_avail = self._expected_avail_streak_seconds()
                min_streak_to_switch = max(5.0 * restart_overhead, 3.0 * gap, 600.0)
                switch_guard = max(3.0 * restart_overhead, 1.0 * gap, 180.0)

                if last_cluster_type == ClusterType.ON_DEMAND:
                    # Avoid switching near the end or when spot is likely to be too short-lived.
                    if slack_now < switch_guard:
                        choice = ClusterType.ON_DEMAND
                    elif expected_avail < min_streak_to_switch and slack_now < 2.0 * 3600.0:
                        choice = ClusterType.ON_DEMAND
                    else:
                        choice = ClusterType.SPOT
                else:
                    choice = ClusterType.SPOT
            else:
                predicted_outage_rem = self._predicted_remaining_outage_seconds()
                wait_extra = max(2.0 * restart_overhead, 1.0 * gap, 180.0)

                # Only wait if we have meaningful slack and can likely absorb the remaining outage.
                if slack_now > (predicted_outage_rem + wait_extra) and slack_now > 1800.0:
                    choice = ClusterType.NONE
                else:
                    choice = ClusterType.ON_DEMAND

        # Enforce API requirement: never choose SPOT when unavailable.
        if (not has_spot) and choice == ClusterType.SPOT:
            choice = ClusterType.ON_DEMAND

        # Update switch stats + OD min-run lock.
        if choice != last_cluster_type:
            self._switch_times.append(elapsed)
            if choice == ClusterType.ON_DEMAND and last_cluster_type != ClusterType.ON_DEMAND:
                min_run = max(1800.0, 10.0 * restart_overhead, 6.0 * gap)
                self._od_lock_until = max(self._od_lock_until, elapsed + min_run)

        # If slack recovers (rare), allow unlocking OD. Keep conservative: only unlock with plenty of slack.
        if self._od_lock and slack_now > 3.5 * 3600.0 and elapsed >= self._od_lock_until:
            self._od_lock = False

        return choice

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)