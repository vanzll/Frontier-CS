import math
from typing import Any, Optional

try:
    from sky_spot.strategies.strategy import Strategy
    from sky_spot.utils import ClusterType
except Exception:  # pragma: no cover
    class Strategy:
        def __init__(self, *args, **kwargs):
            self.env = None

    class _CT:
        SPOT = "SPOT"
        ON_DEMAND = "ON_DEMAND"
        NONE = "NONE"

    ClusterType = _CT()


class Solution(Strategy):
    NAME = "deadline_guard_v1"

    def __init__(self, args: Any = None):
        try:
            super().__init__(args)
        except Exception:
            try:
                super().__init__()
            except Exception:
                pass

        self._last_elapsed: float = -1.0

        # Spot availability stats (per episode)
        self._total_steps: int = 0
        self._spot_steps: int = 0
        self._transitions_up: int = 0
        self._prev_has_spot: Optional[bool] = None

        # Run-length stats (seconds, EMA)
        self._up_run_steps: int = 0
        self._down_run_steps: int = 0
        self._ema_up_run_s: Optional[float] = None
        self._ema_down_run_s: Optional[float] = None

        # Decision state
        self._od_lock: bool = False

        # Task done cache
        self._td_id: int = 0
        self._td_prefix_len: int = 0
        self._td_prefix_sum: float = 0.0

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _reset_episode(self) -> None:
        self._last_elapsed = -1.0

        self._total_steps = 0
        self._spot_steps = 0
        self._transitions_up = 0
        self._prev_has_spot = None

        self._up_run_steps = 0
        self._down_run_steps = 0
        self._ema_up_run_s = None
        self._ema_down_run_s = None

        self._od_lock = False

        self._td_id = 0
        self._td_prefix_len = 0
        self._td_prefix_sum = 0.0

    @staticmethod
    def _ema(old: Optional[float], new: float, alpha: float = 0.2) -> float:
        if old is None:
            return float(new)
        return (1.0 - alpha) * old + alpha * float(new)

    def _update_spot_stats(self, has_spot: bool, gap: float) -> bool:
        transition_up = False
        if self._prev_has_spot is not None and (not self._prev_has_spot) and has_spot:
            self._transitions_up += 1
            transition_up = True

        self._total_steps += 1
        if has_spot:
            self._spot_steps += 1

        # Run-length bookkeeping
        if self._prev_has_spot is None:
            if has_spot:
                self._up_run_steps = 1
                self._down_run_steps = 0
            else:
                self._down_run_steps = 1
                self._up_run_steps = 0
        else:
            if has_spot:
                if self._prev_has_spot:
                    self._up_run_steps += 1
                else:
                    # down run ended
                    if self._down_run_steps > 0:
                        self._ema_down_run_s = self._ema(self._ema_down_run_s, self._down_run_steps * gap, 0.2)
                    self._down_run_steps = 0
                    self._up_run_steps = 1
            else:
                if not self._prev_has_spot:
                    self._down_run_steps += 1
                else:
                    # up run ended
                    if self._up_run_steps > 0:
                        self._ema_up_run_s = self._ema(self._ema_up_run_s, self._up_run_steps * gap, 0.2)
                    self._up_run_steps = 0
                    self._down_run_steps = 1

        self._prev_has_spot = has_spot
        return transition_up

    def _seg_duration(self, seg: Any, elapsed: float) -> float:
        try:
            if seg is None:
                return 0.0
            if isinstance(seg, (int, float)):
                v = float(seg)
                return v if v >= 0.0 else 0.0
            if isinstance(seg, dict):
                if "duration" in seg:
                    v = float(seg["duration"])
                    return v if v >= 0.0 else 0.0
                if "start" in seg and "end" in seg:
                    s = float(seg["start"])
                    e = seg["end"]
                    if e is None:
                        e = elapsed
                    e = float(e)
                    d = e - s
                    return d if d > 0.0 else 0.0
                if "begin" in seg and "finish" in seg:
                    s = float(seg["begin"])
                    e = seg["finish"]
                    if e is None:
                        e = elapsed
                    e = float(e)
                    d = e - s
                    return d if d > 0.0 else 0.0
                return 0.0
            if isinstance(seg, (tuple, list)):
                if len(seg) >= 2:
                    s = float(seg[0])
                    e = seg[1]
                    if e is None:
                        e = elapsed
                    e = float(e)
                    d = e - s
                    return d if d > 0.0 else 0.0
                if len(seg) == 1:
                    v = float(seg[0])
                    return v if v >= 0.0 else 0.0
            return 0.0
        except Exception:
            return 0.0

    def _get_done_seconds(self, elapsed: float) -> float:
        td = getattr(self, "task_done_time", None)
        if td is None:
            return 0.0
        if isinstance(td, (int, float)):
            v = float(td)
            if v < 0.0:
                return 0.0
            return min(v, max(0.0, elapsed))

        if not isinstance(td, list):
            return 0.0

        # If it's a numeric list, it might be durations OR cumulative done values.
        # Use a heuristic each call (cheap; length small in practice).
        if len(td) == 0:
            return 0.0
        all_numeric = True
        for x in td:
            if not isinstance(x, (int, float)):
                all_numeric = False
                break
        if all_numeric:
            vals = [float(x) for x in td]
            mx = max(vals) if vals else 0.0
            s = sum(vals) if vals else 0.0
            # If values look cumulative (nondecreasing and sum much larger than max), take max.
            nondecreasing = True
            for i in range(1, len(vals)):
                if vals[i] + 1e-9 < vals[i - 1]:
                    nondecreasing = False
                    break
            if nondecreasing and (mx > 0.0) and (s > 1.5 * mx):
                done = mx
            else:
                done = s
            if done < 0.0:
                done = 0.0
            return min(done, max(0.0, elapsed))

        # Tuple/dict segment list: cache prefix excluding potentially-updated last segment.
        cur_id = id(td)
        if cur_id != self._td_id:
            self._td_id = cur_id
            self._td_prefix_len = 0
            self._td_prefix_sum = 0.0

        n = len(td)
        if n == 1:
            done = self._seg_duration(td[0], elapsed)
            return min(max(0.0, done), max(0.0, elapsed))

        # Ensure cached prefix covers elements [0 .. n-2]
        target_prefix_len = n - 1
        if self._td_prefix_len > target_prefix_len:
            # List shrank or replaced content unexpectedly; reset cache.
            self._td_prefix_len = 0
            self._td_prefix_sum = 0.0

        if self._td_prefix_len < target_prefix_len:
            for i in range(self._td_prefix_len, target_prefix_len):
                self._td_prefix_sum += self._seg_duration(td[i], elapsed)
            self._td_prefix_len = target_prefix_len

        done = self._td_prefix_sum + self._seg_duration(td[-1], elapsed)
        if done < 0.0:
            done = 0.0
        return min(done, max(0.0, elapsed))

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        gap = float(getattr(self.env, "gap_seconds", 60.0))

        if self._last_elapsed >= 0.0 and elapsed + 1e-9 < self._last_elapsed:
            self._reset_episode()
        self._last_elapsed = elapsed

        transition_up = self._update_spot_stats(has_spot, gap)

        task_duration = float(getattr(self, "task_duration", 0.0))
        deadline = float(getattr(self, "deadline", task_duration))
        restart = float(getattr(self, "restart_overhead", 0.0))

        done = self._get_done_seconds(elapsed)
        if done > task_duration:
            done = task_duration
        remaining = task_duration - done
        if remaining <= 1e-9:
            return ClusterType.NONE

        time_left = deadline - elapsed
        if time_left <= 1e-9:
            return ClusterType.ON_DEMAND

        slack_remaining = time_left - remaining  # how much non-progress time we can still afford

        # Hard feasibility / urgency locks
        urgent_lock_slack = max(1200.0, restart + 2.0 * gap)  # >= 20 min, or enough to cover immediate restart risk
        soft_margin = max(2700.0, 2.0 * restart + 6.0 * gap)  # >= 45 min, conservative buffer

        if self._od_lock:
            return ClusterType.ON_DEMAND

        if slack_remaining <= 0.0:
            self._od_lock = True
            return ClusterType.ON_DEMAND

        # Conservative spot effectiveness estimate
        # Beta prior favors spot early to avoid premature OD lock.
        a, b = 8.0, 2.0
        p_hat = (self._spot_steps + a) / (self._total_steps + a + b)
        # Estimate fraction of time "lost" to restarts when using spot:
        r_up = self._transitions_up / max(elapsed, gap)  # events per second
        p_eff = p_hat - r_up * restart
        if p_eff < 0.02:
            p_eff = 0.02
        if p_eff > 1.0:
            p_eff = 1.0

        expected_finish_spot = elapsed + (remaining / p_eff)

        # If spot uptimes are typically shorter than restart overhead, spot is risky.
        if self._ema_up_run_s is not None and self._ema_up_run_s < max(0.75 * restart, gap):
            # Don't lock immediately unless we must, but treat spot as low value.
            # This will be picked up by expected_finish_spot and slack thresholds.
            pass

        if slack_remaining <= urgent_lock_slack:
            self._od_lock = True
            return ClusterType.ON_DEMAND

        if expected_finish_spot > deadline - soft_margin:
            self._od_lock = True
            return ClusterType.ON_DEMAND

        # Not locked: prefer waiting for spot to avoid OD costs.
        if has_spot:
            # If spot just came up and historically runs are short, wait one step to confirm stability,
            # only when we have enough slack to spend.
            if (
                transition_up
                and last_cluster_type != ClusterType.SPOT
                and slack_remaining > soft_margin + gap
                and (
                    self._ema_up_run_s is not None
                    and self._ema_up_run_s < max(2.0 * restart, 3.0 * gap)
                )
            ):
                return ClusterType.NONE
            return ClusterType.SPOT

        return ClusterType.NONE