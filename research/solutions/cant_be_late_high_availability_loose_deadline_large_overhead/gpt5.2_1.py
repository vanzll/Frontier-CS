import math
import numbers
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args: Any = None):
        super().__init__(args)
        self._prev_elapsed: Optional[float] = None
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self._total_steps = 0
        self._spot_steps = 0
        self._transitions = 0
        self._spot_losses = 0

        self._last_has_spot: Optional[bool] = None
        self._spot_streak = 0
        self._cur_no_spot_run = 0
        self._ema_outage_steps: Optional[float] = None

        self._od_start_time: Optional[float] = None

        self._td_mode: str = "unknown"  # "segments", "cumulative", "intervals", "unknown"
        self._td_cache_len: int = 0
        self._td_cache_done: float = 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _is_number(x: Any) -> bool:
        return isinstance(x, numbers.Real) and math.isfinite(float(x))

    def _maybe_reset_on_new_episode(self) -> None:
        try:
            elapsed = float(self.env.elapsed_seconds)
        except Exception:
            return
        if self._prev_elapsed is None:
            self._prev_elapsed = elapsed
            return
        if elapsed + 1e-9 < self._prev_elapsed:
            self._reset_episode_state()
        self._prev_elapsed = elapsed

    def _update_spot_stats(self, has_spot: bool) -> None:
        self._total_steps += 1
        if has_spot:
            self._spot_steps += 1

        if self._last_has_spot is not None:
            if self._last_has_spot != has_spot:
                self._transitions += 1
            if self._last_has_spot and (not has_spot):
                self._spot_losses += 1

        if has_spot:
            self._spot_streak += 1
            if self._last_has_spot is False:
                ended_len = self._cur_no_spot_run
                if ended_len > 0:
                    if self._ema_outage_steps is None:
                        self._ema_outage_steps = float(ended_len)
                    else:
                        self._ema_outage_steps = 0.8 * self._ema_outage_steps + 0.2 * float(ended_len)
                self._cur_no_spot_run = 0
        else:
            self._spot_streak = 0
            self._cur_no_spot_run += 1

        self._last_has_spot = has_spot

    def _detect_td_mode_numeric(self, vals: list[float], gap: float) -> str:
        n = len(vals)
        if n >= 5:
            base = max(1e-9, max(vals[: min(3, n)]))
            if vals[-1] > max(3.0 * gap, 4.0 * base):
                return "cumulative"
        return "segments"

    def _compute_done_work(self) -> float:
        td = getattr(self, "task_done_time", None)
        if td is None:
            return 0.0

        if self._is_number(td):
            return max(0.0, float(td))

        if not isinstance(td, (list, tuple)) or len(td) == 0:
            return 0.0

        gap = 0.0
        try:
            gap = float(self.env.gap_seconds)
        except Exception:
            gap = 0.0

        if self._td_mode == "unknown":
            first = td[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2 and self._is_number(first[0]) and self._is_number(first[1]):
                self._td_mode = "intervals"
                self._td_cache_len = 0
                self._td_cache_done = 0.0
            elif self._is_number(first):
                vals = []
                for x in td[: min(8, len(td))]:
                    if not self._is_number(x):
                        vals = []
                        break
                    vals.append(float(x))
                if vals:
                    if gap <= 0:
                        gap = vals[0] if vals[0] > 0 else 1.0
                    self._td_mode = self._detect_td_mode_numeric(vals + ([float(td[-1])] if self._is_number(td[-1]) else []), gap)
                    self._td_cache_len = 0
                    self._td_cache_done = 0.0
                else:
                    self._td_mode = "unknown"
                    self._td_cache_len = 0
                    self._td_cache_done = 0.0

        if self._td_mode == "cumulative":
            last = td[-1]
            if self._is_number(last):
                return max(0.0, float(last))
            self._td_mode = "unknown"

        if self._td_mode == "segments":
            cur_len = len(td)
            if cur_len < self._td_cache_len:
                self._td_cache_len = 0
                self._td_cache_done = 0.0

            if self._td_cache_len == 0:
                done = 0.0
                for x in td:
                    if not self._is_number(x):
                        self._td_mode = "unknown"
                        return 0.0
                    done += max(0.0, float(x))
                self._td_cache_len = cur_len
                self._td_cache_done = done
                return done

            done = self._td_cache_done
            for i in range(self._td_cache_len, cur_len):
                x = td[i]
                if not self._is_number(x):
                    self._td_mode = "unknown"
                    return 0.0
                done += max(0.0, float(x))
            self._td_cache_len = cur_len
            self._td_cache_done = done
            return done

        if self._td_mode == "intervals":
            cur_len = len(td)
            if cur_len < self._td_cache_len:
                self._td_cache_len = 0
                self._td_cache_done = 0.0

            done = self._td_cache_done
            for i in range(self._td_cache_len, cur_len):
                seg = td[i]
                if not isinstance(seg, (list, tuple)) or len(seg) < 2 or (not self._is_number(seg[0])) or (not self._is_number(seg[1])):
                    self._td_mode = "unknown"
                    return done
                a = float(seg[0])
                b = float(seg[1])
                done += max(0.0, b - a)
            self._td_cache_len = cur_len
            self._td_cache_done = done
            return done

        # Fallback for unknown formats: try best-effort
        first = td[0]
        if self._is_number(first):
            done = 0.0
            for x in td:
                if self._is_number(x):
                    done += max(0.0, float(x))
            return done
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            done = 0.0
            for seg in td:
                if isinstance(seg, (list, tuple)) and len(seg) >= 2 and self._is_number(seg[0]) and self._is_number(seg[1]):
                    done += max(0.0, float(seg[1]) - float(seg[0]))
            return done
        return 0.0

    def _expected_remaining_outage_seconds(self, gap: float) -> float:
        ema = self._ema_outage_steps
        if ema is None or not math.isfinite(ema) or ema <= 0:
            ema = 3.0
        rem_steps = max(0.0, float(ema) - float(self._cur_no_spot_run))
        return rem_steps * max(0.0, gap)

    def _compute_need_slack(self, remaining_time: float, gap: float, overhead: float, elapsed: float) -> float:
        elapsed = max(elapsed, max(gap, 1.0))
        loss_rate = float(self._spot_losses) / elapsed  # losses per second
        pred_losses = loss_rate * max(0.0, remaining_time)
        pred_overhead = pred_losses * max(0.0, overhead)

        base = 3.0 * max(0.0, overhead) + 2.0 * max(0.0, gap)
        p_est = (float(self._spot_steps) + 1.0) / (float(self._total_steps) + 2.0)
        if p_est < 0.55:
            base += (0.5 - p_est) * 6.0 * overhead

        # Add some margin for volatility (frequent transitions)
        if self._total_steps > 0:
            trans_rate = float(self._transitions) / float(self._total_steps)
            base += trans_rate * 6.0 * overhead

        return pred_overhead + base

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._maybe_reset_on_new_episode()
        self._update_spot_stats(has_spot)

        done = self._compute_done_work()
        task_duration = float(getattr(self, "task_duration", 0.0))
        deadline = float(getattr(self, "deadline", 0.0))
        overhead = float(getattr(self, "restart_overhead", 0.0))

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        gap = float(getattr(self.env, "gap_seconds", 0.0))
        if gap <= 0:
            gap = 300.0

        remaining_work = max(0.0, task_duration - max(0.0, done))
        if remaining_work <= 1e-9:
            self._od_start_time = None
            return ClusterType.NONE

        remaining_time = max(0.0, deadline - elapsed)
        if remaining_time <= 1e-9:
            return ClusterType.ON_DEMAND if has_spot or (not has_spot) else ClusterType.ON_DEMAND

        slack = remaining_time - remaining_work

        need_slack = self._compute_need_slack(remaining_time, gap, overhead, elapsed)
        critical = (slack <= need_slack) or (remaining_time <= remaining_work + need_slack)

        switch_back_streak = max(2, int(round(overhead / max(gap, 1e-6))) + 1)
        od_min_run = max(3600.0, 3.0 * overhead, 4.0 * gap)

        choice: ClusterType

        if critical:
            choice = ClusterType.ON_DEMAND
        else:
            if not has_spot:
                if last_cluster_type == ClusterType.ON_DEMAND:
                    choice = ClusterType.ON_DEMAND
                else:
                    extra_slack = slack - need_slack
                    exp_rem_out = self._expected_remaining_outage_seconds(gap)
                    if extra_slack >= exp_rem_out + gap:
                        choice = ClusterType.NONE
                    else:
                        choice = ClusterType.ON_DEMAND
            else:
                if last_cluster_type == ClusterType.ON_DEMAND:
                    if self._od_start_time is not None and (elapsed - self._od_start_time) < od_min_run:
                        choice = ClusterType.ON_DEMAND
                    else:
                        extra_slack = slack - need_slack
                        if self._spot_streak >= switch_back_streak and extra_slack >= max(2.0 * overhead, 2.0 * gap):
                            choice = ClusterType.SPOT
                        elif extra_slack >= max(6.0 * overhead, 6.0 * gap) and self._spot_streak >= 1:
                            choice = ClusterType.SPOT
                        else:
                            choice = ClusterType.ON_DEMAND
                else:
                    choice = ClusterType.SPOT

        if (not has_spot) and choice == ClusterType.SPOT:
            choice = ClusterType.ON_DEMAND

        if choice == ClusterType.ON_DEMAND and last_cluster_type != ClusterType.ON_DEMAND:
            self._od_start_time = elapsed
        if choice != ClusterType.ON_DEMAND:
            if last_cluster_type == ClusterType.ON_DEMAND and choice == ClusterType.SPOT:
                # keep od_start_time; it is used only while on-demand, but no harm
                pass

        return choice

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)