from __future__ import annotations

import json
import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_adaptive_v1"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass

        self._steps = 0
        self._spot_steps = 0
        self._last_has_spot: Optional[bool] = None
        self._spot_stay = 0
        self._spot_lost = 0

        self._thresholds_inited = False
        self._slack_wait = 0.0
        self._slack_crit = 0.0
        self._switch_min_slack = 0.0
        self._runlen_threshold = 0.0
        self._min_od_run_sec = 0.0
        self._min_none_to_spot_sec = 0.0

        self._od_since: Optional[float] = None
        self._none_since: Optional[float] = None
        self._spot_since: Optional[float] = None

    def solve(self, spec_path: str) -> "Solution":
        try:
            with open(spec_path, "r", encoding="utf-8") as f:
                spec = json.load(f)
            cfg = spec.get("strategy_config", {}) if isinstance(spec, dict) else {}
            if isinstance(cfg, dict):
                name = cfg.get("name")
                if isinstance(name, str) and name:
                    self.NAME = name
        except Exception:
            pass
        return self

    @staticmethod
    def _is_num(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def _done_seconds(self) -> float:
        t = getattr(self, "task_done_time", None)
        if t is None:
            return 0.0
        if self._is_num(t):
            return float(t)
        # Common cases: list of segments, list of cumulative, list of (start,end)
        try:
            if isinstance(t, (list, tuple)):
                if not t:
                    return 0.0
                # If list of pairs -> sum durations
                pair_sum = 0.0
                pair_count = 0
                for x in t:
                    if isinstance(x, (list, tuple)) and len(x) >= 2 and self._is_num(x[0]) and self._is_num(x[1]):
                        pair_sum += max(0.0, float(x[1]) - float(x[0]))
                        pair_count += 1
                if pair_count == len(t) and pair_count > 0:
                    return pair_sum

                # If list of numbers: either durations or cumulative.
                nums = []
                all_nums = True
                for x in t:
                    if self._is_num(x):
                        nums.append(float(x))
                    else:
                        all_nums = False
                        break
                if all_nums and nums:
                    # Detect non-decreasing (likely cumulative)
                    nondecreasing = True
                    for i in range(len(nums) - 1):
                        if nums[i] > nums[i + 1] + 1e-9:
                            nondecreasing = False
                            break
                    if nondecreasing:
                        last = nums[-1]
                        # Favor cumulative interpretation if sum would be implausibly large
                        s = sum(nums)
                        td = float(getattr(self, "task_duration", 0.0) or 0.0)
                        if td > 0 and s > td * 1.2:
                            return min(last, td)
                        return min(last, td) if td > 0 else last
                    # Otherwise treat as segments/durations
                    return float(sum(nums))
        except Exception:
            pass
        return 0.0

    def _init_thresholds(self) -> None:
        if self._thresholds_inited:
            return
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        td = float(getattr(self, "task_duration", 0.0) or 0.0)
        dl = float(getattr(self, "deadline", 0.0) or 0.0)
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        base_slack = max(0.0, dl - td)
        # Reserve slack for overheads and avoid missing deadline.
        slack_crit = max(base_slack * 0.10, 20.0 * 60.0)
        slack_crit = max(slack_crit, 4.0 * ro + 2.0 * gap)

        slack_wait = max(base_slack * 0.35, 60.0 * 60.0)
        slack_wait = max(slack_wait, slack_crit + 30.0 * 60.0)

        # If slack is very small in a different setting, disable waiting.
        if base_slack > 0:
            slack_wait = min(slack_wait, max(0.0, base_slack - 2.0 * gap))
            slack_crit = min(slack_crit, max(0.0, base_slack - gap))

        self._slack_wait = float(slack_wait)
        self._slack_crit = float(slack_crit)

        self._switch_min_slack = max(self._slack_crit * 1.25, 2.0 * ro + 2.0 * gap)
        self._runlen_threshold = max(6.0 * ro, 20.0 * 60.0)
        self._min_od_run_sec = max(6.0 * gap, 20.0 * 60.0)
        self._min_none_to_spot_sec = max(3.0 * gap, 10.0 * 60.0)

        self._thresholds_inited = True

    def _est_spot_runlen_sec(self) -> float:
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        den = self._spot_stay + self._spot_lost
        if den <= 0:
            return 0.0
        # Beta(1,1) posterior mean for p_stay
        p_stay = (self._spot_stay + 1.0) / (den + 2.0)
        p_loss = max(1e-3, 1.0 - p_stay)
        runlen_steps = 1.0 / p_loss
        return runlen_steps * gap

    def _should_switch_to_spot(self, last_cluster_type: ClusterType, slack: float, elapsed: float) -> bool:
        if last_cluster_type == ClusterType.SPOT:
            return True

        if slack < self._switch_min_slack:
            return False

        # Avoid thrashing when spot flickers.
        runlen_sec = self._est_spot_runlen_sec()
        if runlen_sec > 0.0 and runlen_sec < self._runlen_threshold:
            return False

        if last_cluster_type == ClusterType.ON_DEMAND and self._od_since is not None:
            if elapsed - self._od_since < self._min_od_run_sec:
                return False

        if last_cluster_type == ClusterType.NONE and self._none_since is not None:
            if elapsed - self._none_since < self._min_none_to_spot_sec:
                return False

        return True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._init_thresholds()

        # Update online stats about spot trace.
        if self._last_has_spot is not None and self._last_has_spot:
            if has_spot:
                self._spot_stay += 1
            else:
                self._spot_lost += 1
        self._last_has_spot = bool(has_spot)
        self._steps += 1
        if has_spot:
            self._spot_steps += 1

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)

        done = self._done_seconds()
        if done >= task_duration - 1e-6:
            decision = ClusterType.NONE
        else:
            work_left = max(0.0, task_duration - done)
            time_left = max(0.0, deadline - elapsed)
            slack = time_left - work_left

            # Hard guard: if we don't run now, we may be unable to finish even on OD.
            start_od_overhead = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else ro
            hard_must_compute = time_left <= work_left + start_od_overhead + max(gap, 0.0)

            if hard_must_compute:
                if has_spot and last_cluster_type == ClusterType.SPOT and slack < 1.2 * ro:
                    decision = ClusterType.SPOT
                else:
                    decision = ClusterType.ON_DEMAND
            else:
                # Modes based on remaining slack:
                # - WAIT: use spot when available, else pause to save OD for later.
                # - FILL: never pause; use spot when available, else OD.
                # - CRITICAL: avoid switching overhead; prefer OD.
                if slack > self._slack_wait:
                    decision = ClusterType.SPOT if has_spot else ClusterType.NONE
                elif slack > self._slack_crit:
                    if has_spot:
                        decision = ClusterType.SPOT if self._should_switch_to_spot(last_cluster_type, slack, elapsed) else ClusterType.ON_DEMAND
                    else:
                        decision = ClusterType.ON_DEMAND
                else:
                    # Critical: keep progress and reduce preemption/switch risk.
                    if has_spot and last_cluster_type == ClusterType.SPOT and slack < 1.2 * ro:
                        decision = ClusterType.SPOT
                    else:
                        decision = ClusterType.ON_DEMAND

        if decision == ClusterType.SPOT and not has_spot:
            decision = ClusterType.ON_DEMAND

        # Update mode timing state (hysteresis helpers)
        if decision == ClusterType.ON_DEMAND:
            if last_cluster_type != ClusterType.ON_DEMAND:
                self._od_since = elapsed
            self._none_since = None
            self._spot_since = None
        elif decision == ClusterType.SPOT:
            if last_cluster_type != ClusterType.SPOT:
                self._spot_since = elapsed
            self._od_since = None
            self._none_since = None
        else:
            if last_cluster_type != ClusterType.NONE:
                self._none_since = elapsed
            self._od_since = None
            self._spot_since = None

        return decision

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)