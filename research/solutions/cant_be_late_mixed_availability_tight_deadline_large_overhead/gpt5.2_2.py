import json
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "safety_guard_v1"

    def __init__(self, args: Optional[Any] = None):
        super().__init__(args)
        self._inited = False

        self._committed_od = False

        self._commit_slack = 0.0
        self._max_wait_seconds = 0.0
        self._switchback_slack = 0.0
        self._switchback_min_remaining = 0.0
        self._min_od_steps_before_switchback = 1

        self._work_done_est = 0.0
        self._overhead_remaining = 0.0

        self._last_action: Optional[ClusterType] = None
        self._last_has_spot: Optional[bool] = None
        self._last_overhead_start = 0.0

        self._od_run_steps = 0

    def solve(self, spec_path: str) -> "Solution":
        # Optional: read configuration if present; defaults are robust.
        try:
            with open(spec_path, "r") as f:
                spec = json.load(f)
            _ = spec  # unused
        except Exception:
            pass
        return self

    def _lazy_init(self):
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        oh = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        task_dur = float(getattr(self, "task_duration", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        slack_total = max(0.0, deadline - task_dur)

        # Commit to on-demand when remaining slack is low (reserve for last-mile uncertainty).
        self._commit_slack = max(4.0 * oh, 12.0 * gap, 0.15 * slack_total, oh + 2.0 * gap)

        # When spot is unavailable and slack is still large, prefer waiting (free) up to this buffer.
        self._max_wait_seconds = max(0.0, min(2.0 * 3600.0, 0.5 * slack_total))

        # Switch back from OD to spot only when there is ample slack and enough remaining work.
        self._switchback_slack = self._commit_slack + 3.0 * oh + 12.0 * gap
        self._switchback_min_remaining = max(3.0 * 3600.0, 0.10 * task_dur)

        self._min_od_steps_before_switchback = max(1, int((oh / gap) + 1)) if gap > 0 else 1

        self._inited = True

    @staticmethod
    def _segment_work(seg: Any) -> float:
        try:
            if isinstance(seg, (int, float)):
                return float(seg)
            if isinstance(seg, (tuple, list)) and len(seg) >= 2:
                a = float(seg[0])
                b = float(seg[1])
                return max(0.0, b - a)
            if isinstance(seg, dict):
                for k in ("work", "duration", "seconds", "done", "progress"):
                    if k in seg and isinstance(seg[k], (int, float)):
                        return float(seg[k])
        except Exception:
            return 0.0
        return 0.0

    def _work_done_from_env(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            return 0.0

        task_dur = float(getattr(self, "task_duration", 0.0) or 0.0)
        try:
            if all(isinstance(x, (int, float)) for x in tdt):
                vals = [float(x) for x in tdt]
                s = sum(vals)
                mx = max(vals) if vals else 0.0
                # Heuristic: if values look like cumulative progress snapshots, use last.
                if vals and mx <= task_dur and s > 1.2 * task_dur:
                    return max(0.0, min(task_dur, vals[-1]))
                return max(0.0, min(task_dur, s))
        except Exception:
            pass

        total = 0.0
        for seg in tdt:
            total += self._segment_work(seg)
        return max(0.0, min(task_dur, total))

    def _finalize_previous_step_estimate(self):
        if self._last_action is None or self._last_has_spot is None:
            return

        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        if gap <= 0:
            return

        running = False
        if self._last_action == ClusterType.ON_DEMAND:
            running = True
        elif self._last_action == ClusterType.SPOT:
            running = bool(self._last_has_spot)

        if running:
            work_inc = max(0.0, gap - float(self._last_overhead_start))
            self._work_done_est += work_inc
            self._overhead_remaining = max(0.0, float(self._last_overhead_start) - gap)
        else:
            # No instance running: no progress; overhead does not elapse without a running instance.
            self._overhead_remaining = float(self._last_overhead_start)

        task_dur = float(getattr(self, "task_duration", 0.0) or 0.0)
        if task_dur > 0:
            self._work_done_est = max(0.0, min(task_dur, self._work_done_est))

    def _compute_remaining(self) -> float:
        task_dur = float(getattr(self, "task_duration", 0.0) or 0.0)
        done_env = self._work_done_from_env()
        done_est = float(self._work_done_est)
        done = max(done_env, done_est)
        return max(0.0, task_dur - done)

    def _should_switchback_to_spot(self, remaining_work: float, slack_left: float) -> bool:
        if self._committed_od:
            return False
        if remaining_work < self._switchback_min_remaining:
            return False
        if slack_left < self._switchback_slack:
            return False
        if self._od_run_steps < self._min_od_steps_before_switchback:
            return False
        return True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._inited:
            self._lazy_init()

        # Update our internal estimate of work done during the previous step.
        self._finalize_previous_step_estimate()

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        time_left = max(0.0, deadline - elapsed)

        remaining_work = self._compute_remaining()
        if remaining_work <= 0.0:
            action = ClusterType.NONE
            self._last_action = action
            self._last_has_spot = bool(has_spot)
            self._last_overhead_start = float(self._overhead_remaining)
            self._od_run_steps = 0
            return action

        slack_left = time_left - remaining_work

        # If slack is exhausted (or negative), commit immediately to OD.
        if slack_left <= 0.0:
            self._committed_od = True

        if self._committed_od:
            action = ClusterType.ON_DEMAND
        else:
            if has_spot:
                if slack_left <= self._commit_slack:
                    self._committed_od = True
                    action = ClusterType.ON_DEMAND
                else:
                    if last_cluster_type == ClusterType.ON_DEMAND:
                        if self._should_switchback_to_spot(remaining_work, slack_left):
                            action = ClusterType.SPOT
                        else:
                            action = ClusterType.ON_DEMAND
                    else:
                        action = ClusterType.SPOT
            else:
                # Spot unavailable; never return SPOT.
                if slack_left <= self._commit_slack:
                    self._committed_od = True
                    action = ClusterType.ON_DEMAND
                else:
                    if last_cluster_type == ClusterType.ON_DEMAND:
                        action = ClusterType.ON_DEMAND
                    else:
                        if slack_left >= self._commit_slack + self._max_wait_seconds:
                            action = ClusterType.NONE
                        else:
                            action = ClusterType.ON_DEMAND

        # Enforce validity rule.
        if action == ClusterType.SPOT and not has_spot:
            action = ClusterType.ON_DEMAND

        # Determine overhead at the start of this step (for next-step finalization).
        overhead_start = float(self._overhead_remaining)
        if action in (ClusterType.SPOT, ClusterType.ON_DEMAND):
            start_new = False
            if last_cluster_type != action:
                start_new = True
            elif last_cluster_type == ClusterType.NONE:
                start_new = True
            if start_new:
                overhead_start = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        self._last_action = action
        self._last_has_spot = bool(has_spot)
        self._last_overhead_start = overhead_start

        if action == ClusterType.ON_DEMAND:
            self._od_run_steps += 1
        else:
            self._od_run_steps = 0

        return action

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)