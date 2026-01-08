from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "safe_wait_spot_deadline"

    def __init__(self, args):
        super().__init__(args)
        self._init_done = False
        self._committed_to_od = False
        self._force_all_on_demand = False
        self._commit_margin = 0.0
        self._slack_total = 0.0
        self._tdt_cache_len = 0
        self._tdt_done_est = 0.0
        self._args = args

    def solve(self, spec_path: str) -> "Solution":
        # No offline precomputation needed for this heuristic.
        return self

    def _ensure_init(self):
        if self._init_done:
            return

        # Total slack available between task duration and hard deadline.
        slack_total = float(max(self.deadline - self.task_duration, 0.0))
        self._slack_total = slack_total

        # Restart overhead and environment step size.
        restart_overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)

        # Minimal safety margin must at least cover one restart plus
        # a couple of steps for discretization / estimation error.
        min_safe_margin = restart_overhead + 2.0 * gap

        if slack_total <= 0.0 or slack_total <= min_safe_margin * 1.1:
            # Almost no slack: always run on on-demand to maximize chance of finishing.
            self._force_all_on_demand = True
            self._commit_margin = max(min_safe_margin, slack_total)
        else:
            # Commit margin: when (time_left - remaining_work_est) drops below this,
            # we permanently switch to on-demand.
            base_target = max(min_safe_margin, 0.3 * slack_total)
            max_target = 0.7 * slack_total
            self._commit_margin = min(max(base_target, min_safe_margin), max_target)
            self._force_all_on_demand = False

        self._init_done = True

    def _update_progress_estimate(self) -> float:
        """
        Estimate total effective compute time done so far.
        The estimate is intentionally conservative (never exceeds true progress).
        """
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            return 0.0

        try:
            n = len(tdt)
        except TypeError:
            # Unexpected structure; fall back to 0 progress (safe).
            return 0.0

        if n < self._tdt_cache_len:
            # Environment reset its tracking structure; recompute from scratch.
            self._tdt_cache_len = 0
            self._tdt_done_est = 0.0

        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)

        for i in range(self._tdt_cache_len, n):
            seg = tdt[i]
            dur = 0.0
            try:
                if isinstance(seg, (int, float)):
                    dur = float(seg)
                elif isinstance(seg, (list, tuple)):
                    if len(seg) >= 2:
                        try:
                            dur = float(seg[1]) - float(seg[0])
                        except Exception:
                            dur = 0.0
                    elif len(seg) == 1:
                        try:
                            dur = float(seg[0])
                        except Exception:
                            dur = 0.0
                else:
                    try:
                        dur = float(getattr(seg, "duration", 0.0))
                    except Exception:
                        dur = 0.0
            except Exception:
                dur = 0.0

            if dur < 0.0:
                dur = 0.0
            # Clamp to at most one step to avoid ever overestimating work.
            if gap > 0.0 and dur > gap:
                dur = gap

            self._tdt_done_est += dur

        self._tdt_cache_len = n
        return self._tdt_done_est

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._ensure_init()

        # If task is logically complete (environment may still call _step),
        # choose NONE to avoid unnecessary cost.
        done_est = self._update_progress_estimate()
        remaining_work_est = max(self.task_duration - done_est, 0.0)
        if remaining_work_est <= 0.0:
            return ClusterType.NONE

        time_left = float(self.deadline - self.env.elapsed_seconds)

        # If already past the deadline, keep running on on-demand
        # (nothing can fix the miss now, but don't make things worse).
        if time_left <= 0.0:
            return ClusterType.ON_DEMAND

        if self._force_all_on_demand:
            return ClusterType.ON_DEMAND

        margin_est = time_left - remaining_work_est  # slack estimate

        # Once we commit to on-demand, we never go back to spot/none.
        if self._committed_to_od or margin_est <= self._commit_margin:
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        # Pre-commit phase:
        # - Use spot whenever available (cheap compute, tolerate preemptions).
        # - Otherwise, wait (NONE) to save cost, consuming some slack.
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)