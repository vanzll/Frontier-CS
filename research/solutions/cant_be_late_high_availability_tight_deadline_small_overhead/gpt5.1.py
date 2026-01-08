from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_switch_heuristic"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialized_policy = False
        self._use_always_od = False
        self._idle_slack_threshold = 0.0
        self._od_switch_slack = 0.0
        self._initial_slack = 0.0
        self._seg_cache_len = None
        self._done_cache = 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)

    def _init_policy(self):
        # Initialize scheduling thresholds once env is available.
        try:
            task_dur = float(self.task_duration)
            deadline = float(self.deadline)
            restart_overhead = float(self.restart_overhead)
        except Exception:
            # If anything goes wrong, fall back to always using on-demand.
            self._use_always_od = True
            self._initialized_policy = True
            return

        gap = getattr(self.env, "gap_seconds", 60.0)
        initial_slack = max(0.0, deadline - task_dur)
        self._initial_slack = initial_slack

        if initial_slack <= 0.0:
            # No slack: safest is to always use on-demand.
            self._use_always_od = True
            self._idle_slack_threshold = 0.0
            self._od_switch_slack = 0.0
            self._initialized_policy = True
            return

        self._use_always_od = False

        # Commit to on-demand once we've burned about half our slack.
        base_switch = 0.5 * initial_slack
        # Ensure switch threshold is comfortably above a couple of restart+gap intervals.
        min_switch = 2.0 * (restart_overhead + gap)
        od_switch = max(base_switch, min_switch)
        if od_switch > initial_slack:
            od_switch = initial_slack
        self._od_switch_slack = od_switch

        # Only idle when slack is very high (~75% of initial slack left).
        idle_th = 0.75 * initial_slack
        if idle_th < od_switch:
            idle_th = od_switch
        self._idle_slack_threshold = idle_th

        self._initialized_policy = True

    def _segment_duration(self, seg) -> float:
        # Robustly extract duration from various possible segment representations.
        if isinstance(seg, (int, float)):
            try:
                return float(seg)
            except (TypeError, ValueError):
                return 0.0

        # Sequence-like [start, end]
        if hasattr(seg, "__len__") and not isinstance(seg, (str, bytes)):
            try:
                l = len(seg)
            except TypeError:
                l = 0
            if l >= 2:
                try:
                    s = float(seg[0])
                    e = float(seg[1])
                    return max(0.0, e - s)
                except (TypeError, ValueError):
                    pass

        # Object-like with start/end
        if hasattr(seg, "start") and hasattr(seg, "end"):
            try:
                s = float(seg.start)
                e = float(seg.end)
                return max(0.0, e - s)
            except (TypeError, ValueError):
                pass

        # Object-like with duration
        if hasattr(seg, "duration"):
            try:
                return float(seg.duration)
            except (TypeError, ValueError):
                pass

        return 0.0

    def _compute_done_time(self) -> float:
        segs = getattr(self, "task_done_time", None)

        if segs is None:
            self._seg_cache_len = None
            self._done_cache = 0.0
            return 0.0

        # If env exposes a simple numeric progress.
        if isinstance(segs, (int, float)):
            try:
                total = float(segs)
            except (TypeError, ValueError):
                total = 0.0
            self._seg_cache_len = None
            self._done_cache = total
            return total

        # Try list-like structure with caching by length.
        try:
            n = len(segs)
            indexable = True
        except TypeError:
            indexable = False
            n = None

        if not indexable:
            # Fallback: iterate fully each time.
            total = 0.0
            try:
                for seg in segs:
                    total += self._segment_duration(seg)
            except TypeError:
                total = 0.0
            self._seg_cache_len = None
            self._done_cache = total
            return total

        # Indexable sequence.
        if self._seg_cache_len is None or self._seg_cache_len > n:
            # No cache or list reset; recompute.
            total = 0.0
            for i in range(n):
                total += self._segment_duration(segs[i])
            self._seg_cache_len = n
            self._done_cache = total
            return total
        else:
            # Only sum newly added segments.
            total = self._done_cache
            for i in range(self._seg_cache_len, n):
                total += self._segment_duration(segs[i])
            self._seg_cache_len = n
            self._done_cache = total
            return total

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._initialized_policy:
            self._init_policy()

        # If policy initialization failed, default to safe behavior.
        if self._use_always_od:
            # Respect availability: never choose SPOT when unavailable.
            return ClusterType.ON_DEMAND

        # Compute remaining work.
        try:
            done = self._compute_done_time()
            task_dur = float(self.task_duration)
        except Exception:
            # If we cannot compute progress, behave conservatively.
            if has_spot:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        remaining = task_dur - done
        if remaining <= 0.0:
            return ClusterType.NONE

        # Time left to deadline.
        try:
            elapsed = float(self.env.elapsed_seconds)
            deadline = float(self.deadline)
        except Exception:
            elapsed = getattr(self.env, "elapsed_seconds", 0.0)
            deadline = getattr(self, "deadline", 0.0)

        time_left = deadline - elapsed
        if time_left <= 0.0:
            # Deadline already passed; nothing to do but use OD.
            return ClusterType.ON_DEMAND

        slack = time_left - remaining

        # Once slack is at or below the switch threshold, lock into on-demand.
        if slack <= self._od_switch_slack:
            return ClusterType.ON_DEMAND

        # Enough slack to prefer spot when available.
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable: decide between idling and on-demand.
        if slack > self._idle_slack_threshold:
            return ClusterType.NONE
        else:
            return ClusterType.ON_DEMAND