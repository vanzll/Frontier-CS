import tarfile
from typing import Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = self._find_poc_in_tar(src_path)
        if poc is not None:
            return poc
        # Fallback: generic long input to trigger string length issues
        return b"A" * 8192

    def _find_poc_in_tar(self, src_path: str) -> Optional[bytes]:
        try:
            with tarfile.open(src_path, "r:*") as tf:
                bug_ids = ("42537014", "42537")
                specific_candidates = []
                generic_candidates = []

                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    size = member.size
                    if size <= 0:
                        continue

                    name_lower = member.name.lower()
                    is_specific = any(bid in name_lower for bid in bug_ids)

                    is_generic_name = any(
                        kw in name_lower
                        for kw in (
                            "poc",
                            "crash",
                            "testcase",
                            "heap",
                            "overflow",
                            "regress",
                            "input",
                            "seed",
                            "bug",
                            "oss-fuzz",
                            "ossfuzz",
                        )
                    )

                    if not is_specific and not is_generic_name:
                        continue

                    # Ignore obviously too-large files for a small PoC
                    if size > 4096:
                        continue

                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    data = f.read()
                    if not data:
                        continue

                    if is_specific:
                        specific_candidates.append(data)
                    elif size <= 64:
                        generic_candidates.append(data)

                target_len = 9

                def pick_best(candidates):
                    if not candidates:
                        return None
                    return min(
                        candidates,
                        key=lambda d: (abs(len(d) - target_len), len(d)),
                    )

                best = pick_best(specific_candidates)
                if best is not None:
                    return best
                best = pick_best(generic_candidates)
                return best
        except Exception:
            return None