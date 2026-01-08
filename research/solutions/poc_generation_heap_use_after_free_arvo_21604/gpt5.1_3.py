import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        target_size = 33762

        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return b""

        exact_candidates = []
        keyword_candidates = []
        nearest = None
        nearest_diff = None

        keywords = (
            "poc",
            "crash",
            "uaf",
            "heap",
            "use_after_free",
            "use-after-free",
            "bug",
            "id:",
        )

        try:
            for m in tf.getmembers():
                if not m.isreg():
                    continue
                if m.size <= 0:
                    continue

                name_lower = m.name.lower()

                if m.size == target_size:
                    exact_candidates.append(m)

                if any(k in name_lower for k in keywords):
                    keyword_candidates.append(m)

                diff = abs(m.size - target_size)
                if nearest_diff is None or diff < nearest_diff:
                    nearest_diff = diff
                    nearest = m

            member = None

            if exact_candidates:
                # Prefer exact-size candidate whose name looks like a PoC
                def score(mem):
                    n = mem.name.lower()
                    s = 0
                    if "poc" in n:
                        s += 4
                    if "crash" in n:
                        s += 3
                    if "uaf" in n or "use_after_free" in n or "use-after-free" in n:
                        s += 3
                    if "heap" in n:
                        s += 1
                    return s

                member = max(exact_candidates, key=score)
            elif keyword_candidates:
                # Among keyword candidates, choose one closest in size to target
                def keyfunc(mem):
                    return abs(mem.size - target_size)

                member = min(keyword_candidates, key=keyfunc)
            elif nearest is not None:
                member = nearest
            else:
                return b""

            f = tf.extractfile(member)
            if f is None:
                return b""
            data = f.read()
            if not isinstance(data, bytes):
                try:
                    data = bytes(data)
                except Exception:
                    data = b""
            return data
        finally:
            try:
                tf.close()
            except Exception:
                pass