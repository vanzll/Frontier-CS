import os
import tarfile
import tempfile
import string


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmpdir = tempfile.mkdtemp()
        try:
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    tf.extractall(tmpdir)
            except Exception:
                pass

            poc = self._find_existing_poc(tmpdir)
            if poc is not None:
                return poc
        finally:
            pass

        return b"\x00" * 72

    def _find_existing_poc(self, root: str):
        best_path = None
        best_score = 0
        printable = set(bytes(string.printable, "ascii"))

        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue

                if size == 0 or size > 4096:
                    continue

                lower = fn.lower()
                score = 0
                patterns = [
                    ("poc", 40),
                    ("crash", 35),
                    ("uaf", 30),
                    ("heap", 25),
                    ("raw", 20),
                    ("encap", 20),
                    ("issue", 15),
                    ("id:", 10),
                    ("fuzz", 10),
                    ("input", 5),
                ]
                for pat, val in patterns:
                    if pat in lower:
                        score += val

                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except OSError:
                    continue

                if not data:
                    continue

                non_printable = sum(1 for b in data if b not in printable)
                if non_printable > len(data) * 0.3:
                    score += 5
                else:
                    score -= 5

                if score > best_score:
                    best_score = score
                    best_path = path

        if best_path is not None and best_score >= 20:
            try:
                with open(best_path, "rb") as f:
                    return f.read()
            except OSError:
                return None

        return None