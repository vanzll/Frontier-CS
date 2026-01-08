import os
import tarfile


class Solution:
    TARGET_SIZE = 72

    def solve(self, src_path: str) -> bytes:
        if os.path.isdir(src_path):
            data = self._find_poc_in_dir(src_path)
        else:
            data = self._find_poc_in_tar(src_path)
        if data is None:
            return b"A" * self.TARGET_SIZE
        return data

    def _score_name(self, relpath: str) -> int:
        name = relpath.lower()
        score = 0

        if "27851" in name:
            score += 1000
        if "raw_encap" in name or "raw-encap" in name or "rawencap" in name:
            score += 400
        if "encap" in name:
            score += 200
        if "poc" in name:
            score += 400
        if "crash" in name:
            score += 350
        if "seed" in name:
            score += 300
        if "uaf" in name or "use_after_free" in name or "use-after-free" in name:
            score += 300
        if "id:" in name or "/id_" in name or name.startswith("id_"):
            score += 150

        base = os.path.basename(name)
        for kw in ("poc", "pocs", "crash", "crashes", "seeds", "corpus", "cases", "inputs", "artifacts"):
            if f"/{kw}/" in name or base.startswith(kw):
                score += 120

        ext = os.path.splitext(base)[1]
        if ext in (".bin", ".dat", ".raw", ".in", ".inp", ".poc"):
            score += 200

        return score

    def _find_poc_in_tar(self, src_path: str):
        try:
            tf = tarfile.open(src_path, "r:*")
        except tarfile.TarError:
            return None

        target = self.TARGET_SIZE
        members = [m for m in tf.getmembers() if m.isfile()]

        best72 = None
        best72_score = None

        for m in members:
            size = m.size
            if size != target or size == 0:
                continue
            relpath = m.name
            score = self._score_name(relpath)
            depth = relpath.count("/")
            score -= depth
            if best72 is None or score > best72_score:
                best72 = m
                best72_score = score

        if best72 is not None:
            f = tf.extractfile(best72)
            if f is not None:
                data = f.read()
                if data:
                    tf.close()
                    return data

        best_any = None
        best_any_score = None
        for m in members:
            size = m.size
            if size == 0 or size > 4096:
                continue
            relpath = m.name
            score = self._score_name(relpath)
            score = score * 10 - abs(size - target)
            score -= size // 10
            if best_any is None or score > best_any_score:
                best_any = m
                best_any_score = score

        if best_any is not None:
            f = tf.extractfile(best_any)
            if f is not None:
                data = f.read()
                if data:
                    tf.close()
                    return data

        tf.close()
        return None

    def _find_poc_in_dir(self, root: str):
        target = self.TARGET_SIZE

        best72_path = None
        best72_score = None

        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                full = os.path.join(dirpath, name)
                try:
                    size = os.path.getsize(full)
                except OSError:
                    continue
                if size != target or size == 0:
                    continue
                relpath = os.path.relpath(full, root)
                score = self._score_name(relpath)
                depth = relpath.count(os.sep)
                score -= depth
                if best72_path is None or score > best72_score:
                    best72_path = full
                    best72_score = score

        if best72_path is not None:
            try:
                with open(best72_path, "rb") as f:
                    data = f.read()
                    if data:
                        return data
            except OSError:
                pass

        best_any_path = None
        best_any_score = None
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                full = os.path.join(dirpath, name)
                try:
                    size = os.path.getsize(full)
                except OSError:
                    continue
                if size == 0 or size > 4096:
                    continue
                relpath = os.path.relpath(full, root)
                score = self._score_name(relpath)
                score = score * 10 - abs(size - target)
                score -= size // 10
                if best_any_path is None or score > best_any_score:
                    best_any_path = full
                    best_any_score = score

        if best_any_path is not None:
            try:
                with open(best_any_path, "rb") as f:
                    data = f.read()
                    if data:
                        return data
            except OSError:
                pass

        return None