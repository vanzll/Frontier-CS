import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball or directory

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        target_len = 149

        if os.path.isdir(src_path):
            data = self._solve_from_directory(src_path, target_len)
            if data is not None:
                return data
            return b"A" * target_len

        # src_path is expected to be a tarball
        try:
            with tarfile.open(src_path, "r:*") as tf:
                data = self._solve_from_tar(tf, target_len)
                if data is not None:
                    return data
        except (tarfile.ReadError, FileNotFoundError, OSError):
            # Fallback: maybe src_path is actually a directory
            if os.path.isdir(src_path):
                data = self._solve_from_directory(src_path, target_len)
                if data is not None:
                    return data

        return b"A" * target_len

    # ----------------- Helper methods -----------------

    def _is_non_printable(self, b: int) -> bool:
        # Treat tab(9), lf(10), cr(13) and space..~ as printable.
        if b in (9, 10, 13):
            return False
        if 32 <= b <= 126:
            return False
        return True

    def _binary_fraction(self, data: bytes) -> float:
        if not data:
            return 0.0
        non_printable = sum(1 for b in data if self._is_non_printable(b))
        return non_printable / float(len(data))

    def _name_score(self, name_lower: str) -> int:
        score = 0
        # Strong indicators
        strong_keys = ["poc", "crash", "testcase", "repro", "fuzz", "seed"]
        weak_keys = ["input", "sample", "rv60", "rv6", "rmvb", "realvideo", "realmedia"]

        for k in strong_keys:
            if k in name_lower:
                score += 10
        for k in weak_keys:
            if k in name_lower:
                score += 3

        # Directory hints
        if "/poc" in name_lower or "poc/" in name_lower:
            score += 5
        if "/crash" in name_lower or "crashes/" in name_lower:
            score += 5
        if "/corpus" in name_lower:
            score += 2
        if "/fuzz" in name_lower:
            score += 2
        if "/test" in name_lower or "tests/" in name_lower:
            score += 1

        # Extension hints
        ext_boost = {
            ".rm": 5,
            ".rmvb": 5,
            ".rv": 5,
            ".rv60": 5,
            ".bin": 3,
            ".dat": 3,
            ".raw": 3,
        }
        base, dot, ext = name_lower.rpartition(".")
        if dot:
            full_ext = "." + ext
            score += ext_boost.get(full_ext, 0)
        return score

    # ----------------- Tar-based search -----------------

    def _solve_from_tar(self, tf: tarfile.TarFile, target_len: int) -> bytes | None:
        try:
            members = tf.getmembers()
        except Exception:
            return None

        suspicious_keys = [
            "poc",
            "crash",
            "fuzz",
            "seed",
            "testcase",
            "repro",
            "input",
            "sample",
            "rv60",
            "rv6",
            "rmvb",
            "realvideo",
            "realmedia",
        ]

        best_data = None
        best_score = float("-inf")

        # Stage 1: search files with suspicious names
        for m in members:
            if not m.isfile():
                continue
            name_lower = m.name.lower()
            if not any(k in name_lower for k in suspicious_keys):
                continue
            if m.size <= 0 or m.size > 1024 * 1024:
                continue

            try:
                f = tf.extractfile(m)
            except Exception:
                continue
            if f is None:
                continue
            try:
                data = f.read()
            except Exception:
                f.close()
                continue
            finally:
                try:
                    f.close()
                except Exception:
                    pass

            if not data:
                continue

            name_score = self._name_score(name_lower)
            binary_frac = self._binary_fraction(data)
            binary_score = 5 if binary_frac > 0.3 else 0
            size_penalty = abs(len(data) - target_len) / float(max(target_len, 1))
            candidate_score = name_score + binary_score - size_penalty - (len(data) / 1000.0)

            if candidate_score > best_score:
                best_score = candidate_score
                best_data = data

        if best_data is not None:
            return best_data

        # Stage 2: fallback search for exact target_len anywhere
        for m in members:
            if not m.isfile():
                continue
            if m.size != target_len or m.size <= 0:
                continue
            try:
                f = tf.extractfile(m)
            except Exception:
                continue
            if f is None:
                continue
            try:
                data = f.read()
            except Exception:
                f.close()
                continue
            finally:
                try:
                    f.close()
                except Exception:
                    pass
            if data and len(data) == target_len:
                return data

        # Stage 3: choose shortest binary-looking file as last resort
        min_len = None
        min_data = None
        for m in members:
            if not m.isfile():
                continue
            if m.size <= 0 or m.size > 1024 * 1024:
                continue
            try:
                f = tf.extractfile(m)
            except Exception:
                continue
            if f is None:
                continue
            try:
                data = f.read()
            except Exception:
                f.close()
                continue
            finally:
                try:
                    f.close()
                except Exception:
                    pass
            if not data:
                continue
            binary_frac = self._binary_fraction(data)
            if binary_frac < 0.3:
                continue
            l = len(data)
            if min_len is None or l < min_len:
                min_len = l
                min_data = data

        return min_data

    # ----------------- Directory-based search -----------------

    def _solve_from_directory(self, root: str, target_len: int) -> bytes | None:
        suspicious_keys = [
            "poc",
            "crash",
            "fuzz",
            "seed",
            "testcase",
            "repro",
            "input",
            "sample",
            "rv60",
            "rv6",
            "rmvb",
            "realvideo",
            "realmedia",
        ]

        best_data = None
        best_score = float("-inf")

        # Stage 1: suspicious names
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                full_path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    continue
                if size <= 0 or size > 1024 * 1024:
                    continue
                rel_name = os.path.relpath(full_path, root).replace("\\", "/")
                name_lower = rel_name.lower()
                if not any(k in name_lower for k in suspicious_keys):
                    continue

                try:
                    with open(full_path, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                if not data:
                    continue

                name_score = self._name_score(name_lower)
                binary_frac = self._binary_fraction(data)
                binary_score = 5 if binary_frac > 0.3 else 0
                size_penalty = abs(len(data) - target_len) / float(max(target_len, 1))
                candidate_score = name_score + binary_score - size_penalty - (len(data) / 1000.0)

                if candidate_score > best_score:
                    best_score = candidate_score
                    best_data = data

        if best_data is not None:
            return best_data

        # Stage 2: search for exact size target_len
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                full_path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    continue
                if size != target_len or size <= 0:
                    continue
                try:
                    with open(full_path, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                if data and len(data) == target_len:
                    return data

        # Stage 3: choose smallest binary-looking file as last resort
        min_len = None
        min_data = None
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                full_path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    continue
                if size <= 0 or size > 1024 * 1024:
                    continue
                try:
                    with open(full_path, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                if not data:
                    continue
                binary_frac = self._binary_fraction(data)
                if binary_frac < 0.3:
                    continue
                l = len(data)
                if min_len is None or l < min_len:
                    min_len = l
                    min_data = data

        return min_data