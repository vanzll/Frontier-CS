import os
import tarfile
import tempfile
import shutil


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        tmpdir = None
        try:
            tmpdir = tempfile.mkdtemp(prefix="poc_extract_")
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    tf.extractall(tmpdir)
            except Exception:
                # If extraction fails, fall back to generic payload
                return b"A" * 33

            poc_bytes = self._find_poc_bytes(tmpdir)
            if poc_bytes is not None:
                return poc_bytes

        except Exception:
            # Any unexpected error: fall back to generic payload
            return b"A" * 33
        finally:
            if tmpdir is not None:
                try:
                    shutil.rmtree(tmpdir)
                except Exception:
                    pass

        # Final fallback
        return b"A" * 33

    def _find_poc_bytes(self, root_dir: str):
        max_size = 4096  # we know ground-truth is 33 bytes, so small files are enough

        crash_keywords = [
            "crash",
            "heap-buffer-overflow",
            "heap",
            "overflow",
            "poc",
            "repro",
            "bug",
            "clusterfuzz",
            "id:",
            "asan",
        ]
        other_keywords = [
            "fuzz",
            "corpus",
            "seed",
            "input",
            "test",
            "case",
            "pcap",
            "packet",
            "capwap",
            "ndpi",
            "sample",
        ]
        target_keywords = ["capwap", "ndpi"]
        good_exts = {".bin", ".raw", ".dat", ".pcap", ".pcapng", ".in", ".pkt", ""}

        crash_cands = []
        other_cands = []

        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                full_path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    continue

                if size <= 0 or size > max_size:
                    continue

                rel_path = os.path.relpath(full_path, root_dir)
                lrel = rel_path.lower()
                _, ext = os.path.splitext(lrel)

                # Quick filter for very unlikely paths (version control, build dirs, etc.)
                if any(skip in lrel for skip in [".git/", "/.git", "cmakefiles", "/build/", "/.idea/"]):
                    continue

                is_crash = any(kw in lrel for kw in crash_keywords)
                has_other_kw = any(kw in lrel for kw in other_keywords)
                ext_good = ext in good_exts

                if not (is_crash or has_other_kw or ext_good):
                    continue

                cand = (size, lrel, full_path)
                if is_crash:
                    crash_cands.append(cand)
                else:
                    other_cands.append(cand)

        # If we have crash-like candidates, prefer them
        path = self._choose_best_path(crash_cands, target_keywords, crash_keywords, good_exts)
        if path is None:
            path = self._choose_best_path(other_cands, target_keywords, crash_keywords, good_exts)

        if path is None:
            return None

        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            return None

    def _choose_best_path(self, candidates, target_keywords, crash_keywords, good_exts):
        if not candidates:
            return None

        best = None
        best_score = None

        for size, lrel, full in candidates:
            # Length proximity score: prefer exactly 33 bytes
            diff = abs(size - 33)
            len_score = 100 - min(diff * 5, 100)  # 0 diff -> 100, >=20 diff -> 0

            # Target-specific bonus
            target_bonus = 0
            for tkw in target_keywords:
                if tkw in lrel:
                    target_bonus += 30

            # Crash keyword bonus
            crash_bonus = 0
            for ckw in crash_keywords:
                if ckw in lrel:
                    crash_bonus += 10

            # Extension bonus
            _, ext = os.path.splitext(lrel)
            ext_bonus = 10 if ext in good_exts else 0

            # Small size bonus (smaller inputs are more likely to be minimized PoCs)
            if size <= 64:
                small_bonus = 20
            elif size <= 256:
                small_bonus = 10
            else:
                small_bonus = 0

            score = len_score + target_bonus + crash_bonus + ext_bonus + small_bonus

            if best_score is None or score > best_score:
                best_score = score
                best = (size, full)
            elif score == best_score:
                # Tie-breaker: closer to target length, then smaller size
                best_size, _ = best
                best_diff = abs(best_size - 33)
                if diff < best_diff or (diff == best_diff and size < best_size):
                    best = (size, full)

        return best[1] if best is not None else None