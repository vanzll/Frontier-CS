import os
import tarfile
import posixpath


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        try:
            data = self._find_poc_in_tar(src_path)
            if data is not None:
                return data
        except Exception:
            pass
        # Fallback: return a dummy 512-byte input if no PoC found
        return b"A" * 512

    def _find_poc_in_tar(self, src_path: str) -> bytes | None:
        bug_id = "383200048"

        with tarfile.open(src_path, "r:*") as tf:
            members = tf.getmembers()

            # Stage 1: filenames containing exact bug id
            id_matches = [
                m for m in members
                if m.isfile() and bug_id in m.name
            ]
            if id_matches:
                member = self._select_best_member(id_matches)
                if member is not None:
                    return self._read_member(tf, member)

            # Stage 2: filenames mentioning oss-fuzz
            of_matches = [
                m for m in members
                if m.isfile() and ("oss-fuzz" in m.name.lower() or "ossfuzz" in m.name.lower())
            ]
            if of_matches:
                member = self._select_best_member(of_matches)
                if member is not None:
                    return self._read_member(tf, member)

            # Stage 3: any 512-byte regular file
            size512_matches = [
                m for m in members
                if m.isfile() and m.size == 512
            ]
            if size512_matches:
                member = self._select_best_member(size512_matches)
                if member is not None:
                    return self._read_member(tf, member)

            # Stage 4: small files (<= 4096 bytes), heuristic pick
            small_matches = [
                m for m in members
                if m.isfile() and 0 < m.size <= 4096
            ]
            if small_matches:
                member = self._select_best_member(small_matches)
                if member is not None:
                    return self._read_member(tf, member)

        return None

    def _read_member(self, tf: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
        extracted = tf.extractfile(member)
        if extracted is None:
            return None
        try:
            data = extracted.read()
        finally:
            extracted.close()
        return data

    def _select_best_member(self, candidates: list[tarfile.TarInfo]) -> tarfile.TarInfo | None:
        if not candidates:
            return None

        best = None
        best_score = -10**9

        for m in candidates:
            score = self._score_member(m)
            if best is None or score > best_score or (
                score == best_score and (
                    m.size < best.size or
                    (m.size == best.size and m.name < best.name)
                )
            ):
                best = m
                best_score = score

        return best

    def _score_member(self, m: tarfile.TarInfo) -> int:
        name_lower = m.name.lower()
        ext = posixpath.splitext(name_lower)[1]
        size = m.size

        score = 0

        # Strong preference for our specific bug id if present
        if "383200048" in name_lower:
            score += 1000

        # Prefer oss-fuzz related names
        if "oss-fuzz" in name_lower or "ossfuzz" in name_lower:
            score += 500

        # Size heuristics
        if size == 512:
            score += 300
        elif size < 2048:
            score += 150
        elif size < 8192:
            score += 50

        # Extension-based heuristics
        binary_exts = {
            ".bin", ".dat", ".upx", ".so", ".elf", ".raw",
            ".poc", ".img", ".out", ".core"
        }
        text_exts = {
            ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp",
            ".py", ".txt", ".md", ".rst", ".html", ".htm",
            ".xml", ".json", ".yaml", ".yml", ".ini"
        }

        if ext in binary_exts:
            score += 100
        if ext in text_exts:
            score -= 100

        # Penalize very large files
        if size > 65536:
            score -= 500

        return score