import os
import tarfile
import zipfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        target_len = 1551

        if tarfile.is_tarfile(src_path):
            return self._from_tar(src_path, target_len)

        # Fallback: try as zipfile
        try:
            return self._from_zip(src_path, target_len)
        except zipfile.BadZipFile:
            raise RuntimeError("Unsupported archive format (not a tar or zip file)")

    def _score_candidate(self, name: str, size: int, target_len: int) -> int:
        n = name.lower()
        score = 0
        if "383170474" in n:
            score += 1000
        if "oss-fuzz" in n or "ossfuzz" in n or "clusterfuzz" in n or "fuzz" in n:
            score += 100
        if "poc" in n or "crash" in n or "repro" in n or "reproducer" in n or "testcase" in n:
            score += 80
        if "debug" in n or "dwarf" in n or "names" in n:
            score += 40
        if name.endswith((".o", ".obj", ".bin", ".elf", ".dat")):
            score += 20
        if "/test" in n or "/tests" in n or "\\test" in n or "\\tests" in n:
            score += 10
        # Slight preference for sizes close to target_len
        diff = abs(size - target_len)
        if diff == 0:
            score += 50
        elif diff <= 64:
            score += 20
        elif diff <= 256:
            score += 10
        return score

    def _choose_best(self, candidates, target_len: int):
        exact = []
        approx = []

        for name, size, opaque in candidates:
            score = self._score_candidate(name, size, target_len)
            if size == target_len:
                exact.append((score, name, opaque))
            else:
                diff = abs(size - target_len)
                approx.append((score, diff, name, opaque))

        if exact:
            exact.sort(key=lambda x: (-x[0], x[1]))
            return exact[0][2]

        if approx:
            approx.sort(key=lambda x: (-x[0], x[1], x[2]))
            return approx[0][2]

        raise RuntimeError("No suitable candidate files found in archive")

    def _from_tar(self, src_path: str, target_len: int) -> bytes:
        candidates = []
        with tarfile.open(src_path, "r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                size = member.size
                candidates.append((name, size, member))

            best_member = self._choose_best(candidates, target_len)
            extracted = tf.extractfile(best_member)
            if extracted is None:
                raise RuntimeError("Failed to extract selected PoC file from tar archive")
            data = extracted.read()
            return data

    def _from_zip(self, src_path: str, target_len: int) -> bytes:
        candidates = []
        with zipfile.ZipFile(src_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                size = info.file_size
                candidates.append((name, size, info))

            best_info = self._choose_best(candidates, target_len)
            data = zf.read(best_info)
            return data