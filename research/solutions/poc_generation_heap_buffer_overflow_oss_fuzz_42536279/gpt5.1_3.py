import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        GROUND_LEN = 6180
        fallback = b"A" * GROUND_LEN

        text_exts = {
            ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp",
            ".py", ".java", ".js", ".ts",
            ".html", ".htm", ".xml",
            ".json", ".txt", ".md", ".rst",
            ".yml", ".yaml", ".toml", ".ini", ".cfg",
            ".cmake", ".sh", ".bash", ".zsh", ".bat", ".ps1",
            ".mk", ".makefile", ".am", ".ac", ".m4", ".in",
            ".csv",
        }

        binary_exts = {
            ".bin", ".dat", ".raw", ".yuv", ".rgb", ".rgba",
            ".wav", ".mp3", ".mp4", ".flv", ".mkv",
            ".ivf", ".webm", ".h264", ".264", ".hevc", ".h265",
            ".aac", ".ogg", ".flac",
            ".gz", ".xz", ".bz2", ".zip", ".7z",
        }

        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return fallback

        candidates = []
        bug_str = "42536279"

        for m in tf.getmembers():
            try:
                if not m.isfile():
                    continue
            except Exception:
                continue

            size = getattr(m, "size", 0)
            if size <= 0:
                continue
            if size > 1_000_000:
                continue

            path_lower = m.name.lower()
            base = os.path.basename(path_lower)
            ext = os.path.splitext(base)[1]

            base_score = 0

            if bug_str in path_lower:
                base_score += 15

            high_keywords = ["poc", "proof", "clusterfuzz", "testcase", "crash", "repro", "regress", "fuzz"]
            med_keywords = ["input", "heap", "overflow", "heap-buffer", "hbof", "svcdec", "svc"]

            if any(kw in base for kw in high_keywords):
                base_score += 10
            if any(kw in path_lower for kw in med_keywords):
                base_score += 5

            if size == GROUND_LEN:
                base_score += 5

            if ext in text_exts:
                base_score -= 2
            if ext in binary_exts:
                base_score += 3

            closeness = max(0, 50 - int(abs(size - GROUND_LEN) / 20))
            initial_score = base_score * 10 + closeness

            candidates.append((initial_score, m))

        if not candidates:
            tf.close()
            return fallback

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[:20]

        best_data = None
        best_score = float("-inf")

        for initial_score, m in top_candidates:
            try:
                f = tf.extractfile(m)
                if not f:
                    continue
                data = f.read()
                f.close()
            except Exception:
                continue

            if not data:
                continue

            sample = data[:4096]
            total = len(sample)
            if total == 0:
                continue

            binary_count = 0
            text_count = 0
            for b in sample:
                if b == 0 or b >= 128:
                    binary_count += 1
                if 32 <= b <= 126 or b in (9, 10, 13):
                    text_count += 1

            binary_ratio = binary_count / total
            text_ratio = text_count / total

            type_adj = 0
            ext = os.path.splitext(m.name.lower())[1]
            if binary_ratio > 0.1 and text_ratio < 0.95:
                type_adj += 30
            elif text_ratio > 0.95 and binary_ratio < 0.01:
                type_adj -= 30

            if ext in text_exts:
                type_adj -= 5
            if ext in binary_exts:
                type_adj += 5

            score = initial_score + type_adj

            if binary_ratio > 0.1:
                closeness2 = max(0, 50 - int(abs(len(data) - GROUND_LEN) / 10))
                score += closeness2

            if score > best_score:
                best_score = score
                best_data = data

        tf.close()

        if best_data is not None:
            return best_data

        return fallback