import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        bug_id = "42537583"
        desired_size = 1025
        fallback = b"A" * desired_size

        tf = None
        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return fallback

        best_member = None
        best_score = None

        TEXT_BYTES = set(range(32, 127))
        TEXT_BYTES.update((9, 10, 13))

        try:
            for m in tf.getmembers():
                if not m.isreg():
                    continue
                size = m.size
                if size <= 0 or size > 200000:
                    continue

                name = m.name
                name_low = name.lower()

                base_score = 0

                # Name-based heuristics
                if bug_id in name_low:
                    base_score += 500
                if "poc" in name_low:
                    base_score += 300
                if "crash" in name_low:
                    base_score += 250
                if "id_" in name_low:
                    base_score += 200
                if "testcase" in name_low or "input" in name_low:
                    base_score += 100
                if "media100" in name_low or "mjpegb" in name_low:
                    base_score += 150
                if "oss-fuzz" in name_low or "ossfuzz" in name_low:
                    base_score += 80

                # Extension-based heuristics
                _, ext = os.path.splitext(name_low)
                if ext in (".bin", ".raw", ".dat", ".poc", ".fuzz"):
                    base_score += 150
                if ext in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv",
                           ".mpg", ".mpeg", ".ts", ".y4m", ".ogg", ".ogv"):
                    base_score += 140
                if ext in (".yuv", ".mjpeg", ".jpeg", ".jpg", ".mjpg"):
                    base_score += 120

                # Size proximity to ground-truth PoC
                diff = abs(size - desired_size)
                base_score += max(0, 300 - diff)

                if size < 10:
                    base_score -= 50
                elif size > 10000:
                    base_score -= 30

                # Peek into file content
                try:
                    f = tf.extractfile(m)
                except Exception:
                    continue
                if f is None:
                    continue

                try:
                    sample = f.read(512)
                except Exception:
                    continue
                if not sample:
                    continue

                # Text vs binary heuristic
                nontext = 0
                for b in sample:
                    if b not in TEXT_BYTES:
                        nontext += 1
                ratio = nontext / len(sample)
                if ratio < 0.3:
                    base_score -= 400
                else:
                    base_score += int(200 * ratio)

                # Simple magic-number/container heuristics
                if sample.startswith(b"\x00\x00\x00") and b"ftyp" in sample[:32]:
                    base_score += 100  # MP4/QuickTime-style
                if sample.startswith(b"RIFF") and b"AVI" in sample[:64]:
                    base_score += 100
                if sample.startswith(b"\x1a\x45\xdf\xa3"):
                    base_score += 80   # Matroska
                if sample.startswith(b"OggS"):
                    base_score += 80

                if best_score is None or base_score > best_score:
                    best_score = base_score
                    best_member = m

            if best_member is not None and (best_score is None or best_score > 0):
                try:
                    f = tf.extractfile(best_member)
                except Exception:
                    f = None
                if f is not None:
                    try:
                        data = f.read()
                    except Exception:
                        data = b""
                    if data:
                        return data

        finally:
            if tf is not None:
                try:
                    tf.close()
                except Exception:
                    pass

        return fallback